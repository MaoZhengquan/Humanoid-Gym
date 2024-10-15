# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import math
import torch
from humanoid.envs import LeggedRobot
from humanoid.utils.terrain import HumanoidTerrain
from humanoid.utils.calculate_gait import get_coefficients
from collections import deque
import random


class GR(LeggedRobot):
    '''
    XBotLFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.feet_vel_z = torch.zeros_like(self.rigid_state[:, self.feet_indices, 9])
        self.last_feet_vel_z = torch.zeros_like(self.rigid_state[:, self.feet_indices, 9])
        self.foot_height_max = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                           device=self.device, requires_grad=False)
        self.foot_height_max_buffer = torch.zeros_like(self.foot_height_max)
        self.compute_observations()

        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
                """
        push_curriculum_scale = np.clip((self.step_count - self.cfg.domain_rand.push_curriculum_start_step) / self.cfg.domain_rand.push_curriculum_common_step,
                                        0, 1)
        if push_curriculum_scale > 0.:
            print("------push robot at step:", self.step_count)
        max_lin_vel = self.cfg.domain_rand.max_push_vel_xy * push_curriculum_scale
        max_ang_vel = self.cfg.domain_rand.max_push_vel_ang * push_curriculum_scale
        self.root_states[:, 7:9] += torch_rand_float(-max_lin_vel, max_lin_vel, (self.num_envs, 2),
                                                     device=self.device)  # lin vel x/y
        self.root_states[:, 10:] += torch_rand_float(-max_ang_vel, max_ang_vel, (self.num_envs, 3),
                                                     device=self.device)  # ang vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask

    def compute_ref_state(self):
        phase = self._get_phase()
        # print("phase_state",phase[0])
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        mask_right = (torch.floor(phase) + 1) % 2
        mask_left = torch.floor(phase) % 2
        cos_pos = (1 - torch.cos(2 * torch.pi * phase)) / 2 # 得到一条从0开始增加，频率为step_freq，振幅0～1的曲线，接地比较平滑
        self.cos_pos[:, 0] = cos_pos * mask_left
        self.cos_pos[:, 1] = cos_pos * mask_right
        self.ref_mask[:, 0] = mask_left
        self.ref_mask[:, 1] = mask_right
        scale_1 = self.cfg.commands.step_joint_offset
        scale_2 = 2 * scale_1
        scale_3 = 0.5 * scale_1
        self.ref_dof_pos[:, :] = self.default_dof_pos[0, :]
        # right foot stance phase set to default joint pos
        # left foot stance phase set to default joint pos

        # self.ref_dof_pos[:, 2] += -self.cos_pos[:, 0] * scale_1
        # self.ref_dof_pos[:, 3] += self.cos_pos[:, 0] * 2 * scale_1
        # self.ref_dof_pos[:, 4] += -self.cos_pos[:, 0] * scale_1
        # left foot stance phase set to default joint pos
        # sin_pos_l[sin_pos_l > 0] = 0
        # self.ref_dof_pos[:, 8] += -self.cos_pos[:, 1] * scale_1
        # self.ref_dof_pos[:, 9] += self.cos_pos[:, 1] * 2 * scale_1
        # self.ref_dof_pos[:, 10] += -self.cos_pos[:, 1] * scale_1
        # self.ref_dof_pos[torch.abs(cos_pos) < 0.1] = self.default_dof_pos
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1 + self.cfg.init_state.default_joint_angles['left_hip_pitch_joint']
        self.ref_dof_pos[:, 3] = -sin_pos_l * scale_2 + self.cfg.init_state.default_joint_angles['left_knee_pitch_joint']
        # self.ref_dof_pos[:, 4] = sin_pos_l * scale_3 + self.cfg.init_state.default_joint_angles['left_ankle_pitch_joint']
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 7] = -sin_pos_r * scale_1 + self.cfg.init_state.default_joint_angles['right_hip_pitch_joint']
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_2 + self.cfg.init_state.default_joint_angles['right_knee_pitch_joint']
        # self.ref_dof_pos[:, 9] = -sin_pos_r * scale_3 + self.cfg.init_state.default_joint_angles['right_ankle_pitch_joint']
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = self.default_dof_pos

        # print("ref_pos", self.ref_dof_pos[0, [2, 8]])
        self.ref_action = 2 * self.ref_dof_pos

    def compute_foot_traj(self):
        phase = self._get_phase() * self.cfg.rewards.cycle_time
        # print("phase_traj",phase[0])
        mask_right = (torch.floor(phase) + 1) % 2
        mask_left = torch.floor(phase) % 2
        cos_pos = (1 - torch.cos(2 * torch.pi * phase)) / 2  # 得到一条从0开始增加，频率为step_freq，振幅0～1的曲线，接地比较平滑

        swing_time = self.cfg.rewards.swing_time
        phase = phase % swing_time
        coeffs = get_coefficients(0,0,0.05,0,0.05,swing_time,8)
        a5,a4,a3,a2,a1,a0 = coeffs
        a5,a4,a3,a2,a1,a0 = coeffs
        stance_mask = self._get_gait_phase() # 1 stance 0 swing
        swing_mask = 1 - stance_mask
        self.ref_foot_height = torch.zeros_like(stance_mask)
        self.ref_foot_vel = torch.zeros_like(stance_mask)
        self.ref_foot_acc = torch.zeros_like(stance_mask)
        foot_height = a5 * phase**5 + a4 * phase**4 + a3 * phase**3 + a2 * phase**2 + a1 * phase + a0
        foot_vel = 5 * a5 * phase**4 + 4 * a4 * phase**3 + 3 * a3 * phase**2 + 2 * a2 * phase + a1
        foot_acc = (20 * a5 * phase**3 + 12 * a4 * phase**2 + 6 * a3 * phase + 2 * a2) / 50
        foot_height = foot_height.unsqueeze(1)  # 4096, ——>  4096,1
        foot_vel = foot_vel.unsqueeze(1)
        foot_acc = foot_acc.unsqueeze(1)
        foot_height = foot_height * swing_mask
        foot_vel = foot_vel * swing_mask
        foot_acc = foot_acc * swing_mask
        self.ref_foot_height = foot_height
        self.ref_foot_vel = foot_vel
        # print("mask",self.ref_mask[0])
        # print("actual_height",self.rigid_state[0, self.feet_indices, 2])
        # print("foot_height", self.ref_foot_height[0])
        # print("actual_vel",self.rigid_state[0, self.feet_indices, 9])
        # print("foot_vel",self.ref_foot_vel[0])
        # print("===============================")
        self.ref_foot_acc = foot_acc
        # print("foot_height", self.ref_foot_height[0])
        # print("foot_vel", self.ref_foot_vel[0])
        # print("========================================")
        return foot_height

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """

        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 5] = 0.  # commands

        noise_vec[5:15] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[15:25] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[25:28] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[28:31] = noise_scales.quat * self.obs_scales.quat
        noise_vec[31:41] = 0

        return noise_vec

    def clip_actions(self, actions):
        clip_actions_max = torch.tensor(self.cfg.normalization.clip_actions_max).to(torch.float32).to(self.device)
        clip_actions_min = torch.tensor(self.cfg.normalization.clip_actions_min).to(torch.float32).to(self.device)

        actions_cliped = torch.clip(actions, clip_actions_min, clip_actions_max).to(self.device)
        return actions_cliped

    def step(self, actions):
        # 从on_policy_runner进来的action，刚从act获取
        # 步态生成
        # time.sleep(0.10)
        self.ref_count += 1
        self.step_count += 1
        if self.cfg.env.use_ref_actions:
            actions += self.ref_dof_pos
        clip_actions = self.cfg.normalization.clip_actions
        # clip_actions_max = torch.tensor(self.cfg.normalization.clip_actions_max).to(torch.float32).to(self.device)
        # clip_actions_min = torch.tensor(self.cfg.normalization.clip_actions_min).to(torch.float32).to(self.device)
        # actions = torch.clip(actions, clip_actions_min, clip_actions_max).to(self.device)
        actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.actions = actions.clone()
        # self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # 下行延迟：延长将action送往扭矩的时间
        self.action_history.appendleft(self.actions.clone())
        index_act = random.randint(self.act_latency_rng[0], self.act_latency_rng[1])  # ms
        if 0 <= index_act < 10:
            action_delayed_0 = self.action_history[0]
            action_delayed_1 = self.action_history[1]
        elif 10 <= index_act < 20:
            action_delayed_0 = self.action_history[1]
            action_delayed_1 = self.action_history[2]
        elif 20 <= index_act < 30:
            action_delayed_0 = self.action_history[2]
            action_delayed_1 = self.action_history[3]
        elif 30 <= index_act < 40:
            action_delayed_0 = self.action_history[3]
            action_delayed_1 = self.action_history[4]
        elif 40 <= index_act < 50:
            action_delayed_0 = self.action_history[4]
            action_delayed_1 = self.action_history[5]
        else:
            raise ValueError

        action_delayed = action_delayed_0 + (action_delayed_1 - action_delayed_0) * torch.rand(self.num_envs, 1,
                                                                                               device=self.sim_device)

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.sim_step_last_dof_pos = self.dof_pos.clone()
            self.torques = self._compute_torques(action_delayed).view(self.torques.shape)
            if self.cfg.domain_rand.randomize_motor_strength:
                rng = self.cfg.domain_rand.scaled_motor_strength_range
                strength_scale = rng[0] + (rng[1] - rng[0]) * torch.rand(self.num_envs, self.num_dofs,
                                                                         device=self.sim_device)  # randomize 10% torque error
            else:
                strength_scale = 1.0
            randomized_torques = torch.clip(self.torques * strength_scale, -self.torque_limits, self.torque_limits)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(randomized_torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.dof_vel = (self.dof_pos - self.sim_step_last_dof_pos) / self.sim_params.dt

        self.post_physics_step()

        # 上行延迟：延迟获取obs,但观察到当前帧的action。
        self.obs_history.appendleft(self.obs_buf.clone())
        index_obs = random.randint(self.obs_latency_rng[0], self.obs_latency_rng[1])  # ms
        if 0 <= index_obs < 10:
            obs_delayed_0 = self.obs_history[0]
            obs_delayed_1 = self.obs_history[1]
        elif 10 <= index_obs < 20:
            obs_delayed_0 = self.obs_history[1]
            obs_delayed_1 = self.obs_history[2]
        elif 20 <= index_obs < 30:
            obs_delayed_0 = self.obs_history[2]
            obs_delayed_1 = self.obs_history[3]
        elif 30 <= index_obs < 40:
            obs_delayed_0 = self.obs_history[3]
            obs_delayed_1 = self.obs_history[4]
        elif 40 <= index_obs < 50:
            obs_delayed_0 = self.obs_history[4]
            obs_delayed_1 = self.obs_history[5]
        else:
            raise ValueError

        obs_delayed = obs_delayed_0 + (obs_delayed_1 - obs_delayed_0) * torch.rand(self.num_envs, 1,
                                                                                   device=self.sim_device)
        self.obs_buf[:, 5:-self.num_actions] = obs_delayed[:, 5:self.num_obs - self.num_actions]

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

        # if self.cfg.env.use_ref_actions:
        #     actions += self.ref_action
        # # dynamic randomization
        # delay = torch.rand((self.num_envs, 1), device=self.device)
        # actions = (1 - delay) * actions + delay * self.actions
        # actions += self.cfg.domain_rand.dynamic_randomization * toniuniurch.randn_like(actions) * actions
        # return super()guaibu.step(actions)

    def compute_observations(self):
        phase = self._get_phase()
        stance_mask = self._get_gait_phase()
        # print("stance_mask", stance_mask[0])
        # print("ref_mask", self.ref_mask[0])
        # print("ref_count", self.ref_count[0])
        # print("====================================")

        self.compute_ref_state()
        self.compute_foot_traj()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        # self.surround_heights_offset = \
        #         torch.clip(self.root_states[:, 2].unsqueeze(1)
        #                - self.cfg.rewards.base_height_target
        #                - self.measured_heights,
        #                min=-1.0,
        #                max=1.0) \
        #     * self.obs_scales.height_measurements
        # print("surround_heights_offset",self.surround_heights_offset)
        # print("measured_heights",self.measured_heights)
        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        diff = self.dof_pos - self.ref_dof_pos

        self.obs_buf = torch.cat((
            sin_pos,
            cos_pos,
            self.commands[:, :3],  # 3D(vel_x, vel_y, aug_vel_yaw)
            q,  # 12D
            dq,  # 12D
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.actions,  # 12D
        ), dim=-1)

        self.feet_vel = self.rigid_state[:, self.feet_indices, 7:10].view(-1, 6)
        foot_pos = self.rigid_state[:, self.feet_indices, :3].view(-1, 6)
        cycle_time_tensor = torch.full((self.num_envs, 1), self.cfg.rewards.cycle_time, device=self.device)
        self.feet_input = torch.cat((foot_pos, self.feet_vel), dim=-1, )
        torque = self._compute_torques(actions=self.actions)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        else:
            self.measured_heights = torch.zeros((self.num_envs, 96), device=self.device)
        # print(self.obs_buf.size())
        # print(self.base_lin_vel.size())
        # print(self.friction_coeffs_tensor.size())
        # print(self.rand_push_force.size())
        # print(self.rand_push_torque.size())
        # print(cycle_time_tensor.size())
        # print(self.ref_mask.size())
        # print(self.feet_input.size())
        # print(contact_mask.size())
        # print(self.mass_params_tensor.size())
        # print(self.rew_buf.size())
        # print(torque.size())
        # print(self.measured_heights.size())

        self.privileged_obs_buf = torch.cat((
            self.obs_buf, # 41
            self.base_lin_vel * self.obs_scales.lin_vel, #3
            self.friction_coeffs_tensor, # 1
            self.rand_push_force, # 3
            self.rand_push_torque, # 3
            cycle_time_tensor,# 1
            self.ref_mask, # 2
            self.feet_input, # 2
            contact_mask,  # 2
            self.mass_params_tensor, # 4
            self.rew_buf.view(-1, 1), # 1
            torque, # 10
            self.measured_heights # 96
        ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        if torch.isnan(self.obs_buf).any():
            self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0001)

        self.obs_storage_history.append(self.obs_buf)
        self.critic_storage_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_storage_history[i]
                                   for i in range(self.obs_storage_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_storage_history[i] for i in range(self.cfg.env.c_frame_stack)],
                                            dim=1)
    def _refresh_gym_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        return

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids].clone()
        self.last_dof_vel[env_ids] = self.dof_vel[env_ids].clone()
        # 在这里重置重设指令的步态生成器的初始计数值
        self.ref_count[env_ids] = 0
        # self.ref_freq[env_ids] = self.cfg.commands.step_freq
        # self.ref_count[env_ids] = torch.randint(0, 5, (torch.numel(env_ids),), device=self.sim_device)  # randomize phase ahead with 0~100 ms

        # update all obs_buffer with new reset state
        self._refresh_gym_tensors()
        # self.compute_observations()

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] = self.obs_buf.clone()[env_ids]
        for i in range(self.action_history.maxlen):
            self.action_history[i][env_ids] *= 0.0

    # ================================================ Rewards ================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        reward1 = torch.sum(torch.where(abs(diff[:, 2:5]) < 0.02, 1.0, -0.3), dim=-1)
        reward2 = torch.sum(torch.where(abs(diff[:, 8:11]) < 0.02, 1.0, -0.3), dim=-1)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    #
    #
    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        # print("contact", self.contact_forces[:, self.feet_indices, 2])
        stance_mask = 1 - self.ref_mask
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.32) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    #
    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        # stance_mask = self._get_gait_phase()
        reward = torch.where(contact == self.ref_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    #
    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 30)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        # print("feet_contact_forces",self.contact_forces[:, self.feet_indices, :])
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_roll = joint_diff[:,:2]
        right_roll = joint_diff[:,5:7]
        yaw_roll = torch.norm(left_roll, dim=1) + torch.norm(right_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_default_joint_roll_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_roll = joint_diff[:,0]
        right_roll = joint_diff[:,5]
        yaw_roll = torch.norm(left_roll, dim=0) + torch.norm(right_roll, dim=0)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 30) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_default_joint_yaw_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw = joint_diff[:,1]
        right_yaw = joint_diff[:,6]
        yaw_roll = torch.norm(left_yaw, dim=0) + torch.norm(right_yaw, dim=0)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 30) - 0.01 * torch.norm(joint_diff, dim=1)


    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height
        of its feet when they are in contact with the ground.
        """
        stance_mask = 1 - self.ref_mask
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities.
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    #
    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes.
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.lin_tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """

        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.ang_tracking_sigma)

    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05

        # print("feet_z{}\n".format(feet_z[0]))
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * self.ref_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    #
    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    #
    def _reward_period_force(self):
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5
        stance_mask = 1 - self.ref_mask
        # actual_mask = contact_mask == stance_mask
        contact = self.contact_forces[:, self.feet_indices, 2]  # stance 1 swing 0
        reward = torch.sum(torch.where(contact_mask == stance_mask, 0.5, -1.0), dim=1)
        # reward = torch.sum(contact * actual_mask,dim=1) / torch.sum(stance_mask,dim=1)
        # rew_force = (reward / self.cfg.rewards.max_contact_force).clip(min=0., max=1.)
        return reward

    #
    #
    def _reward_period_vel(self):
        # contact = self.contact_forces[:,self.feet_indices,2] > 5
        stance_mask = self.ref_mask
        # actual_mask = (~contact) == swing_mask  # 摆腿阶段的掩码
        feet_speed = self.rigid_state[:, self.feet_indices, 9]
        feet_speed = torch.where(feet_speed.abs() < 0.1, torch.tensor(0.0, device=self.device),
                                 torch.tensor(1.0, device=self.device))
        reward = torch.sum(torch.where(feet_speed == stance_mask, 0.5, -1.0), dim=1)
        # reward = torch.sum(feet_speed * actual_mask,dim=1)
        return reward

    #
    #
    def _reward_foot_height_tracking(self):
        foot_height = (self.rigid_state[:, self.feet_indices, 2]) - 0.05  # 两脚实际的
        ref_foot_height = self.ref_foot_height.clone()  # 两脚理想的
        diff = abs(foot_height - ref_foot_height)
        reward = torch.sum(torch.where(diff < 0.02, 0.2, -diff)) * 0.001
        # diff = torch.square(torch.sum(foot_height - ref_foot_height,dim=1))
        # return torch.exp(-diff * self.cfg.rewards.tracking_feet_height_sigma)
        return reward

    def _reward_foot_vel_tracking(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 50.
        self.feet_vel_z = self.rigid_state[:, self.feet_indices, 9]
        contact_vel_error = torch.abs(torch.sum(contact * self.feet_vel_z, dim=-1))
        swing_mask = 1 - self._get_gait_phase()
        tracking_reward = torch.sum(torch.square(self.ref_foot_vel * swing_mask - self.ref_foot_vel * 0.3), dim=1)
        return torch.exp(-tracking_reward * self.cfg.rewards.tracking_feet_vel_sigma) - 0.02 * contact_vel_error

    #
    #     def _reward_large_contact(self):
    #         """
    #         Calculates the reward for keeping contact forces within a specified range. Penalizes
    #         high contact forces on the feet.
    #         """
    #         # print("feet_contact_forces",self.contact_forces[:, self.feet_indices, :])
    #         return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)
    #
    #     def _reward_default_joint(self):
    #         """
    #         Calculates the reward for keeping joint positions close to default positions, with a focus
    #         on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
    #         """
    #         joint_diff = self.dof_pos - self.default_joint_pd_target
    #         left_yaw_roll = joint_diff[:, :2]
    #         right_yaw_roll = joint_diff[:, 6: 8]
    #         yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
    #         yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
    #         return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
    #
    #
    #     def _reward_energy_cost(self):
    #         # print("tau",torch.sum(torch.norm(self.torques ,dim=-1)))
    #         # print("vel",torch.sum(torch.norm(self.dof_vel ,dim=-1)))
    #         # print("energy cost",torch.sum(torch.square(self.torques * self.dof_vel),dim=1))
    #         return torch.sum(torch.square(self.torques * self.dof_vel),dim=1)
    #
    #     def _reward_feet_movements(self):
    #         self.feet_vel_z = self.rigid_state[:,self.feet_indices,9]
    #         self.feet_acc_z = (self.feet_vel_z - self.last_feet_vel_z) / self.dt
    #         self.last_feet_vel_z = self.feet_vel_z
    #         term1 = torch.norm(self.feet_vel_z - self.ref_foot_vel, dim=1)
    #         term2 = torch.norm(self.feet_acc_z - self.ref_foot_acc, dim=1)
    #         return term1 + term2

    "Zhong Qing"

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 1.1
        single_contact = torch.sum(1. * contacts, dim=1) == 1
        return 1. * single_contact

    def _reward_target_joint_pos_l(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        # joint_diff_r = torch.sum((self.dof_pos[:, 0:6] - self.ref_dof_pos[:, 0:6]) ** 2, dim=1)
        joint_diff_l = torch.sum((self.dof_pos[:, 6:12] - self.ref_dof_pos[:, 6:12]) ** 2, dim=1)
        # imitate_reward = torch.exp(-7*(joint_diff_r + joint_diff_l))  # positive reward, not the penalty
        imitate_reward = torch.exp(-7 * joint_diff_l)
        return imitate_reward

    def _reward_target_joint_pos_r(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff_r = torch.sum((self.dof_pos[:, 0:6] - self.ref_dof_pos[:, 0:6]) ** 2, dim=1)
        # joint_diff_l = torch.sum((self.dof_pos[:, 6:12] - self.ref_dof_pos[:, 6:12]) ** 2, dim=1)
        imitate_reward = torch.exp(-7 * joint_diff_r)  # positive reward, not the penalty
        return imitate_reward

    # def _reward_orientation(self):
    #     # positive reward non flat base orientation
    #     return torch.exp(-10. * torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1))

    def _reward_tracking_lin_x_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_y_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.abs(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    # def _reward_body_feet_dist(self):
    #     # Penalize body root xy diff feet xy
    #     self.gym.clear_lines(self.viewer)
    #
    #     # foot_pos = self.rigid_state[:, self.feet_indices, :3]
    #     # center_pos = torch.mean(foot_pos, dim=1)
    #     self.body_pos[:, :3] = self.root_states[:, :3]
    #     # self.body_pos[:, :3] = self.init_position[:, :3]
    #     # self.body_pos[:, 2] -= 0.75
    #
    #     pos_dist = torch.norm(self.body_pos[:, :] - self.init_position[:, :], dim=1)
    #
    #     sphere_geom_1 = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
    #     sphere_geom_2 = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 0, 1))
    #     sphere_pose_1 = gymapi.Transform(gymapi.Vec3(self.body_pos[0, 0], self.body_pos[0, 1], self.body_pos[0, 2] - 0.7), r=None)
    #     sphere_pose_2 = gymapi.Transform(gymapi.Vec3(self.init_position[0, 0], self.init_position[0, 1], self.init_position[0, 2] - 0.7), r=None)
    #     # sphere_pose_2 = gymapi.Transform(gymapi.Vec3(center_pos[0, 0], center_pos[0, 1], center_pos[0, 2]), r=None)
    #
    #     gymutil.draw_lines(sphere_geom_1, self.gym, self.viewer, self.envs[0], sphere_pose_1)
    #     gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[0], sphere_pose_2)
    #
    #     reward = torch.square(pos_dist * 10)
    #     # print(f'dist={pos_dist[0]}')
    #     return reward

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_ankle_action_rate(self):
        # Penalize changes in ankle actions
        diff1 = self.last_actions[:, 4:6] - self.actions[:, 4:6]
        diff2 = self.last_actions[:, 10:] - self.actions[:, 10:]
        return torch.sum(torch.abs(diff1) + torch.abs(diff2), dim=1)

    def _reward_ankle_dof_acc(self):
        # Penalize ankle dof accelerations
        diff1 = self.last_dof_vel[:, 4:6] - self.dof_vel[:, 4:6]
        diff2 = self.last_dof_vel[:, 10:] - self.dof_vel[:, 10:]
        return torch.sum(torch.abs(diff1 / self.dt) + torch.abs(diff2 / self.dt), dim=1)

    def _reward_target_ankle_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        diff = torch.abs(self.dof_pos[:, 4:6] - self.ref_dof_pos[:, 4:6])
        diff += torch.abs(self.dof_pos[:, 10:] - self.ref_dof_pos[:, 10:])
        ankle_imitate_reward = torch.exp(-20 * torch.sum(diff, dim=1))  # positive reward, not the penalty
        return ankle_imitate_reward

    def _reward_target_hip_roll_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
         on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        diff = torch.abs(self.dof_pos[:, 0] - self.ref_dof_pos[:, 0])
        diff += torch.abs(self.dof_pos[:, 6] - self.ref_dof_pos[:, 6])
        roll_imitate_reward = torch.exp(-15 * diff)  # positive reward, not the penalty
        return roll_imitate_reward

    def _reward_peroid_force(self):
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 10
        stance_mask = self._get_gait_phase()

        # actual_mask = contact_mask == stance_mask
        reward = torch.sum(torch.where(contact_mask == stance_mask, 0.5, -1.0), dim=1)
        return reward
        # stance_mask = self._get_gait_phase()
        # force_norm = torch.norm(self.contact_forces[:, self.feet_indices, 0:3], dim=-1)
        # scale_force = torch.sum(
        #     ((force_norm * stance_mask) / self.cfg.rewards.max_contact_force).clip(0, 1), dim=1)
        # return scale_force


    def _reward_peroid_vel(self):
        # contact = self.contact_forces[:,self.feet_indices,2] > 5
        swing_mask = 1 - self._get_gait_phase()
        feet_speed = self.rigid_state[:, self.feet_indices, 9]
        feet_speed = torch.where(feet_speed.abs() < 0.1, torch.tensor(0.0, device=self.device),
                                 torch.tensor(1.0, device=self.device))
        reward = torch.sum(torch.where(feet_speed == swing_mask, 0.5, -1.0), dim=1)
        return reward