from inspect import stack

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from legged_gym.gym_utils.helpers import class_to_dict
from legged_gym import LEGGED_GYM_ROOT_DIR, POSE_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.humanoid import Humanoid
# from .humanoid_config import HumanoidCfg
from .gr1_explicit_config import GR1_explicitCfg
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.gym_utils.terrain_x1 import Terrain
from legged_gym.gym_utils.math import *
from legged_gym.gym_utils.motor_delay_fft import MotorDelay_130, MotorDelay_80


class GR1_explicit(Humanoid):
    '''
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
        terrain (Terrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_stance_mask(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''

    def __init__(self, cfg: GR1_explicitCfg, sim_params, physics_engine, sim_device, headless):
        self.control_index = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
        self.cfg = cfg
        self.use_motor_model = self.cfg.env.use_motor_model
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        phase = self._get_phase()
        self.compute_ref_state()

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_xy
        self.rand_push_force[:, :2] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device) # angular vel xyz
        self.root_states[:, 10:13] = self.rand_push_torque
        # self.root_states[:, 7:9] = -3.5 + 0*torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        # self.root_states[:, 8] = 0
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        if self.cfg.commands.sw_switch:
            stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
            # print("stand_command", stand_command)
            self.phase_length_buf[stand_command] = 0 # set this as 0 for which env is standing
            phase = (self.phase_length_buf * self.dt / cycle_time + self.gait_start) * (~stand_command)
        else:
            phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_stance_mask(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        stance_mask = torch.zeros((self.num_envs, 2),device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Add double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        # stand mask == 1 means stand leg
        return stance_mask

    def generate_gait_time(self, envs):
        if len(envs) == 0:
            return

        # rand sample
        random_tensor_list = []
        for i in range(len(self.cfg.commands.gait)):
            name = self.cfg.commands.gait[i]
            gait_time_range = self.cfg.commands.gait_time_range[name]
            random_tensor_single = torch_rand_float(gait_time_range[0],
                                                    gait_time_range[1],
                                                    (len(envs), 1), device=self.device)
            random_tensor_list.append(random_tensor_single)
        random_tensor = torch.cat([random_tensor_list[i] for i in range(len(self.cfg.commands.gait))],dim=1)
        current_sum = torch.sum(random_tensor, dim=1, keepdim=True) # 计算所有步态的总和时间
        # led_tensor store proportion for each gait type
        scaled_tensor = random_tensor * (self.max_episode_length / current_sum) # 随机生成的时间标准化 总和等于max_episode_length
        scaled_tensor[:, 1:] = scaled_tensor[:, :-1].clone() # 标准化后的时间进行位移处理
        scaled_tensor[:,0] *= 0.0 # 第一个步态设置为0
        # self.gait_time accumulate gait_duration_tick
        # self.gait_time = |__gait1__|__gait2__|__gait3__|
        # self.gait_time triger resample gait command
        # 每个环境在每种步态下的具体持续时间
        self.gait_time[envs] = torch.cumsum(scaled_tensor, dim=1).int()

    def _resample_commands(self):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        for i in range(len(self.cfg.commands.gait)):
            env_ids = (self.episode_length_buf == self.gait_time[:,i]).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                # according to gait type create a name
                name = '_resample_' + self.cfg.commands.gait[i] + '_command'
                # get function from self based on name
                resample_command = getattr(self, name)
                resample_command(env_ids)


    def _resample_stand_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)
        command_x = self.commands[:, 0]
        count = torch.sum(command_x == 0).item()

    def _resample_walk_sagittal_command(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_walk_lateral_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_rotate_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

    def _resample_walk_omnidirectional_command(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.05).unsqueeze(1)
        command_x = self.commands[:, 0]
        count = torch.sum(command_x == 0).item()

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        self.phase_length_buf += 1
        self._resample_commands()
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            # get all robot surrounding height
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots:
            i = int(self.common_step_counter / self.cfg.domain_rand.update_step)
            if i >= len(self.cfg.domain_rand.push_duration):
                i = len(self.cfg.domain_rand.push_duration) - 1
            duration = self.cfg.domain_rand.push_duration[i] / self.dt
            if self.common_step_counter % self.cfg.domain_rand.push_interval <= duration:
                self._push_robots()
            else:
                self.rand_push_torque.zero_()
                self.rand_push_force.zero_()

    def compute_ref_state(self):
        phase = self._get_phase()
        _sin_pos_l = torch.sin(2 * torch.pi * phase)
        _sin_pos_r = torch.sin(2 * torch.pi * phase + torch.pi)
        sin_pos_l = _sin_pos_l.clone()
        sin_pos_r = _sin_pos_r.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        ratio_l = torch.clamp(torch.abs(sin_pos_l) - self.cfg.rewards.double_support_threshold, min=0, max=1) / \
                    (1 - self.cfg.rewards.double_support_threshold) * torch.sign(sin_pos_l)
        self.ref_dof_pos[:, 2] = ratio_l * scale_1
        self.ref_dof_pos[:, 3] = -ratio_l * scale_2
        self.ref_dof_pos[:, 4] = ratio_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r > 0] = 0
        ratio_r = torch.clamp(torch.abs(sin_pos_r) - self.cfg.rewards.double_support_threshold, min=0, max=1) / (
                    1 - self.cfg.rewards.double_support_threshold) * torch.sign(sin_pos_r)
        self.ref_dof_pos[:, 8] = ratio_r * scale_1
        self.ref_dof_pos[:, 9] = -ratio_r * scale_2
        self.ref_dof_pos[:, 10] = ratio_r * scale_1
        # Double support phase
        indices = (torch.abs(sin_pos_l) < self.cfg.rewards.double_support_threshold) & (
                    torch.abs(sin_pos_r) < self.cfg.rewards.double_support_threshold)
        self.ref_dof_pos[indices] = 0
        self.ref_dof_pos += self.default_dof_pos_all
        self.ref_action = 2 * self.ref_dof_pos

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel_exp"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales["tracking_lin_vel_exp"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.25,
                                                          -self.cfg.commands.max_curriculum / 2, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
                                                          self.cfg.commands.max_curriculum)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _init_buffers(self):
        super()._init_buffers()
        self.gait_time = torch.zeros(self.num_envs, len(self.cfg.commands.gait), dtype=torch.int, device=self.device,
                                     requires_grad=False)
        self.phase_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.gait_start = torch.randint(0, 2, (self.num_envs,)).to(self.device)*0.5
        if self.use_motor_model:
            self.motordelay0 = MotorDelay_80(self.num_envs, 1, device=self.device)
            self.motordelay6 = MotorDelay_80(self.num_envs, 1, device=self.device)

            self.motordelay2 = MotorDelay_130(self.num_envs, 1, device=self.device)
            self.motordelay3 = MotorDelay_130(self.num_envs, 1, device=self.device)
            self.motordelay8 = MotorDelay_130(self.num_envs, 1, device=self.device)
            self.motordelay9 = MotorDelay_130(self.num_envs, 1, device=self.device)

            self.fric_para_0 = torch.zeros(self.num_envs, 5,
                                           dtype=torch.float, device=self.device, requires_grad=False)
            self.fric_para_1 = torch.zeros(self.num_envs, 5,
                                           dtype=torch.float, device=self.device, requires_grad=False)
            self.fric_para_2 = torch.zeros(self.num_envs, 5,
                                           dtype=torch.float, device=self.device, requires_grad=False)
            self.fric_para_3 = torch.zeros(self.num_envs, 5,
                                           dtype=torch.float, device=self.device, requires_grad=False)
            self.fric = torch.zeros(self.num_envs, self.num_dofs,
                                    dtype=torch.float, device=self.device, requires_grad=False)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        start = time()
        print("*"*80)
        print("Start creating ground...")
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)

        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*" * 80)
        self._create_envs()

    def _get_body_indices(self):
        upper_arm_names = [s for s in self.body_names if self.cfg.asset.upper_arm_name in s]
        lower_arm_names = [s for s in self.body_names if self.cfg.asset.lower_arm_name in s]
        torso_name = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_name), dtype=torch.long, device=self.device,
                                         requires_grad=False)
        for j in range(len(torso_name)):
            self.torso_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                          torso_name[j])
        self.upper_arm_indices = torch.zeros(len(upper_arm_names), dtype=torch.long, device=self.device,
                                             requires_grad=False)
        for j in range(len(upper_arm_names)):
            self.upper_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                              upper_arm_names[j])
        self.lower_arm_indices = torch.zeros(len(lower_arm_names), dtype=torch.long, device=self.device,
                                             requires_grad=False)
        for j in range(len(lower_arm_names)):
            self.lower_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                              lower_arm_names[j])
        knee_names = [s for s in self.body_names if self.cfg.asset.shank_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         knee_names[i])

    def _reset_buffers_extra(self, env_ids):
        if self.use_motor_model:
            self._reset_fric_para(env_ids)
            self.motordelay0.reset(env_ids)
            self.motordelay6.reset(env_ids)
            self.motordelay2.reset(env_ids)
            self.motordelay3.reset(env_ids)
            self.motordelay8.reset(env_ids)
            self.motordelay9.reset(env_ids)

    def compute_observations(self):
        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_stance_mask()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale),dim=1)

        # critic no lag
        diff = self.dof_pos - self.ref_dof_pos
        # 73
        # print("base_lin_vel", self.base_lin_vel[0])
        # print("base_ang_vel", self.base_ang_vel[0])
        # print("base_euler_xyz", self.base_euler_xyz[0])


        privileged_obs_buf = torch.cat((
            self.command_input, # 2 + 3
            (self.dof_pos - self.default_joint_pd_target) * self.obs_scales.dof_pos, # 12
            self.dof_vel * self.obs_scales.dof_vel, # 12
            self.actions, # 10
            diff, # 12
            self.base_lin_vel * self.obs_scales.lin_vel, # 3
            self.base_ang_vel * self.obs_scales.ang_vel, # 3
            self.base_euler_xyz * self.obs_scales.imu, # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque, # 3
            self.env_frictions, # 1
            self.body_mass / 10., # 1 sum of all fix link mass
            stance_mask, # 2
            contact_mask, # 2
        ),dim=-1)

        # random add dof_pos and dof_vel same lag
        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0],
                                                      self.cfg.domain_rand.dof_lag_timesteps_range[1] + 1,
                                                      (self.num_envs,), device=self.device)
                cond = self.dof_lag_timestep > self.last_dof_lag_timestep + 1
                self.dof_lag_timestep[cond] = self.last_dof_lag_timestep[cond] + 1
                self.last_dof_lag_timestep = self.dof_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_lag_buffer[torch.arange(self.num_envs), :self.num_actions,
                                  self.dof_lag_timestep.long()]
            self.lagged_dof_vel = self.dof_lag_buffer[torch.arange(self.num_envs), -self.num_actions:,
                                  self.dof_lag_timestep.long()]
            # random add dof_pos and dof_vel different lag
        elif self.cfg.domain_rand.add_dof_pos_vel_lag:
            if self.cfg.domain_rand.randomize_dof_pos_lag_timesteps_perstep:
                self.dof_pos_lag_timestep = torch.randint(self.cfg.domain_rand.dof_pos_lag_timesteps_range[0],
                                                          self.cfg.domain_rand.dof_pos_lag_timesteps_range[1] + 1,
                                                          (self.num_envs,), device=self.device)
                cond = self.dof_pos_lag_timestep > self.last_dof_pos_lag_timestep + 1
                self.dof_pos_lag_timestep[cond] = self.last_dof_pos_lag_timestep[cond] + 1
                self.last_dof_pos_lag_timestep = self.dof_pos_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_pos_lag_buffer[torch.arange(self.num_envs), :,
                                  self.dof_pos_lag_timestep.long()]

            if self.cfg.domain_rand.randomize_dof_vel_lag_timesteps_perstep:
                self.dof_vel_lag_timestep = torch.randint(self.cfg.domain_rand.dof_vel_lag_timesteps_range[0],
                                                          self.cfg.domain_rand.dof_vel_lag_timesteps_range[1] + 1,
                                                          (self.num_envs,), device=self.device)
                cond = self.dof_vel_lag_timestep > self.last_dof_vel_lag_timestep + 1
                self.dof_vel_lag_timestep[cond] = self.last_dof_vel_lag_timestep[cond] + 1
                self.last_dof_vel_lag_timestep = self.dof_vel_lag_timestep.clone()
            self.lagged_dof_vel = self.dof_vel_lag_buffer[torch.arange(self.num_envs), :,
                                  self.dof_vel_lag_timestep.long()]
        # dof_pos and dof_vel has no lag
        else:
            self.lagged_dof_pos = self.dof_pos
            self.lagged_dof_vel = self.dof_vel

            # imu lag, including rpy and omega
            if self.cfg.domain_rand.add_imu_lag:
                if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                    self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0],
                                                          self.cfg.domain_rand.imu_lag_timesteps_range[1] + 1,
                                                          (self.num_envs,), device=self.device)
                    cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                    self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                    self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
                self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.int()]
                self.lagged_base_ang_vel = self.lagged_imu[:, :3].clone()
                self.lagged_base_euler_xyz = self.lagged_imu[:, -3:].clone()
            # no imu lag
            else:
                self.lagged_base_ang_vel = self.base_ang_vel[:, :3]
                self.lagged_base_euler_xyz = self.base_euler_xyz[:, -3:]

        # obs q and dq

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        # 44
        obs_buf = torch.cat((
            self.command_input, # 5 = 2D(sin cos) + 3D(vel_x, vel_y, ang_vel_yaw)
            q, # 12
            dq, # 12
            self.actions, # 10
            self.base_ang_vel * self.obs_scales.ang_vel, # 3
            self.base_euler_xyz[:,:2] * self.obs_scales.imu, # 2
        ), dim=-1)

        if self.cfg.env.num_single_obs == 45:
            stand_command = (
                        torch.norm(self.commands[:, :3], dim=1, keepdim=True) <= self.cfg.commands.stand_com_threshold)
            obs_buf = torch.cat((obs_buf, stand_command), dim=1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            privileged_obs_buf = torch.cat((privileged_obs_buf.clone(), heights), dim=-1)

        if self.cfg.noise.add_noise and self.headless:
            obs_now = obs_buf.clone() + (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * min(
                self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24), 1.)
        elif self.cfg.noise.add_noise and not self.headless:
            obs_now = obs_buf.clone() + (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.critic_history.append(privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1) # N,  T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def _parse_cfg(self, cfg):
        ''' Parse simulation, reward, command, terrain, random configuration
        '''
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        # print("reward_scales",self.reward_scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.ext_force_interval = np.ceil(self.cfg.domain_rand.ext_force_interval_s / self.dt)

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]()
            self.rew_buf += rew * self.reward_scales[name]
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

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
        noise_vec[0: self.cfg.env.num_commands] = 0.  # commands
        noise_vec[self.cfg.env.num_commands: self.cfg.env.num_commands+self.num_dofs] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[self.cfg.env.num_commands+self.num_dofs: self.cfg.env.num_commands+2*self.num_dofs] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[self.cfg.env.num_commands+2*self.num_dofs: self.cfg.env.num_commands+2*self.num_actions+self.num_dofs] = 0.  # previous actions
        noise_vec[self.cfg.env.num_commands+2*self.num_actions+self.num_dofs: self.cfg.env.num_commands+2*self.num_actions+self.num_dofs + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel   # ang vel
        noise_vec[self.cfg.env.num_commands+2*self.num_actions+self.num_dofs + 3: self.cfg.env.num_commands+2*self.num_actions+self.num_dofs + 5] = noise_scales.imu * self.obs_scales.imu         # euler x,y

        return noise_vec

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
            # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)
            # reset rand dof_pos and dof_vel=0
        self._reset_dofs(env_ids)

        # reset base position
        self._reset_root_states(env_ids)

        # Randomize joint parameters, like torque gain friction ...
        self.randomize_dof_props(env_ids)
        self.randomize_lag_props(env_ids)

        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.phase_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # rand 0 or 0.5
        self.gait_start[env_ids] = torch.randint(0, 2, (len(env_ids),)).to(self.device) * 0.5

        # resample command
        self.generate_gait_time(env_ids)
        self._resample_commands()

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["metric_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids] * self.reward_scales[key]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        count = torch.sum(torch.norm(self.commands[:, :3], dim=1) < self.cfg.commands.stand_com_threshold).item()
        self.extras["episode"]["count_command"] = count
        # fix reset gravity bug
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_euler_xyz = get_euler_xyz_tensor(self.feet_quat)

        # clear obs history buffer and privileged obs buffer
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # rand env reset self.dof_pos, which sync with self.dof_state
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.terrain.curriculum:
                platform = self.cfg.terrain.platform
                self.root_states[env_ids, :2] += torch_rand_float(-platform / 3, platform / 3, (len(env_ids), 2),
                                                                  device=self.device)  # xy position within 1m of the center
            else:
                terrain_length = self.cfg.terrain.terrain_length
                self.root_states[env_ids, :2] += torch_rand_float(-terrain_length / 2, terrain_length / 2,
                                                                  (len(env_ids), 2),
                                                                  device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # if base is fix, set fix_base_link True
        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids, 7:13] = 0
            self.root_states[env_ids, 2] += 1.8
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _reset_fric_para(self, env_ids):
        self.fric_para_0[env_ids, 0] = torch_rand_float(3.7, 6.6, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_0[env_ids, 1] = torch_rand_float(3.3, 5.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_0[env_ids, 2] = torch_rand_float(-5.0, -3.3, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_0[env_ids, 3] = torch_rand_float(0.7, 0.9, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_0[env_ids, 4] = torch_rand_float(0.7, 0.9, (len(env_ids), 1), device=self.device).squeeze(1)

        self.fric_para_1[env_ids, 0] = torch_rand_float(1.2, 2.75, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_1[env_ids, 1] = torch_rand_float(1.0, 1.55, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_1[env_ids, 2] = torch_rand_float(-1.55, -1.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_1[env_ids, 3] = torch_rand_float(0.4, 0.65, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_1[env_ids, 4] = torch_rand_float(0.4, 0.65, (len(env_ids), 1), device=self.device).squeeze(1)

        self.fric_para_2[env_ids, 0] = torch_rand_float(1.9, 3.3, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_2[env_ids, 1] = torch_rand_float(1.15, 2.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_2[env_ids, 2] = torch_rand_float(-2.0, -1.3, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_2[env_ids, 3] = torch_rand_float(0.14, 0.18, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_2[env_ids, 4] = torch_rand_float(0.14, 0.18, (len(env_ids), 1), device=self.device).squeeze(1)

        self.fric_para_3[env_ids, 0] = torch_rand_float(0.25, 1.25, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_3[env_ids, 1] = torch_rand_float(0.2, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_3[env_ids, 2] = torch_rand_float(-1.0, -0.2, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_3[env_ids, 3] = torch_rand_float(0.14, 0.18, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_3[env_ids, 4] = torch_rand_float(0.14, 0.18, (len(env_ids), 1), device=self.device).squeeze(1)

    def _compute_torques(self, actions):
        # pd controller
        actions_scaled_raw = actions * self.cfg.control.action_scale
        actions_scaled = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        actions_scaled[:, self.control_index] = actions_scaled_raw
        control_type = self.cfg.control.control_type
        if control_type == "P":
            if not self.cfg.domain_rand.randomize_motor:
                torques = self.p_gains * (
                            actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains * self.dof_vel
            else:
                torques = self.motor_strength[0] * self.p_gains * (
                            actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[
                              1] * self.d_gains * self.dof_vel

        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (
                        self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        if self.use_motor_model:
            torques[:, [0]] = self.motordelay0(torques[:, [0]])
            torques[:, [6]] = self.motordelay6(torques[:, [6]])
            torques[:, [2]] = self.motordelay2(torques[:, [2]])
            torques[:, [3]] = self.motordelay3(torques[:, [3]])
            torques[:, [8]] = self.motordelay8(torques[:, [8]])
            torques[:, [9]] = self.motordelay9(torques[:, [9]])

            self.friction_0_6()
            self.friction_1_7()
            self.friction_2_3_8_9()

            torques -= self.fric

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def friction_0_6(self):
        flag_0 = (self.dof_vel[:, 0] <= 0.002) & (self.dof_vel[:, 0] >= -0.002)
        flag_1 = ((self.dof_vel[:, 0] > 0.002) & (self.dof_vel[:, 0] <= 0.16))
        flag_2 = (self.dof_vel[:, 0] > 0.16)
        flag_3 = ((self.dof_vel[:, 0] < -0.002) & (self.dof_vel[:, 0] >= -0.16))
        flag_4 = (self.dof_vel[:, 0] < -0.16)

        self.fric[:, 0] = self.fric_para_0[:, 0] / 0.002 * self.dof_vel[:, 0] * flag_0 + \
                          ((self.fric_para_0[:, 1] - self.fric_para_0[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 0] - 0.002) + self.fric_para_0[:, 0]) * flag_1 + \
                          (self.fric_para_0[:, 1] + self.fric_para_0[:, 3] * (self.dof_vel[:, 0] - 0.16)) * flag_2 + \
                          ((self.fric_para_0[:, 2] + self.fric_para_0[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 0] + 0.002) - self.fric_para_0[:, 0]) * flag_3 + \
                          (self.fric_para_0[:, 2] + self.fric_para_0[:, 4] * (self.dof_vel[:, 0] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 6] <= 0.002) & (self.dof_vel[:, 6] >= -0.002)
        flag_1 = ((self.dof_vel[:, 6] > 0.002) & (self.dof_vel[:, 6] <= 0.16))
        flag_2 = (self.dof_vel[:, 6] > 0.16)
        flag_3 = ((self.dof_vel[:, 6] < -0.002) & (self.dof_vel[:, 6] >= -0.16))
        flag_4 = (self.dof_vel[:, 6] < -0.16)

        self.fric[:, 6] = self.fric_para_0[:, 0] / 0.002 * self.dof_vel[:, 6] * flag_0 + \
                          ((self.fric_para_0[:, 1] - self.fric_para_0[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 6] - 0.002) + self.fric_para_0[:, 0]) * flag_1 + \
                          (self.fric_para_0[:, 1] + self.fric_para_0[:, 3] * (self.dof_vel[:, 6] - 0.16)) * flag_2 + \
                          ((self.fric_para_0[:, 2] + self.fric_para_0[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 6] + 0.002) - self.fric_para_0[:, 0]) * flag_3 + \
                          (self.fric_para_0[:, 2] + self.fric_para_0[:, 4] * (self.dof_vel[:, 6] + 0.16)) * flag_4

    def friction_1_7(self):
        flag_0 = (self.dof_vel[:, 1] <= 0.002) & (self.dof_vel[:, 1] >= -0.002)
        flag_1 = ((self.dof_vel[:, 1] > 0.002) & (self.dof_vel[:, 1] <= 0.16))
        flag_2 = (self.dof_vel[:, 1] > 0.16)
        flag_3 = ((self.dof_vel[:, 1] < -0.002) & (self.dof_vel[:, 1] >= -0.16))
        flag_4 = (self.dof_vel[:, 1] < -0.16)

        self.fric[:, 1] = self.fric_para_1[:, 0] / 0.002 * self.dof_vel[:, 1] * flag_0 + \
                          ((self.fric_para_1[:, 1] - self.fric_para_1[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 1] - 0.002) + self.fric_para_1[:, 0]) * flag_1 + \
                          (self.fric_para_1[:, 1] + self.fric_para_1[:, 3] * (self.dof_vel[:, 1] - 0.16)) * flag_2 + \
                          ((self.fric_para_1[:, 2] + self.fric_para_1[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 1] + 0.002) - self.fric_para_1[:, 0]) * flag_3 + \
                          (self.fric_para_1[:, 2] + self.fric_para_1[:, 4] * (self.dof_vel[:, 1] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 7] <= 0.002) & (self.dof_vel[:, 7] >= -0.002)
        flag_1 = ((self.dof_vel[:, 7] > 0.002) & (self.dof_vel[:, 7] <= 0.16))
        flag_2 = (self.dof_vel[:, 7] > 0.16)
        flag_3 = ((self.dof_vel[:, 7] < -0.002) & (self.dof_vel[:, 7] >= -0.16))
        flag_4 = (self.dof_vel[:, 7] < -0.16)

        self.fric[:, 7] = self.fric_para_1[:, 0] / 0.002 * self.dof_vel[:, 7] * flag_0 + \
                          ((self.fric_para_1[:, 1] - self.fric_para_1[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 7] - 0.002) + self.fric_para_1[:, 0]) * flag_1 + \
                          (self.fric_para_1[:, 1] + self.fric_para_1[:, 3] * (self.dof_vel[:, 7] - 0.16)) * flag_2 + \
                          ((self.fric_para_1[:, 2] + self.fric_para_1[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 7] + 0.002) - self.fric_para_1[:, 0]) * flag_3 + \
                          (self.fric_para_1[:, 2] + self.fric_para_1[:, 4] * (self.dof_vel[:, 7] + 0.16)) * flag_4

    def friction_2_3_8_9(self):
        flag_0 = (self.dof_vel[:, 2] <= 0.002) & (self.dof_vel[:, 2] >= -0.002)
        flag_1 = ((self.dof_vel[:, 2] > 0.002) & (self.dof_vel[:, 2] <= 0.16))
        flag_2 = (self.dof_vel[:, 2] > 0.16)
        flag_3 = ((self.dof_vel[:, 2] < -0.002) & (self.dof_vel[:, 2] >= -0.16))
        flag_4 = (self.dof_vel[:, 2] < -0.16)

        self.fric[:, 2] = self.fric_para_2[:, 0] / 0.002 * self.dof_vel[:, 2] * flag_0 + \
                          ((self.fric_para_2[:, 1] - self.fric_para_2[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 2] - 0.002) + self.fric_para_2[:, 0]) * flag_1 + \
                          (self.fric_para_2[:, 1] + self.fric_para_2[:, 3] * (self.dof_vel[:, 2] - 0.16)) * flag_2 + \
                          ((self.fric_para_2[:, 2] + self.fric_para_2[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 2] + 0.002) - self.fric_para_2[:, 0]) * flag_3 + \
                          (self.fric_para_2[:, 2] + self.fric_para_2[:, 4] * (self.dof_vel[:, 2] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 3] <= 0.002) & (self.dof_vel[:, 3] >= -0.002)
        flag_1 = ((self.dof_vel[:, 3] > 0.002) & (self.dof_vel[:, 3] <= 0.16))
        flag_2 = (self.dof_vel[:, 3] > 0.16)
        flag_3 = ((self.dof_vel[:, 3] < -0.002) & (self.dof_vel[:, 3] >= -0.16))
        flag_4 = (self.dof_vel[:, 3] < -0.16)

        self.fric[:, 3] = self.fric_para_2[:, 0] / 0.002 * self.dof_vel[:, 3] * flag_0 + \
                          ((self.fric_para_2[:, 1] - self.fric_para_2[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 3] - 0.002) + self.fric_para_2[:, 0]) * flag_1 + \
                          (self.fric_para_2[:, 1] + self.fric_para_2[:, 3] * (self.dof_vel[:, 3] - 0.16)) * flag_2 + \
                          ((self.fric_para_2[:, 2] + self.fric_para_2[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 3] + 0.002) - self.fric_para_2[:, 0]) * flag_3 + \
                          (self.fric_para_2[:, 2] + self.fric_para_2[:, 4] * (self.dof_vel[:, 3] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 8] <= 0.002) & (self.dof_vel[:, 8] >= -0.002)
        flag_1 = ((self.dof_vel[:, 8] > 0.002) & (self.dof_vel[:, 8] <= 0.16))
        flag_2 = (self.dof_vel[:, 8] > 0.16)
        flag_3 = ((self.dof_vel[:, 8] < -0.002) & (self.dof_vel[:, 8] >= -0.16))
        flag_4 = (self.dof_vel[:, 8] < -0.16)

        self.fric[:, 8] = self.fric_para_2[:, 0] / 0.002 * self.dof_vel[:, 8] * flag_0 + \
                          ((self.fric_para_2[:, 1] - self.fric_para_2[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 8] - 0.002) + self.fric_para_2[:, 0]) * flag_1 + \
                          (self.fric_para_2[:, 1] + self.fric_para_2[:, 3] * (self.dof_vel[:, 8] - 0.16)) * flag_2 + \
                          ((self.fric_para_2[:, 2] + self.fric_para_2[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 8] + 0.002) - self.fric_para_2[:, 0]) * flag_3 + \
                          (self.fric_para_2[:, 2] + self.fric_para_2[:, 4] * (self.dof_vel[:, 8] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 9] <= 0.002) & (self.dof_vel[:, 9] >= -0.002)
        flag_1 = ((self.dof_vel[:, 9] > 0.002) & (self.dof_vel[:, 9] <= 0.16))
        flag_2 = (self.dof_vel[:, 9] > 0.16)
        flag_3 = ((self.dof_vel[:, 9] < -0.002) & (self.dof_vel[:, 9] >= -0.16))
        flag_4 = (self.dof_vel[:, 9] < -0.16)

        self.fric[:, 9] = self.fric_para_2[:, 0] / 0.002 * self.dof_vel[:, 9] * flag_0 + \
                          ((self.fric_para_2[:, 1] - self.fric_para_2[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 9] - 0.002) + self.fric_para_2[:, 0]) * flag_1 + \
                          (self.fric_para_2[:, 1] + self.fric_para_2[:, 3] * (self.dof_vel[:, 9] - 0.16)) * flag_2 + \
                          ((self.fric_para_2[:, 2] + self.fric_para_2[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 9] + 0.002) - self.fric_para_2[:, 0]) * flag_3 + \
                          (self.fric_para_2[:, 2] + self.fric_para_2[:, 4] * (self.dof_vel[:, 9] + 0.16)) * flag_4

    # ======================================================================================================================
    # Reward functions
    # ======================================================================================================================
    def _reward_ref_joint_pos(self):
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        pos_target[stand_command] = self.default_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        r[stand_command] = 1.0

        return r


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

    def _reward_foot_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_body_states[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_stand_still(self):
        # penalize motion at zero commands
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        r = torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1))
        # r = torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1))
        r = torch.where(stand_command, r.clone(),
                        torch.zeros_like(r))
        return r

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_stance_mask()
        stance_mask[torch.norm(self.commands[:, :3], dim=1) < 0.05] = 1
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    def _reward_dof_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_orientation(self):
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]),dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2],dim=-1) * 20)
        return (quat_mismatch + orientation) / 2

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, [0,1]]
        right_yaw_roll = joint_diff[:, [6,7]]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_tracking_lin_vel_exp(self):
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_error_square = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_error_abs = torch.sum(torch.abs(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        r_square = torch.exp(-lin_vel_error_square * self.cfg.rewards.tracking_sigma)
        r_abs = torch.exp(-lin_vel_error_abs * self.cfg.rewards.tracking_sigma * 2)
        r = torch.where(stand_command, r_abs, r_square)
        # r = torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        return r

    def _reward_tracking_ang_vel(self):
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        ang_vel_error_square = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_abs = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        r_square = torch.exp(-ang_vel_error_square * self.cfg.rewards.tracking_sigma)
        r_abs = torch.exp(-ang_vel_error_abs * self.cfg.rewards.tracking_sigma * 2)
        r = torch.where(stand_command, r_abs, r_square)
        # r = torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma_ang)
        # ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return r

    def _reward_feet_clearance(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        feet_z = self.rigid_body_states[:, self.feet_indices, 2] - 0.05

        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z
        swing_mask = 1 - self._get_stance_mask()
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_feet_contact_number(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_stance_mask().clone()
        stance_mask[torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold] = 1
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_feet_contact_forces(self):
        rew = torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0,400), dim=-1)
        return rew

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

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_stance_mask()

        measured_heights = torch.sum(
            self.rigid_body_states[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - self.cfg.rewards.feet_to_ankle_distance)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    # def _reward_feet_rotation(self):
    #     feet_euler_xyz = self.feet_euler_xyz
    #     rotation = torch.sum(torch.square(feet_euler_xyz[:, :, :2]), dim=[1, 2])
    #     # rotation = torch.sum(torch.square(feet_euler_xyz[:,:,1]),dim=1)
    #     r = torch.exp(-rotation * 15)
    #     return r