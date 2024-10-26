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


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class GRCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_dofs = 12
        num_single_obs = 41
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 164
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 10
        num_envs = 4096

        history_len = 10
        n_proprio = 2 + 3 + 3 + 2 + 2 * (num_dofs) + num_actions
        n_priv_latent = 4 + 1 + 2 * (num_dofs) + 3 #484 32
        queue_len_obs = 4
        queue_len_act = 4
        control_indices = [0,1,2,3,4,6,7,8,9,10]
        # 系统延迟
        obs_latency = [5, 20]
        act_latency = [5, 20]
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False


    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.9

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/gr1t1/urdf/GR1T1_5DoF.urdf'

        name = "GR1T1"
        foot_name = "foot_pitch"
        knee_name = "shank"

        terminate_after_contacts_on = ['base']
        penalize_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        # measured_points_x = np.arange(-0.2, 0.39, 0.05)
        # measured_points_y = np.arange(-0.2, 0.19, 0.05)
        measured_points_x = [-0.2, -0.15, -0.10, -0.05, -0.0, 0.05, 0.1, 0.15, 0.2]
        measured_points_y = [-0.2, -0.15, -0.10, -0.05, -0.0, 0.05, 0.1, 0.15, 0.2]
        measure_heights = True
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 1.0    # scales other values

        class noise_scales:
            dof_pos = 0.3
            dof_vel = 1.
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.1
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.90]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.,
            'left_hip_pitch_joint': -0.4,
            'left_knee_pitch_joint': 0.8,
            'left_ankle_pitch_joint': -0.4,
            'left_ankle_roll_joint': -0.0,
            'right_hip_roll_joint': -0.0,
            'right_hip_yaw_joint': -0.,
            'right_hip_pitch_joint': -0.4,
            'right_knee_pitch_joint': 0.8,
            'right_ankle_pitch_joint': -0.4,
            'right_ankle_roll_joint': -0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_roll': 200, 'hip_yaw': 200, 'hip_pitch': 350,
            'knee_pitch': 350,'ankle_pitch': 20}
        damping = {'hip_roll': 20, 'hip_yaw': 20, 'hip_pitch': 35,
            'knee_pitch': 35,'ankle_pitch': 2}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 50hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # 200 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        restitution_range = [0.0, 0.1]
        dynamic_randomization = 0.1

        random_quaternion = True
        quaternion_range = [-0.1, 0.2]

        # randomize friction
        randomize_friction = True
        friction_range = [0.2, 1.5]

        # randomize payload
        randomize_body_mass = True
        added_body_mass_range = [-1.50, 2.50]
        added_leg_mass_range = [-0.1, 0.30]

        # randomize base center of mass  ok
        randomize_body_com = True
        added_body_com_range = [-0.02, 0.02]  # 1cm
        added_leg_com_range = [-0.01, 0.01]  # 0.5cm

        # randomize body_inertia  ok
        randomize_body_inertia = True
        scaled_body_inertia_range = [0.85, 1.15]  # %5 error

        randomize_thigh_mass = True
        randomize_shank_mass = True
        randomize_torso_mass = True
        randomize_upper_arm_mass = True
        randomize_lower_arm_mass = True

        # randomize motor strength ok
        randomize_motor_strength = True
        scaled_motor_strength_range = [0.8, 1.2]


        # randomize external forces ok
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
        max_push_vel_ang = 0.5
        push_curriculum_start_step = 500 * 60
        push_curriculum_common_step = 500 * 60

        apply_forces = False
        continue_time_s = 0.5
        max_ex_forces = [-200.0, 200.0]
        max_ex_torques = [-0.0, 0.0]

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        step_joint_offset = 0.17  # rad
        step_freq = 1.5625  # HZ （e.g. cycle-time=0.66）

        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.6, 0.6]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            heading = [-0.0, 0.0]

    class rewards:
        base_height_target = 0.895
        min_dist = 0.2
        max_dist = 0.6
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.05        # m
        cycle_time = 0.8                  # sec
        swing_time = 0.5 * cycle_time
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        foot_height_tracking = 5
        tracking_feet_height_sigma = 3
        tracking_feet_vel_sigma = 5
        lin_tracking_sigma = 5
        ang_tracking_sigma = 7
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.90
        soft_torque_limit = 0.90
        tracking_sigma = 0.15
        max_contact_force = 900  # Forces above this value are penalized
        min_contact_force = 400

        class scales:
            # reference motion tracking
            joint_pos = 1.6
            feet_clearance = 1.
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.8  # lin_z; ang x,y
            low_speed = 0.8
            track_vel_hard = 0.8
            # base pos
            default_joint_pos = 1.
            # target_hip_roll_pos = 1.
            # default_joint_roll_pos = 1.2
            # default_joint_yaw_pos = 1.2
            orientation = 2.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.
            peroid_force = 1.
            peroid_vel = 1.

            # foot_height_tracking = 1.0
            # foot_vel_tracking = 1.0



    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 0.25
            dof_pos = 1.
            dof_vel = 0.05
            quat = 0.5
            height_measurements = 5.0
        clip_observations = 50.
        clip_actions = 5.



class GRCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 5
        gamma = 0.994
        lam = 0.95
        num_mini_batches = 4
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 3001  # number of policy updates

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = '1014EndWar'
        run_name = ''
        # Load and resume
        resume = True
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt