from legged_gym.envs.base.humanoid_config import HumanoidCfg, HumanoidCfgPPO
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class GR1_explicitCfg(HumanoidCfg):
    class env(HumanoidCfg.env):
        frame_stack = 66 # all history obs num
        short_frame_stack = 5 # short history step
        c_frame_stack = 3 # all history privileged obs num
        num_actions = 10
        num_dofs = 12
        num_single_obs = 2 + 3 + 3 + 2 + 2 * num_dofs + num_actions
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 71
        single_linvel_index = 51
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_envs = 4096
        num_commands = 5
        add_stand_bool = False
        use_motor_model = True
        normalize_obs = True
        history_encoding = False
        contact_buf_len = 10



    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(HumanoidCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/gr1t1/urdf/GR1T1_5dof.urdf'

        torso_name: str = 'base'
        chest_name: str = 'waist_roll'
        forehead_name: str = 'head_pitch'

        waist_name: str = 'waist'
        waist_roll_name: str = 'waist_roll'
        waist_pitch_name: str = 'waist_pitch'
        head_name: str = 'head'
        head_roll_name: str = 'head_roll'
        head_pitch_name: str = 'head_pitch'

        # for link name
        thigh_name: str = 'thigh'
        shank_name: str = 'shank'
        foot_name: str = 'foot_roll'
        upper_arm_name: str = 'upper_arm'
        lower_arm_name: str = 'lower_arm'
        hand_name: str = 'hand'

        # for joint name
        hip_name: str = 'hip'
        hip_roll_name: str = 'hip_roll'
        hip_yaw_name: str = 'hip_yaw'
        hip_pitch_name: str = 'hip_pitch'
        knee_name: str = 'knee'
        ankle_name: str = 'ankle'
        ankle_pitch_name: str = 'ankle_pitch'
        shoulder_name: str = 'shoulder'
        shoulder_pitch_name: str = 'shoulder_pitch'
        shoulder_roll_name: str = 'shoulder_roll'
        shoulder_yaw_name: str = 'shoulder_yaw'
        elbow_name: str = 'elbow'
        wrist_name: str = 'wrist'
        wrist_yaw_name: str = 'wrist_yaw'
        wrist_roll_name: str = 'wrist_roll'
        wrist_pitch_name: str = 'wrist_pitch'

        feet_bodies = ['l_foot_roll', 'r_foot_roll']
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["shoulder", "elbow", "thigh"]
        terminate_after_contacts_on = ['waist']

    class terrain(HumanoidCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        max_init_terrain_level = 5 # starting curriculum state
        platform = 3.
        terrain_dict = {"flat": 0.3,
                        "rough flat": 0.2,
                        "slope up": 0.2,
                        "slope down": 0.2,
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "stairs up": 0.,
                        "stairs down": 0.,
                        "discrete": 0.1,
                        "wave": 0.0, }
        terrain_proportions = list(terrain_dict.values())
        rough_flat_range = [0.005, 0.01]  # meter
        slope_range = [0, 0.1]  # rad
        rough_slope_range = [0.005, 0.02]
        stair_width_range = [0.25, 0.25]
        stair_height_range = [0.01, 0.1]
        discrete_height_range = [0.0, 0.01]
        restitution = 0.

    class noise(HumanoidCfg.noise):
        add_noise = True
        noise_increasing_steps = 5000

        class noise_scales:
            dof_pos = 0.2
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.05
            gravity = 0.05
            imu = 0.1

    class init_state(HumanoidCfg.init_state):
        pos = [0, 0, 0.95]
        default_joint_angles = {
            'l_hip_roll': 0.0,
            'l_hip_yaw': 0.,
            'l_hip_pitch': -0.4,
            'l_knee_pitch': 0.8,
            'l_ankle_pitch': -0.4,
            'l_ankle_roll': 0.0,

            # right leg
            'r_hip_roll': -0.,
            'r_hip_yaw': 0.,
            'r_hip_pitch': -0.4,
            'r_knee_pitch': 0.8,
            'r_ankle_pitch': -0.4,
            'r_ankle_roll': 0.0,

            # waist
            'waist_yaw': 0.0,
            'waist_pitch': 0.0,
            'waist_roll': 0.0,

            # head
            'head_yaw': 0.0,
            'head_pitch': 0.0,
            'head_roll': 0.0,

            # left arm
            'l_shoulder_pitch': 0.0,
            'l_shoulder_roll': 0.2,
            'l_shoulder_yaw': 0.0,
            'l_elbow_pitch': -0.3,
            'l_wrist_yaw': 0.0,
            'l_wrist_roll': 0.0,
            'l_wrist_pitch': 0.0,

            # right arm
            'r_shoulder_pitch': 0.0,
            'r_shoulder_roll': -0.2,
            'r_shoulder_yaw': 0.0,
            'r_elbow_pitch': -0.3,
            'r_wrist_yaw': 0.0,
            'r_wrist_roll': 0.0,
            'r_wrist_pitch': 0.0
        }

    class control(HumanoidCfg.control):
        stiffness = {
            'hip_roll': 200, 'hip_yaw': 200, 'hip_pitch': 350,
            'knee_pitch': 350,
            'ankle_pitch': 10.98, 'ankle_roll': 0.0
        }
        damping = {
            'hip_roll': 20, 'hip_yaw': 20, 'hip_pitch': 20,
            'knee_pitch': 20,
            'ankle_pitch': 0.60, 'ankle_roll': 0.1
        }

        action_scale = 0.5
        decimation = 10 # policy 100Hz

    class sim(HumanoidCfg.sim):
        dt = 0.001 # pd 1000Hz
        substeps = 1
        up_axis = 1

    class normalization(HumanoidCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            imu = 0.5

        clip_actions = 5

    class rewards(HumanoidCfg.rewards):
        regularization_names = [
            "dof_error",
            "dof_error_upper",
            "feet_stumble",
            "feet_contact_forces",
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
            "dof_pos_limits",
            "dof_torque_limits",
            "collision",
            "torque_penalty",
        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8, 2.0]
        regularization_scale_curriculum = True
        regularization_scale_gamma = 0.0001

        class scales:
            joint_pos = 2.2 # 1.6
            feet_clearance = 1.
            feet_contact_number = 2.0

            # gait
            feet_air_time = 1.2
            foot_slip = -0.1
            feet_distance = 0.2
            knee_distance = 0.2

            # contact force
            feet_contact_forces = -2e-3

            # vel tracking
            tracking_lin_vel_exp = 1.8
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5

            # base pose
            default_joint_pos = 1.0
            orientation = 1.0
            feet_rotation = 0.3
            base_acc = 0.2

            # energy
            action_smoothness = -0.002
            torques = -8e-9
            dof_vel = -2e-8
            dof_acc = -1e-7
            collision = -10.0
            stand_still = 2.5

            alive = 2.0
            feet_stumble = -1.25

            # limits
            dof_vel_limits = -1.
            dof_pos_limits = -10.0
            dof_torque_limits = -0.1

        min_dist = 0.2
        max_dist = 0.5
        max_knee_dist = 0.25
        target_joint_pos_scale = 0.17
        target_feet_height = 0.1
        cycle_time = 0.8
        double_support_threshold = 0.5
        only_positive_rewards = True
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 500  # Forces above this value are penalized
        soft_torque_limit = 0.9

    class domain_rand(LeggedRobotCfg.domain_rand):
        domain_rand_general = True  # manually set this, setting from parser does not work;

        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)

        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.]
        restitution_range = [0.0, 0.4]

        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-3., 6.]

        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.1, 0.1]
        added_com_z_range = [-0.15, 0.15]

        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        update_step = 2000 * 24
        push_duration = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        max_push_vel_xy = 1.0
        max_push_ang_xy = 0.2

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.8, 1.2]

        randomize_motor_offset = (True and domain_rand_general)
        motor_offset_range = [-0.035, 0.035]

        action_delay = (True and domain_rand_general)
        action_buf_len = 8

        add_lag = (False and domain_rand_general)
        randomize_lag_timesteps = False
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [5, 40]

        add_dof_lag = (False and domain_rand_general)
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 40]

        add_dof_pos_vel_lag = (False and domain_rand_general)
        randomize_dof_pos_lag_timesteps = False
        randomize_dof_pos_lag_timesteps_perstep = False
        dof_pos_lag_timesteps_range = [7, 25]
        randomize_dof_vel_lag_timesteps = False
        randomize_dof_vel_lag_timesteps_perstep = False
        dof_vel_lag_timesteps_range = [7, 25]

        add_imu_lag = (False and domain_rand_general)
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [1, 10]

        randomize_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  # Factor
        damping_multiplier_range = [0.8, 1.2]  # Factor

        ext_force_interval_s = 10

    class commands:
        curriculum = True
        max_curriculum = 1.5
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        ang_vel_clip = 0.1
        lin_vel_clip = 0.1
        resampling_time = 25.  # time before command are changed[s]
        gait = ["walk_omnidirectional", "stand", "walk_omnidirectional"] # gait type during training
        # proportion during whole life time
        gait_time_range = {"walk_sagittal": [2,6],
                           "walk_lateral": [2,6],
                           "rotate": [2,3],
                           "stand": [2,3],
                           "walk_omnidirectional": [4,6]}

        heading_command = False
        stand_com_threshold = 0.05 # if (lin_vel_x, lin_vel_y, ang_vel_yaw).norm < this, robot should stand
        sw_switch = True # use stand_com_threshold or not


        class ranges:
            lin_vel_x = [-0.4, 1.2]  # min max [m/s]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]
            heading = [-3.14, 3.14]

class GR1_explicitCfgPPO(HumanoidCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunnerExplicit'

    class runner(HumanoidCfgPPO.runner):
        policy_class_name = 'ActorCriticExplicit'
        algorithm_class_name = 'PPOEXPLICIT'
        num_steps_per_env = 24
        max_iterations = 20001  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'GR1_x1_stand'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        num_mini_batches = 4
        gamma = 0.994
        lam = 0.9
        desired_kl = 0.008
        num_mini_batches = 4
        max_grad_norm = 1.
        schedule = 'adaptive'  # could be adaptive, fixed

        if GR1_explicitCfg.terrain.measure_heights:
            lin_vel_idx = (GR1_explicitCfg.env.single_num_privileged_obs + GR1_explicitCfg.terrain.num_height * GR1_explicitCfg.env.c_frame_stack - 1)
            + GR1_explicitCfg.env.single_linvel_index
        else:
            lin_vel_idx = GR1_explicitCfg.env.single_num_privileged_obs * (GR1_explicitCfg.env.c_frame_stack - 1) + GR1_explicitCfg.env.single_linvel_index

        # grad_penalty_coef_schedule = [0.002, 0.002, 700, 1000]

    class policy(HumanoidCfgPPO.policy):
        action_std = [0.3, 0.3, 0.3, 0.4, 0.2] * 2
        fix_action_std = True
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        state_estimator_hidden_dims = [256, 128, 64]

        # for long_history cnn only
        kernel_size = [6, 4]
        filter_size = [32, 16]
        stride_size = [3, 2]
        lh_output_dim = 64 # long history output dim
        in_channels = GR1_explicitCfg.env.frame_stack
