import math
from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 48
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        test = False

    class terrain:
        mesh_type = 'trimesh' # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        terrain_spacing = 0.5 # spacing between different terrain types [m]

        # [wave, slope, rough slope, stairs up, stairs down, obstacles, stepping stones, gap, flat]
        terrain_proportions = [0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        move_down_by_accumulated_xy_command = False # move down the terrain curriculum based on accumulated xy command distance instead of absolute distance

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        zero_command_curriculum = None
        # start training with zero commands and then gradually increase zero command probability
        # eg. {'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.1}
        limit_ang_vel_at_zero_command_prob = 0.0 # probability of add limiting angular velocity commands when zero command is sampled
        limit_vel_prob = 0.0 # probability of limiting linear velocity command
        limit_vel_invert_when_continuous = True # invert the limit logic when using continuous sample limit velocity commands
        limit_vel = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]} # sample vel commands from min [-1] or zero [0] or max [1] range only
        stop_heading_at_limit = True # stop heading updates when vel is limited
        dynamic_resample_commands = False # sample commands with low bounds
        command_range_curriculum = [] # list for command range curriculums at specific training iterations
        # eg: [{
        #     'iter': 20000, # training iteration at which the command ranges are updated
        #     'lin_vel_x': [-1.0, 1.0], # min max [m/s]
        #     'lin_vel_y': [-1.0, 1.0], # min max [m/s]
        #     'ang_vel_yaw': [-2.0, 2.0], # min max [rad/s]
        #     'heading': [-1.57, 1.57], # min max [rad]
        # }]
        turn_over_zero_time = { # if turn_over is true, time robot must be stable before sampling new commands after a turn over
            "backflip": 5.0,
            "sideflip": 3.0,
        }
        # [wave, slope, rough slope, stairs up, stairs down, obstacles, stepping stones, gap, flat]
        terrain_max_command_ranges = [
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # wave
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # slope
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # rough slope
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs up
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs down
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # obstacles
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stepping stones
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # gap
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # flat
        ]

        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}
        turn_over = False # if true, initialize the robot in a flipped over position
        turn_over_proportions = [0.0, 0.2, 0.8] # proportions for backflip, sideflip, no flip
        turn_over_init_heights = { # initial heights range for each flip type
            'backflip': [0.10, 0.15],
            'sideflip': [0.16, 0.21],
        }

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        ### Robot properties ###
        robot_properties_update = None
        # eg: {'start_iter': 5000, 'interval': 5000}

        randomize_friction = True
        friction_range = [0.2, 1.25]

        randomize_base_mass = True
        added_mass_range = [-1., 1.]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = True
        added_base_com_range = [-0.03, 0.03]

        randomize_restitution = False # restitution to robot links (Robot init)
        restitution_range = [0.0, 0.2]

        ### Environment reset ###
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]  
        damping_multiplier_range = [0.9, 1.1]    

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]

        randomize_motor_strength = False # (Env reset)
        motor_strength_range = [0.8, 1.2]

        ### Environment step ###
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        randomize_action_delay = False # use last_action with 0~20 ms delay, 4 decimation

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.

        class turn_over_scales:
            upright = 1.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized
        curriculum_rewards = None  # reward names to apply curriculum scaling to, List[dict]
        # eg: [{'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0}]
        dynamic_sigma = None # linear interpolation of sigma based on command velocity, **Must start terrain curriculum first**
        # eg: {
        #     "min_vel": 0.5, # min abs velocity to have default sigma
        #     "max_vel": 1.0, # max abs velocity to have max sigma
        #     # wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat]
        #     "max_sigma": [1/3, 1/4, 1/4, 1/2.7, 1/2.7, 1/2, 1, 1, 1/4]
        # }
        turn_over_roll_threshold = math.pi / 4 # threshold on roll to use turn over rewards
        min_legs_distance = 0.1  # min distance between legs to not be considered stumbling

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 2.5
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

    class robogauge:
        enabled = False
        port = 9973

class LeggedRobotCfgCTS(BaseConfig):
    seed = 0
    runner_class_name = "OnPolicyRunnerCTS"
    history_length = 5
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        teacher_encoder_hidden_dims = [512, 256]
        student_encoder_hidden_dims = [512, 256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        latent_dim = 32
        norm_type = 'l2norm' # normalization type for encoders: l2norm, simnorm
    
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        student_encoder_learning_rate = 1e-3
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        teacher_env_ratio = 0.75  # percentage of envs assigned to teacher
        # teacher_env_ratio = 1.00  # percentage of envs assigned to teacher

    class runner:
        policy_class_name = 'ActorCriticCTS'
        algorithm_class_name = 'CTS'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

    class robogauge:
        enabled = False
        port = 9973

class LeggedRobotCfgMoENGCTS(LeggedRobotCfgCTS):
    class policy(LeggedRobotCfgCTS.policy):
        obs_no_goal_mask = None # mask for observation without goal inputs
        student_expert_num = 8 # number of experts in the student model
    
    class algorithm(LeggedRobotCfgCTS.algorithm):
        load_balance_coef = 0.01  # coefficient for load balance loss

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticMoENGCTS'
        algorithm_class_name = 'MoENGCTS'

class LeggedRobotCfgMCPCTS(LeggedRobotCfgCTS):
    class policy(LeggedRobotCfgCTS.policy):
        obs_no_goal_mask = None # mask for observation without goal inputs
        student_expert_num = 8 # number of experts in the student model

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticMCPCTS'
        algorithm_class_name = 'MCPCTS'

class LeggedRobotCfgACMoECTS(LeggedRobotCfgCTS):
    class policy(LeggedRobotCfgCTS.policy):
        expert_num = 8 # number of experts in the student model

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticACMoECTS'
        algorithm_class_name = 'ACMoECTS'

class LeggedRobotCfgDualMoECTS(LeggedRobotCfgCTS):
    class policy(LeggedRobotCfgCTS.policy):
        expert_num = 8 # number of experts in the student model
        student_encoder_hidden_dims = [512, 256, 256]

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticDualMoECTS'
        algorithm_class_name = 'DualMoECTS'

class LeggedRobotCfgMoECTS(LeggedRobotCfgCTS):
    class policy(LeggedRobotCfgCTS.policy):
        expert_num = 8 # number of experts in the student model
        student_encoder_hidden_dims = [512, 256, 256]

    class algorithm(LeggedRobotCfgCTS.algorithm):
        load_balance_coef = 0.01  # coefficient for load balance loss

    class runner(LeggedRobotCfgCTS.runner):
        policy_class_name = 'ActorCriticMoECTS'
        algorithm_class_name = 'MoECTS'
