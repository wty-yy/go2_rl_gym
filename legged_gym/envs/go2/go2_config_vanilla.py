# -*- coding: utf-8 -*-
'''
@File    : go2_config_vanilla.py
@Time    : 2026/01/10 02:26:04
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : Go2 vanilla training config
episode length 25, resample commands 5 sec,
open move_down_by_accumulated_xy_command, dynamic_resample_commands
close heading_command, zero_command_curriculum, limit_vel_prob, command_range_curriculum, dynamic_sigma
'''
import math
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, LeggedRobotCfgCTS, LeggedRobotCfgMoECTS, LeggedRobotCfgMoECTS, LeggedRobotCfgMCPCTS, LeggedRobotCfgACMoECTS, LeggedRobotCfgDualMoECTS, LeggedRobotCfgREMCTS

class GO2Cfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
        turn_over = False # initialize the robot in a flipped over position
        # turn_over_proportions = [0.1, 0.3, 0.6] # proportions for backflip, sideflip, noflip
        turn_over_proportions = [0.0, 0.2, 0.8] # proportions for backflip, sideflip, noflip
        turn_over_init_heights = { # initial heights range for each flip type
            'backflip': [0.10, 0.15],
            'sideflip': [0.16, 0.21],
        }
        # turn_over_proportions = [0.0, 1.0, 0.0] # proportions for backflip, sideflip, noflip

    class env(LeggedRobotCfg.env):
        num_envs = 8192
        num_observations = 45
        # obs(45) + base_lin_vel(3) + height_measurements(187)
        num_privileged_obs = 45 + 3 + 4 + 12 + 12 + 187  # 263
        # num_privileged_obs = 45 + 3 + 187  # 235
        # num_privileged_obs = 48  # without height measurements
        episode_length_s = 25

    class domain_rand(LeggedRobotCfg.domain_rand):
        ### Robot properties ###
        randomize_friction = True
        friction_range = [0.0, 2.0]

        randomize_base_mass = True
        added_mass_range = [-1., 1.]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = True
        added_base_com_range = [-0.03, 0.03]

        randomize_restitution = True # restitution to robot links (Robot init)
        restitution_range = [0.0, 0.5]

        ### Environment reset ###
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]  
        damping_multiplier_range = [0.9, 1.1]    

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]

        randomize_motor_strength = True # (Env reset)
        motor_strength_range = [0.8, 1.2]

        ### Environment step ###
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        randomize_action_delay = True # use last_action with 0~20 ms delay, 4 decimation

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.0}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class terrain(LeggedRobotCfg.terrain):
        max_init_terrain_level = 5
        # [wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat]
        # terrain_proportions = [0.2, 0.05, 0.05, 0.30, 0.05, 0.25, 0.0, 0.0, 0.1]  # 更偏向wave
        terrain_proportions = [0.05, 0.20, 0.05, 0.25, 0.10, 0.20, 0.0, 0.0, 0.15]  # 这个更偏向平地斜坡
        # terrain_proportions = [0.20, 0.05, 0.05, 0.30, 0.15, 0.20, 0.0, 0.0, 0.05]  # 更偏向wave和stairs
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        # terrain_proportions = [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        move_down_by_accumulated_xy_command = True # move down the terrain curriculum based on accumulated xy command distance instead of absolute distance
        
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        # start training with zero commands and then gradually increase zero command probability
        zero_command_curriculum = None
        # zero_command_curriculum = {'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.1}
        limit_ang_vel_at_zero_command_prob = 0.0 # probability of add limiting angular velocity commands when zero command is sampled
        limit_vel_prob = 0.0 # probability of limiting linear velocity command
        limit_vel_invert_when_continuous = True # invert the limit logic when using continuous sample limit velocity commands
        limit_vel = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]} # sample vel commands from min [-1] or zero [0] or max [1] range only
        stop_heading_at_limit = True # stop heading updates when vel is limited
        dynamic_resample_commands = True # sample commands with low bounds
        command_range_curriculum = []
        # command_range_curriculum = [{ # list for command range curriculums at specific training iterations
        #     'iter': 20000, # training iteration at which the command ranges are updated
        #     'lin_vel_x': [-1.0, 1.0], # min max [m/s]
        #     'lin_vel_y': [-1.0, 1.0], # min max [m/s]
        #     'ang_vel_yaw': [-1.5, 1.5], # min max [rad/s]
        #     'heading': [-1.57, 1.57], # min max [rad]
        # }, { # list for command range curriculums at specific training iterations
        #     'iter': 50000, # training iteration at which the command ranges are updated
        #     'lin_vel_x': [-2.0, 2.0], # min max [m/s]
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
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # wave
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # slope
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # rough slope
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs up
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs down
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # obstacles
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stepping stones
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # gap
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0], 'heading': [-1.57, 1.57]},  # flat
        ]

        class ranges:
            lin_vel_x = [-2.0, 2.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0] # min max [m/s]
            ang_vel_yaw = [-2.0, 2.0]   # min max [rad/s]
            heading = [-1.57, 1.57] # min max [rad]
        
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.38
        only_positive_rewards = False
        max_contact_force = 147. # forces above this value are penalized, go2 weight 15kg
        curriculum_rewards = [
            {'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
            {'reward_name': 'correct_base_height', 'start_iter': 0, 'end_iter': 5000, 'start_value': 1.0, 'end_value': 10.0},
            # {'reward_name': 'dof_power', 'start_iter': 0, 'end_iter': 3000, 'start_value': 1.0, 'end_value': 0.1},
            # {'reward_name': 'upright', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
        ]
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        dynamic_sigma = None
        # dynamic_sigma = { # linear interpolation of sigma based on command velocity, **Must start terrain curriculum first**
        #     "min_lin_vel": 0.5, # min abs linear velocity to have default sigma
        #     "max_lin_vel": 1.5, # max abs linear velocity to have max sigma
        #     "min_ang_vel": 1.0, # min abs angular velocity to have default sigma
        #     "max_ang_vel": 2.0, # max abs angular velocity to have max sigma
        #     # wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat]
        #     # "max_sigma": [1/3, 1/4, 1/4, 1/2.7, 1/2.7, 1/2, 1, 1, 1/4]
        #     "max_sigma": [5/12, 1/4, 1/4, 1/2, 1/2, 3/4, 1, 1, 1/4]
        # }
        min_legs_distance = 0.1  # min distance between legs to not be considered stumbling
        class scales:
            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.2
            # lin_vel_z = -10.0
            # base_height = -50.0
            # action_rate = -0.005
            # similar_to_default = -0.1
            # dof_power = -1e-3  # 能够明显抑制跳跃
            # dof_acc = -3e-7

            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.5
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # dof_acc = -2.5e-7
            # dof_power = -1e-3  # 能够明显抑制跳跃
            # # torques = -1e-4  # 无用会走着走着倒了
            # correct_base_height = -10.0
            # action_rate = -0.01
            # action_smoothness = -0.01
            # collision = -1.0
            # dof_pos_limits = -2.0
            # feet_regulation = -0.05
            # hip_to_default = -0.1
            # similar_to_default = -0.05

            # CTS reward
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            dof_power = -2e-5
            torques = -1e-4
            correct_base_height = -1.0
            action_rate = -0.01
            action_smoothness = -0.01
            collision = -1.0
            dof_pos_limits = -2.0
            feet_regulation = -0.05
            # CTS奖励训出来双脚距离非常近, 真机效果很差, 但是sim2sim能上20cm楼梯, 尝试加入hip_to_default奖励或similar_to_default奖励
            hip_to_default = -0.05  # 在训练到y=1.5时, 双脚会明显碰撞, 为避免该问题提升hip, 效果更差, 还是保持0.05 (y最大也只到0.1了)
            # legs_distance = -1.5  # 奖励双脚距离, 避免CTS训练出来双脚距离过近, 尝试加入后robogauge flat验证效果变差, 删除
            # similar_to_default = -0.01
            # feet_contact_forces = -1.0  # 尝试加入但并没有起到任何效果, 删除

        turn_over_roll_threshold = math.pi / 4 # threshold on roll to use turn over rewards
        class turn_over_scales:
            upright = 1.0
            # dof_acc = -2.5e-7
            # dof_power = -2e-5
            # action_rate = -0.001
            # action_smoothness = -0.001

    class noise(LeggedRobotCfg.noise):
        add_noise = True

class GO2CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_ppo'
        max_iterations = 120000
        save_interval = 500

class GO2CfgCTS(LeggedRobotCfgCTS):
    class runner(LeggedRobotCfgCTS.runner):
        num_steps_per_env = 24
        run_name = ''
        experiment_name = 'go2_cts'
        max_iterations = 120000
        save_interval = 500
    
    class policy(LeggedRobotCfgCTS.policy):
        latent_dim = 32
        norm_type = 'l2norm'

class GO2CfgMoECTS(LeggedRobotCfgMoECTS):
    class policy(LeggedRobotCfgMoECTS.policy):
        obs_no_goal_mask = [True] * 6 + [False] * 3 + [True] * 36  # mask for obs without command info
        student_expert_num = 8 # number of experts in the student model
    
    class algorithm(LeggedRobotCfgMoECTS.algorithm):
        load_balance_coef = 0.01

    class runner(LeggedRobotCfgMoECTS.runner):
        run_name = ''
        experiment_name = 'go2_moe_cts'
        max_iterations = 120000
        save_interval = 500

class GO2CfgMCPCTS(LeggedRobotCfgMCPCTS):
    class policy(LeggedRobotCfgMCPCTS.policy):
        obs_no_goal_mask = [True] * 6 + [False] * 3 + [True] * 36  # mask for obs without command info
        student_expert_num = 8 # number of experts in the student model

    class runner(LeggedRobotCfgMCPCTS.runner):
        run_name = ''
        experiment_name = 'go2_mcp_cts'
        max_iterations = 120000
        save_interval = 500

class GO2CfgACMoECTS(LeggedRobotCfgACMoECTS):
    class policy(LeggedRobotCfgACMoECTS.policy):
        expert_num = 8  # number of experts in the student model
    
    class runner(LeggedRobotCfgACMoECTS.runner):
        run_name = ''
        experiment_name = 'go2_ac_moe_cts'
        max_iterations = 120000
        save_interval = 500

class GO2CfgDualMoECTS(LeggedRobotCfgDualMoECTS):
    class policy(LeggedRobotCfgDualMoECTS.policy):
        expert_num = 8  # number of experts in the student model
    
    class runner(LeggedRobotCfgDualMoECTS.runner):
        run_name = ''
        experiment_name = 'go2_dual_moe_cts'
        max_iterations = 120000
        save_interval = 500

class GO2CfgREMCTS(LeggedRobotCfgREMCTS):
    class policy(LeggedRobotCfgREMCTS.policy):
        expert_num = 8  # number of experts in the student model
    
    class runner(LeggedRobotCfgREMCTS.runner):
        run_name = ''
        experiment_name = 'go2_rem_cts'
        max_iterations = 120000
        save_interval = 500
