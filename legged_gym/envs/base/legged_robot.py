from itertools import product
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi, quat_apply_yaw
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.isaacgym_utils import sample_disjoint_intervals, sample_single_interval
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from legged_gym.utils.terrain import Terrain

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        self.reward_curriculum_scales = {}
        self.reward_curriculum_configs = []
        if hasattr(self.cfg.rewards, "curriculum_rewards") and self.cfg.rewards.curriculum_rewards is not None:
            self.reward_curriculum_configs = self.cfg.rewards.curriculum_rewards
            for config in self.reward_curriculum_configs:
                self.reward_curriculum_scales[config['reward_name']] = config['start_value']
        self.num_steps_per_env = 24  # PPO default num_steps_per_env

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        if self.cfg.domain_rand.randomize_action_delay:
            actions_start_decimation = torch.randint(0, self.cfg.control.decimation+1, (self.num_envs, 1), device=self.device)
        for i in range(self.cfg.control.decimation):
            if self.cfg.domain_rand.randomize_action_delay:
                use_actions = (i >= actions_start_decimation).float()
                input_actions = (1 - use_actions) * self.last_actions + use_actions * self.actions
            else:
                input_actions = self.actions
            self.torques = self._compute_torques(input_actions).view(self.torques.shape)
            if self.cfg.domain_rand.randomize_motor_strength:
                self.torques *= self.motor_strengths
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.commands_resampling_step -= 1
        if self.cfg.init_state.turn_over:
            self.turn_over_timer = (self.turn_over_timer - self.dt).clip(min=0.0)

        self.update_reward_curriculum()
        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.max_move_distance = self.max_move_distance.maximum(torch.norm(self.root_states[:, :2] - self.env_origins[:, :2], dim=1))

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
    
    def update_reward_curriculum(self, force_update: bool = False):
        # update reward curriculum
        if self.reward_curriculum_configs:
            if self.common_step_counter % self.num_steps_per_env == 0 or force_update:
                
                for config in self.reward_curriculum_configs:
                    current_scale = self.get_current_scale(config)
                    reward_name = config['reward_name']
                    self.reward_curriculum_scales[reward_name] = current_scale
    
    def get_current_scale(self, config):
        """ config: Dict
            {'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0}
        """
        current_iter = self.common_step_counter // self.num_steps_per_env
        cfg_start_iter = config['start_iter']
        cfg_end_iter = config['end_iter']
        cfg_start_val = config['start_value']
        cfg_end_val = config['end_value']

        percentage = (current_iter - cfg_start_iter) / (cfg_end_iter - cfg_start_iter)
        percentage = max(min(percentage, 1.0), 0.0)
        
        current_scale = (1.0 - percentage) * cfg_start_val + percentage * cfg_end_val
        return current_scale

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not self.cfg.init_state.turn_over:
            self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        ### Domain randomizations ###
        # randomization of the motor strength
        if self.cfg.domain_rand.randomize_motor_strength:
            rng = self.cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids] = torch_rand_float(
                rng[0], rng[1], (len(env_ids), self.num_actions), device=self.device
            )
        # randomization of the motor zero calibration for real machine
        if self.cfg.domain_rand.randomize_motor_zero_offset:
            self.motor_zero_offsets[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_zero_offset_range[0], self.cfg.domain_rand.motor_zero_offset_range[1], (len(env_ids), self.num_actions), device=self.device)
        # randomization of the motor pd gains
        if self.cfg.domain_rand.randomize_pd_gains:
            self.p_gains_multiplier[env_ids] = torch_rand_float(self.cfg.domain_rand.stiffness_multiplier_range[0], self.cfg.domain_rand.stiffness_multiplier_range[1], (len(env_ids), self.num_actions), device=self.device)
            self.d_gains_multiplier[env_ids] =  torch_rand_float(self.cfg.domain_rand.damping_multiplier_range[0], self.cfg.domain_rand.damping_multiplier_range[1], (len(env_ids), self.num_actions), device=self.device)

        # update terrain curriculum before reset root states
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.commands_resampling_step[env_ids] = self.cfg.commands.resampling_time / self.dt
        self.commands_xy_accumulation[env_ids] = 0.0
        if self.cfg.commands.curriculum:
            self.update_command_curriculum(env_ids)
        self._resample_commands(env_ids)
        # fill extras
        self.extras["episode"] = {}
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.extras["episode"]['terrain_level_all'] = torch.mean(self.terrain_levels.float())
            for name, cols in self.terrain.name2cols.items():
                if isinstance(cols, set):
                    cols = self.terrain.name2cols[name] = torch.tensor(list(cols), device=self.device)
                self.extras["episode"]['terrain_level_' + name] = torch.mean(self.terrain_levels[torch.isin(self.terrain_types, cols)].float())
        else:
            self.extras["episode"]['terrain_level_all'] = 0.0
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            raw_rew = self.reward_functions[i]()
            rew = raw_rew * self.reward_scales.get(name, 0.0)
            if self.cfg.init_state.turn_over:
                turn_over_rew = raw_rew * self.reward_turn_over_scales.get(name, 0.0)
            if name in self.reward_curriculum_scales:
                rew *= self.reward_curriculum_scales[name]
                if self.cfg.init_state.turn_over:
                    turn_over_rew *= self.reward_curriculum_scales[name]
            if self.cfg.init_state.turn_over:
                need_turn_over = self.rpy[:, 0].abs() > self.cfg.rewards.turn_over_roll_threshold
                rew = torch.where(need_turn_over, turn_over_rew, rew)
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        
        if self.cfg.domain_rand.randomize_restitution:
            rand_restitution = np.random.uniform(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1])
            for s in range(len(props)):
                props[s].restitution = rand_restitution
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])

        # randomize link masses
        if self.cfg.domain_rand.randomize_link_mass:
            self.multiplied_link_masses_ratio = torch_rand_float(self.cfg.domain_rand.multiplied_link_mass_range[0], self.cfg.domain_rand.multiplied_link_mass_range[1], (1, self.num_bodies-1), device=self.device)
            for i in range(1, len(props)):
                props[i].mass *= self.multiplied_link_masses_ratio[0,i-1]

        # randomize base com
        if self.cfg.domain_rand.randomize_base_com:
            self.added_base_com = torch_rand_float(self.cfg.domain_rand.added_base_com_range[0], self.cfg.domain_rand.added_base_com_range[1], (1, 3), device=self.device)
            props[0].com += gymapi.Vec3(self.added_base_com[0, 0], self.added_base_com[0, 1],
                                    self.added_base_com[0, 2])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        resampling_env_ids = ((self.commands_resampling_step <= 0.0) * (self.episode_length_buf < self.max_episode_length - 1)).nonzero(as_tuple=False).flatten()
        self._resample_commands(resampling_env_ids)
        if self.cfg.commands.heading_command:
            mask = (self.stop_heading == 0.0)
            forward = quat_apply(self.base_quat[mask], self.forward_vec[mask])
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[mask, 2] = torch.clip(
                0.5*wrap_to_pi(self.commands[mask, 3] - heading),
                self.env_command_ranges["ang_vel_yaw"][:, 0],
                self.env_command_ranges["ang_vel_yaw"][:, 1]
            )
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return
        self.stop_heading[env_ids] = False
        # update command curriculum with train steps
        if len(self.cfg.commands.command_range_curriculum):
            current_iter = self.common_step_counter // self.num_steps_per_env
            for i in range(len(self.cfg.commands.command_range_curriculum)-1, -1, -1):  # iterate backwards to be able to pop entries
                cfg = self.cfg.commands.command_range_curriculum[i]
                if current_iter >= cfg["iter"]:
                    self.command_ranges["lin_vel_x"] = cfg["lin_vel_x"]
                    self.command_ranges["lin_vel_y"] = cfg["lin_vel_y"]
                    self.command_ranges["ang_vel_yaw"] = cfg["ang_vel_yaw"]
                    self.command_ranges["heading"] = cfg["heading"]
                    self.max_lin_vel = max(abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]),
                                           abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]))
                    self.cfg.commands.command_range_curriculum.pop(i)
                    self._update_env_command_ranges()
                    print(f"Command range updated at iter {current_iter}: {self.command_ranges}")
        remaining_dist = torch.clip(0.625 * self.cfg.terrain.terrain_length - torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * self.cfg.commands.resampling_time, 0.0)
        self.commands_resampling_step[env_ids] = self.cfg.commands.resampling_time / self.dt
        if self.cfg.commands.dynamic_resample_commands:
            # arrive at boundary 0.625 times the width of the remaining distance
            if ((self.max_episode_length - self.episode_length_buf[env_ids]) == 0).any():
                raise ValueError("Some envs have zero remaining episode length during command resampling")
            vel_low_bound = torch.clip(remaining_dist / ((self.max_episode_length - self.episode_length_buf[env_ids] + 1e-9) * self.dt), 0.0)
            self.commands[env_ids, 0] = sample_disjoint_intervals(
                env_ids,
                vel_low_bound,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device
            )
            self.commands[env_ids, 1] = sample_disjoint_intervals(
                env_ids,
                vel_low_bound,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device
            )
            if self.cfg.commands.heading_command:
                r = torch.rand(len(env_ids), device=self.device)
                lower = self.env_command_ranges["heading"][env_ids, 0]
                upper = self.env_command_ranges["heading"][env_ids, 1]
                self.commands[env_ids, 3] = (upper - lower) * r + lower
            else:
                r = torch.rand(len(env_ids), device=self.device)
                lower = self.env_command_ranges["ang_vel_yaw"][env_ids, 0]
                upper = self.env_command_ranges["ang_vel_yaw"][env_ids, 1]
                self.commands[env_ids, 2] = (upper - lower) * r + lower
        else:
            self.commands[env_ids, 0] = sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device
            )
            self.commands[env_ids, 1] = sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device
            )
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = sample_single_interval(
                    env_ids,
                    self.env_command_ranges["heading"][env_ids, 0],
                    self.env_command_ranges["heading"][env_ids, 1],
                    self.device
                )
            else:
                self.commands[env_ids, 2] = sample_single_interval(
                    env_ids,
                    self.env_command_ranges["ang_vel_yaw"][env_ids, 0],
                    self.env_command_ranges["ang_vel_yaw"][env_ids, 1],
                    self.device
                )

            # set small commands to zero
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        rand_prob = torch.rand(len(env_ids), device=self.device)
        min_prob, max_prob = 0.0, 0.0
        # set limitation lin vel
        if self.limit_vel_prob > 0.0:
            max_prob += self.limit_vel_prob
            lim_mask = (rand_prob >= min_prob) * (rand_prob < max_prob)
            lim_env_ids = env_ids[lim_mask]
            if len(lim_env_ids) > 0:
                change_lim_env_ids = lim_env_ids
                if self.cfg.commands.limit_vel_invert_when_continuous:
                    was_limited = self.last_is_limit_vel[lim_env_ids]
                    invert_env_ids = lim_env_ids[was_limited]
                    self.commands[invert_env_ids, 0] *= -1.0
                    self.commands[invert_env_ids, 1] *= -1.0
                    self.commands[invert_env_ids, 2] *= -1.0
                    change_lim_env_ids = lim_env_ids[~was_limited]
                vel_idx = torch.randint(0, self.limit_vel_comb.shape[0], (len(change_lim_env_ids),), device=self.device)
                lin_vel_x_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 0] == -1,
                    self.env_command_ranges["lin_vel_x"][change_lim_env_ids, 0],
                    self.env_command_ranges["lin_vel_x"][change_lim_env_ids, 1],
                )
                lin_vel_x_lim[self.limit_vel_comb[vel_idx, 0] == 0] = 0.0
                lin_vel_y_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 1] == -1,
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 0],
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 1]
                )
                lin_vel_y_lim[self.limit_vel_comb[vel_idx, 1] == 0] = 0.0
                ang_vel_z_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 2] == -1,
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 0],
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 1]
                )
                ang_vel_z_lim[self.limit_vel_comb[vel_idx, 2] == 0] = 0.0
                self.commands[change_lim_env_ids, 0] = lin_vel_x_lim
                self.commands[change_lim_env_ids, 1] = lin_vel_y_lim
                self.commands[change_lim_env_ids, 2] = ang_vel_z_lim
                if self.cfg.commands.heading_command and self.cfg.commands.stop_heading_at_limit:
                    self.stop_heading[lim_env_ids] = True # stop heading to current heading
                self.last_is_limit_vel[env_ids] = False
                self.last_is_limit_vel[lim_env_ids] = True
            else:
                self.last_is_limit_vel[env_ids] = False
            min_prob += self.limit_vel_prob

        # set all commands to zero with some probability
        if self.cfg.commands.zero_command_curriculum is not None:
            self.zero_command_proba = self.get_current_scale(self.cfg.commands.zero_command_curriculum)
        if self.zero_command_proba > 0.0:
            max_prob += self.zero_command_proba
            next_resampling_step = torch.clip(
                self.max_episode_length - self.episode_length_buf[env_ids] - (remaining_dist / (0.8 * self.max_lin_vel * self.dt + 1e-9)),
                min=0.0,
                max=self.cfg.commands.resampling_time / self.dt,
            )
            zero_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) * (next_resampling_step > 0.0)
            zero_env_ids = env_ids[zero_mask]
            if len(zero_env_ids) > 0:
                self.commands[zero_env_ids, :2] = 0.0
                self.commands_resampling_step[zero_env_ids] = next_resampling_step[zero_mask]
                if self.cfg.commands.limit_ang_vel_at_zero_command_prob > 0.0:
                    ang_vel_rand = torch.rand(len(zero_env_ids), device=self.device) # independent distribution
                    add_ang_mask = ang_vel_rand < self.cfg.commands.limit_ang_vel_at_zero_command_prob
                    add_ang_env_ids = zero_env_ids[add_ang_mask]
                    if len(add_ang_env_ids) > 0:
                        direction_rand = torch.rand(len(add_ang_env_ids), device=self.device)
                        self.commands[add_ang_env_ids, 2] = torch.where(
                            direction_rand < 0.5,
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 0],
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 1]
                        )
                        if self.cfg.commands.heading_command:
                            self.stop_heading[add_ang_env_ids] = True
            min_prob += self.zero_command_proba

        # turn over zero command time
        if self.cfg.init_state.turn_over and (self.turn_over_timer[env_ids] > 0).any():
            zero_mask = self.turn_over_timer[env_ids] > 0
            zero_env_ids = env_ids[zero_mask]
            self.commands[zero_env_ids, :3] = 0.0
            self.stop_heading[zero_env_ids] = True

        self.commands_xy_accumulation[env_ids] += self.commands[env_ids, :2]

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        p_gains = self.p_gains * self.p_gains_multiplier
        d_gains = self.d_gains * self.d_gains_multiplier
        if control_type=="P":
            torques = p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_zero_offsets) - d_gains*self.dof_vel
        elif control_type=="V":
            torques = p_gains*(actions_scaled - self.dof_vel) - d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
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
        if self.cfg.init_state.turn_over:
            self.turn_over_timer[env_ids] = 0.0
        # base position
        random_yaw = torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        def get_quat(target_yaws, roll: float):
            roll_tensor = torch.full((len(target_yaws),), roll, device=self.device)
            pitch_tensor = torch.zeros((len(target_yaws),), device=self.device)
            quat = quat_from_euler_xyz(roll_tensor, pitch_tensor, target_yaws)
            return quat

        base_init_state = self.base_init_state.reshape(1, -1).repeat(len(env_ids), 1)
        if self.cfg.init_state.turn_over:
            rand_prob = torch.rand(len(env_ids), device=self.device)
            proportions = self.cfg.init_state.turn_over_proportions
            init_heights = self.cfg.init_state.turn_over_init_heights

            min_prob, max_prob = 0.0, proportions[0]
            back_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) # backflip
            if back_mask.any():
                heights = torch_rand_float(init_heights['backflip'][0], init_heights['backflip'][1], (torch.sum(back_mask), 1), device=self.device).squeeze(1)
                base_init_state[back_mask, 2] = heights # z
                base_init_state[back_mask, 3:7] = get_quat(random_yaw[back_mask], np.pi)
                if self.cfg.init_state.turn_over:
                    self.turn_over_timer[env_ids[back_mask]] = self.cfg.commands.turn_over_zero_time['backflip']

            min_prob = max_prob
            max_prob += proportions[1]
            side_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) # sideflip
            if side_mask.any():
                side_ids = torch.nonzero(side_mask, as_tuple=False).flatten()
                heights = torch_rand_float(init_heights['sideflip'][0], init_heights['sideflip'][1], (len(side_ids), 1), device=self.device).squeeze(1)
                base_init_state[side_mask, 2] = heights # z
                side_rand_prob = torch.rand(len(side_ids), device=self.device)
                pos_side_mask = side_rand_prob < 0.5
                neg_side_mask = ~pos_side_mask
                if pos_side_mask.any():
                    pos_ids = side_ids[pos_side_mask]
                    base_init_state[pos_ids, 3:7] = get_quat(random_yaw[pos_ids], np.pi/2)
                if neg_side_mask.any():
                    neg_ids = side_ids[neg_side_mask]
                    base_init_state[neg_ids, 3:7] = get_quat(random_yaw[neg_ids], -np.pi/2)
                if self.cfg.init_state.turn_over:
                    self.turn_over_timer[env_ids[side_mask]] = self.cfg.commands.turn_over_zero_time['sideflip']
            
            min_prob = max_prob
            max_prob += proportions[2]
            noflip_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) # noflip
            if noflip_mask.any():
                noflip_indices = torch.nonzero(noflip_mask, as_tuple=False).flatten()
                base_init_state[noflip_mask, 3:7] = get_quat(random_yaw[noflip_indices], 0.0)
        else:
            base_init_state[:, 3:7] = get_quat(random_yaw, 0.0)
                
        if self.custom_origins:
            self.root_states[env_ids] = base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_ang = self.cfg.domain_rand.max_push_ang_vel
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.root_states[:, 10:13] = torch_rand_float(-max_push_ang, max_push_ang, (self.num_envs, 3), device=self.device) # ang vel x/y/z
        
        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

   
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            x_points = self.height_points[0, :, 0]
            y_points = self.height_points[0, :, 1]
            x_mask = (x_points >= -0.2) & (x_points <= 0.2)  # 0.4m length
            y_mask = (y_points >= -0.15) & (y_points <= 0.15)  # 0.3m width
            self.base_height_scan_mask = (x_mask & y_mask).float()
            self.num_base_height_scan_points = self.base_height_scan_mask.sum()
            assert self.num_base_height_scan_points > 0, "No height scan points within the specified area."
        self.measured_heights = 0

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.commands_resampling_step = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_xy_accumulation = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.zero_command_proba = 0.0
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.max_move_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.stop_heading = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_is_limit_vel = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.limit_vel_prob = self.cfg.commands.limit_vel_prob
        self.limit_vel_comb = torch.tensor(list(product(
            self.cfg.commands.limit_vel["lin_vel_x"],
            self.cfg.commands.limit_vel["lin_vel_y"],
            self.cfg.commands.limit_vel["ang_vel_yaw"]
        )), device=self.device, requires_grad=False)
        self.last_robot_props_update_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.turn_over_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.env_command_ranges = {
            'lin_vel_x': torch.tensor(self.command_ranges['lin_vel_x'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
            'lin_vel_y': torch.tensor(self.command_ranges['lin_vel_y'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
            'ang_vel_yaw': torch.tensor(self.command_ranges['ang_vel_yaw'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
            'heading': torch.tensor(self.command_ranges['heading'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
        }
        self._update_env_command_ranges()

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
    
    def _update_env_command_ranges(self):
        """ Update environment-wise command ranges based on current command ranges and terrain type """
        if not hasattr(self, 'terrain_ids'):
            self.env_command_ranges = {
                'lin_vel_x': torch.tensor(self.command_ranges['lin_vel_x'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
                'lin_vel_y': torch.tensor(self.command_ranges['lin_vel_y'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
                'ang_vel_yaw': torch.tensor(self.command_ranges['ang_vel_yaw'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
                'heading': torch.tensor(self.command_ranges['heading'], device=self.device, requires_grad=False).repeat(self.num_envs, 1),
            }
            return
        for terrain_id, terrain_command_ranges in enumerate(self.cfg.commands.terrain_max_command_ranges):
            env_ids = (self.terrain_ids == terrain_id).nonzero(as_tuple=False).flatten()
            if len(env_ids) == 0:
                continue
            self.env_command_ranges['lin_vel_x'][env_ids, 0] = max(
                terrain_command_ranges['lin_vel_x'][0],
                self.command_ranges['lin_vel_x'][0],
            )
            self.env_command_ranges['lin_vel_x'][env_ids, 1] = min(
                terrain_command_ranges['lin_vel_x'][1],
                self.command_ranges['lin_vel_x'][1]
            )
            self.env_command_ranges['lin_vel_y'][env_ids, 0] = max(
                terrain_command_ranges['lin_vel_y'][0],
                self.command_ranges['lin_vel_y'][0]
            )
            self.env_command_ranges['lin_vel_y'][env_ids, 1] = min(
                terrain_command_ranges['lin_vel_y'][1],
                self.command_ranges['lin_vel_y'][1]
            )
            self.env_command_ranges['ang_vel_yaw'][env_ids, 0] = max(
                terrain_command_ranges['ang_vel_yaw'][0],
                self.command_ranges['ang_vel_yaw'][0]
            )
            self.env_command_ranges['ang_vel_yaw'][env_ids, 1] = min(
                terrain_command_ranges['ang_vel_yaw'][1],
                self.command_ranges['ang_vel_yaw'][1]
            )
            if self.cfg.commands.heading_command:
                self.env_command_ranges['heading'][env_ids, 0] = max(
                    terrain_command_ranges['heading'][0],
                    self.command_ranges['heading'][0]
                )
                self.env_command_ranges['heading'][env_ids, 1] = min(
                    terrain_command_ranges['heading'][1],
                    self.command_ranges['heading'][1]
                )

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        def update_scales(scales):
            for key in list(scales.keys()):
                scale = scales[key]
                if scale==0:
                    scales.pop(key) 
                else:
                    scales[key] *= self.dt
        update_scales(self.reward_scales)
        if self.cfg.init_state.turn_over:
            update_scales(self.reward_turn_over_scales)
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        names = set()
        names.update(list(self.reward_scales.keys()))
        if self.cfg.init_state.turn_over:
            names.update(list(self.reward_turn_over_scales.keys()))
        for name in names:
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in names}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        hip_names = [s for s in self.dof_names if 'hip' in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # domain rand
        self.motor_zero_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains_multiplier = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains_multiplier = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.rewards.dynamic_sigma:
            self.dynamic_sigma_cfg = self.cfg.rewards.dynamic_sigma
            self.terrain_max_sigmas = torch.tensor(self.dynamic_sigma_cfg["max_sigma"], device=self.device, requires_grad=False)

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            if i == 0:
                self.default_body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hip_names)):
            self.hip_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], hip_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1

            # random choice terrain levels and types for each env
            # self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            # self.terrain_types = torch.randint(0, self.cfg.terrain.num_cols, (self.num_envs,), device=self.device)

            # levels and types in a round robin manner
            self.terrain_levels = torch.fmod(torch.arange(self.num_envs, device=self.device), max_init_level + 1)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs / self.cfg.terrain.num_cols), rounding_mode="floor").to(torch.long)
            self.terrain_cols2id = torch.tensor(self.terrain.cols2id, device=self.device)
            if len(self.terrain_cols2id):
                self.terrain_ids = self.terrain_cols2id[self.terrain_types]

            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.reward_turn_over_scales = class_to_dict(self.cfg.rewards.turn_over_scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.max_lin_vel = max(abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]),
                               abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]))
        self.cfg.commands.command_range_curriculum = sorted(self.cfg.commands.command_range_curriculum, key=lambda x: x['iter'], reverse=True)

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

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
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done or self.cfg.terrain.mesh_type == 'plane':
            # don't change on initial reset
            return
        # distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        distance = self.max_move_distance[env_ids]
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        if self.cfg.terrain.move_down_by_accumulated_xy_command:
            move_down = (distance < torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * (self.cfg.commands.resampling_time * (1 - self.zero_command_proba)) * 0.5) * ~move_up
        else:
            # robots that walked less than half of their required distance go to simpler terrains
            move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.max_move_distance[env_ids] = 0.0
        

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale


    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = self.root_states[:, 2]
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_base_height(self):
        # Penalize base height away from target
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        if not hasattr(self, 'last_contacts2'):
            self.last_contacts2 = torch.zeros_like(contact)
        contact_filt = torch.logical_or(contact, self.last_contacts2)  # (N, 4)
        self.last_contacts2 = contact
        feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        num_feet_contact = torch.sum(contact_filt, dim=1, keepdim=True).clamp(min=1.0)  # (N, 1)
        feet_contact_pos = (feet_pos * contact_filt.unsqueeze(-1)).sum(dim=1) / num_feet_contact  # (N, 3)
        base_pos = self.root_states[:, 0:3]
        delta_pos = feet_contact_pos - base_pos
        base_height = (delta_pos * self.projected_gravity).sum(1)  # (N,)
        rew = torch.square(base_height - self.cfg.rewards.base_height_target) * (contact_filt.sum(1) > 0)
        return rew

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _get_dynamic_sigma(self, target_vel_abs, v_min, v_max):
        # compute dynamic sigma based on terrain level
        default_sigma = self.cfg.rewards.tracking_sigma
        if not self.cfg.terrain.curriculum or self.cfg.rewards.dynamic_sigma is None or not hasattr(self, 'terrain_ids'):
            return torch.full_like(target_vel_abs, default_sigma)
        target_sigmas = self.terrain_max_sigmas[self.terrain_ids]
        sigma = torch.full_like(target_vel_abs, default_sigma)
        # based on velocity ranges, compute sigma
        # v_min <= v < v_max (linear interpolation)
        mask = (target_vel_abs >= v_min) & (target_vel_abs < v_max)
        if mask.any():
            ratio = (target_vel_abs[mask] - v_min) / (v_max - v_min)
            sigma[mask] = default_sigma + ratio * (target_sigmas[mask] - default_sigma)
        # v >= v_max
        mask = target_vel_abs >= v_max
        if mask.any():
            sigma[mask] = target_sigmas[mask]
        # based on terrain level, compute sigma
        level_scale = torch.clamp(torch.exp((self.terrain_levels.float() + 1.0) / 10.0) - 1.0, max=1.0)
        sigma = default_sigma + level_scale * (sigma - default_sigma)
        return sigma

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        if self.cfg.rewards.dynamic_sigma is None:
            sigma_x = sigma_y = self.cfg.rewards.tracking_sigma
        else:
            vmin = self.dynamic_sigma_cfg["min_lin_vel"]
            vmax = self.dynamic_sigma_cfg["max_lin_vel"]
            sigma_x = self._get_dynamic_sigma(torch.abs(self.commands[:, 0]), vmin, vmax)
            sigma_y = self._get_dynamic_sigma(torch.abs(self.commands[:, 1]), vmin, vmax)
        lin_vel_error_sq = torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2])
        scaled_error = lin_vel_error_sq[:, 0] / sigma_x + lin_vel_error_sq[:, 1] / sigma_y
        # print(f"{self.base_lin_vel[:, :2]=}, {lin_vel_error_sq=}")
        return torch.exp(-scaled_error)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        if self.cfg.rewards.dynamic_sigma is None:
            sigma = self.cfg.rewards.tracking_sigma
        else:
            vmin = self.dynamic_sigma_cfg["min_ang_vel"]
            vmax = self.dynamic_sigma_cfg["max_ang_vel"]
            sigma = self._get_dynamic_sigma(torch.abs(self.commands[:, 2]), vmin, vmax)
        ang_vel_error_sq = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error_sq/sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_action_smoothness(self):
        # a_t - 2a_{t-1} + a_{t-2}
        if not hasattr(self, 'last_last_actions'):
            self.last_last_actions = torch.zeros_like(self.last_actions)
        rew = torch.sum((self.actions - 2 * self.last_actions + self.last_last_actions).pow(2), dim=1)
        self.last_last_actions[:] = self.last_actions[:]
        return rew
    
    def _reward_dof_power(self):
        # Penalize power consumption
        power = self.torques * self.dof_vel
        rew = torch.sum(torch.abs(power), dim=1)
        return rew

    def _get_base_height(self):
        if not self.cfg.terrain.measure_heights:
            return self.root_states[:, 2]
        # 根据高度扫描点计算base link到地面估计高度
        masked_heights = self.measured_heights * self.base_height_scan_mask.unsqueeze(0)
        sum_heights = masked_heights.sum(dim=1)
        estimated_ground_z = sum_heights / self.num_base_height_scan_points

        base_z = self.root_states[:, 2] 
        base_height = base_z - estimated_ground_z  # (N,)
        return base_height

    def _reward_correct_base_height(self):
        base_height = self._get_base_height()
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)
        return rew

    def _reward_feet_regulation(self):
        # CTS抬腿正则奖励, 在脚末端速度增大同时, 要求高度尽可能高
        base_height = self._get_base_height()  # 更新刚体空间位置 (开悟比赛无法修改环境, 只能在奖励中完成计算了)
        feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        feet_xy_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:9]
        base_pos = self.root_states[:, 0:3].unsqueeze(1)
        delta_feet = feet_pos - base_pos
        feet2base_height = (delta_feet * self.projected_gravity.unsqueeze(1)).sum(-1)  # 脚相对于身体的高度 (N, 4)
        feet_height = torch.clamp(base_height.unsqueeze(1) - feet2base_height, min=0.0)  # 脚相对于地面的高度 (N, 4)
        rew = (feet_xy_vel.pow(2).sum(-1) * torch.exp(-feet_height / (0.025 * self.cfg.rewards.base_height_target))).sum(-1)
        return rew

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_upright(self):
        return (-1 - self.projected_gravity[:, 2]) / 2
    
    def _reward_legs_distance(self):
        # Penalize legs being too close to each other
        # feet_names: [FL_foot, FR_foot, RL_foot, RR_foot]
        feet_pos_world = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]  # (N, 4, 3)
        base_pos = self.root_states[:, 0:3]  # (N, 3)
        base_quat = self.base_quat  # (N, 4)
        feet_pos_relative_world = feet_pos_world - base_pos.unsqueeze(1)  # (N, 4, 3)
        local_pos = quat_rotate_inverse(
            base_quat.repeat_interleave(4, dim=0),
            feet_pos_relative_world.reshape(-1, 3)
        ).reshape(self.num_envs, 4, 3)  # (N, 4, 3)

        dy_front =  local_pos[:, 0, 1] - local_pos[:, 1, 1]  # (N,)
        dy_rear =  local_pos[:, 2, 1] - local_pos[:, 3, 1]   # (N,)
        min_dist = self.cfg.rewards.min_legs_distance

        rew_front = torch.square(torch.clamp(min_dist - dy_front, min=0.0))
        rew_rear = torch.square(torch.clamp(min_dist - dy_rear, min=0.0))
        return rew_front + rew_rear
