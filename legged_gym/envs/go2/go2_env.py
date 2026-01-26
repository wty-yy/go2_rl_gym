
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class Go2Robot(LeggedRobot):
    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions

        return noise_vec
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.base_ang_vel  * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3] * self.commands_scale,
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions,
                                  ),dim=-1)
        
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.obs_scales.height_measurements
        
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) * 1e-3,  # foot contact forces (4,)
                                    self.torques / self.torque_limits,  # motor torques (12,)
                                    (self.last_dof_vel - self.dof_vel) / self.dt * 1e-4,  # motor accelerations (12,)
                                    heights,  # height measurements (187,)
                                    ),dim=-1)
        # print(f"foot contact: {self.privileged_obs_buf[:,48:48+4].min(), self.privileged_obs_buf[:,48:48+4].max()}")
        # print(f"torques: {self.privileged_obs_buf[:,48+4:48+4+12].min(), self.privileged_obs_buf[:,48+4:48+4+12].max()}")
        # print(f"acc: {self.privileged_obs_buf[:,48+4+12:48+4+12+12].min(), self.privileged_obs_buf[:,48+4+12:48+4+12+12].max()}")
        
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_hip_to_default(self):
        hip_dof_names = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint']
        hip_dof_indices = [0, 3, 6, 9]
        hip_pos = self.dof_pos[:, hip_dof_indices]
        default_hip_pos = self.default_dof_pos[:, hip_dof_indices]
        return torch.sum(torch.abs(hip_pos - default_hip_pos), dim=1)

    def _reward_x_command_hip_regular(self):
        hip_dof_names = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint']
        hip_dof_indices = [0, 3, 6, 9]
        hip_pos = self.dof_pos[:, hip_dof_indices]
        x_command_ratio = torch.abs(self.commands[:,0]) / torch.norm(self.commands[:,:3], dim=1)
        rew = torch.abs(hip_pos[:,0]+hip_pos[:,1]) + torch.abs(hip_pos[:,2]+hip_pos[:,3])
        return rew * x_command_ratio
