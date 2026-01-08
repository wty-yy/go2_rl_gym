import os
import numpy as np
import random
import torch

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

def sample_disjoint_intervals(env_ids, limit_bound, cfg_min, cfg_max, device):
    """
    sample uniform distribution from [cfg_min, -limit_bound] U [limit_bound, cfg_max]
    """
    width_neg = torch.nn.functional.relu(-limit_bound - cfg_min)
    width_pos = torch.nn.functional.relu(cfg_max - limit_bound)
    
    total_width = width_neg + width_pos + 1e-6 # 加极小值防除零
    u = torch.rand(len(env_ids), device=device) * total_width
    
    samples = torch.where(
        u < width_neg, 
        cfg_min + u, 
        cfg_max - width_pos + (u - width_neg)
    )
    return samples

def sample_single_interval(env_ids, cfg_min, cfg_max, device):
    """
    sample uniform distribution from [cfg_min, cfg_max]
    """
    r = torch.rand(len(env_ids), device=device)
    samples = cfg_min + r * (cfg_max - cfg_min)
    return samples