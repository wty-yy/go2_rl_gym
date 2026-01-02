# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch
from torch import nn
from typing import Optional


def export_policy_as_jit(policy: object, path: str, normalizer: Optional[object] = None, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: Optional[object] = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)



def export_policy_as_pkl(
    policy: nn.Module, path: str, filename="policy.pkl"
):
    """Export policy into a Torch pkl file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported pkl file. Defaults to "policy.pkl".
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    model_dict = policy.state_dict()
    torch.save(model_dict, os.path.join(path, filename))


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "student_encoder"):
            self.student_encoder = copy.deepcopy(policy.student_encoder).cpu()
            self.history = torch.zeros([1, policy.history.shape[1], policy.history.shape[2]], device='cpu')
            self.forward = self.forward_cts
        if hasattr(policy, "student_moe_encoder"):
            self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder).cpu()
            self.obs_no_goal_mask = copy.deepcopy(policy.obs_no_goal_mask).cpu()
            self.history_length = policy.history.shape[1]
            self.history = torch.zeros([1, policy.history.shape[1], policy.history.shape[2]], device='cpu')
            self.forward = self.forward_moe_cts
        if hasattr(policy, "actor_mcp"):
            self.actor = copy.deepcopy(policy.actor_mcp)
            self.obs_no_goal_mask = copy.deepcopy(policy.obs_no_goal_mask).cpu()
            self.forward = self.forward_mcp_cts
        elif hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))
    
    def forward_cts(self, x):  # x is single observations
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        latent = self.student_encoder(self.history.flatten(1))
        x = torch.cat([latent, x], dim=1)
        return self.actor(x), (None, latent)
    
    def forward_moe_cts(self, x):  # x is single observations
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        history_no_goal = self.history.reshape(1, self.history_length, -1)[:, :, self.obs_no_goal_mask].reshape(1, -1)
        latent, weights = self.student_moe_encoder(self.history.flatten(1), history_no_goal)
        x = torch.cat([latent, x], dim=1)
        return self.actor(x), (weights, latent)
    
    def forward_mcp_cts(self, x):  # x is single observations
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        x_no_goal = x[:, self.obs_no_goal_mask]
        latent = self.student_encoder(self.history.flatten(1))
        x = torch.cat([latent, x], dim=1)
        x_no_goal = torch.cat([latent, x_no_goal], dim=1)
        mean_action, _, weights = self.actor(x, x_no_goal)
        return mean_action, (weights, latent)

    @torch.jit.export
    def reset(self):
        if hasattr(self, 'history'):
            self.history = torch.zeros_like(self.history)

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.input_dim = None
        self.num_actions = 12
        self.normalizer = torch.nn.Identity()
        
        # copy policy parameters
        if hasattr(policy, 'student_encoder'):
            self.student_encoder = copy.deepcopy(policy.student_encoder)
            self.forward = self.forward_cts
            self.input_dim = self.student_encoder[0].in_features
            
        elif hasattr(policy, "student_moe_encoder"):
            self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder)
            self.obs_no_goal_mask = copy.deepcopy(policy.obs_no_goal_mask).cpu()
            self.history_length = policy.history.shape[1]
            self.forward = self.forward_moe_cts
            self.input_dim = self.history_length * policy.history.shape[2]
        
        else:  # PPO
            self.forward = self.forward_ppo
            
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if hasattr(self, 'is_recurrent') and self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
            if self.input_dim is None:
                 self.input_dim = self.actor[0].in_features
        elif hasattr(policy, "actor_mcp"):
            self.actor = copy.deepcopy(policy.actor_mcp)
            self.obs_no_goal_mask = copy.deepcopy(policy.obs_no_goal_mask).cpu()
            self.history_length = policy.history.shape[1]
            self.forward = self.forward_mcp_cts 
        else:
            raise ValueError("Policy does not have an actor/student module.")

    def flatten_obs(self, x):  # flatten stack obs by terms to stack by step frames
        term_dims = [3, 3, 3, self.num_actions, self.num_actions, self.num_actions]
        obs_dim = sum(term_dims)
        if x.shape[1] % obs_dim != 0:
            raise ValueError(f"x.shape[1] ({x.shape[1]}) 不是 obs_dim ({obs_dim}) 的整数倍")
            
        frames = x.shape[1] // obs_dim
        split_sizes = [dim * frames for dim in term_dims]
        # [B, dim0*frames], [B, dim1*frames], ...
        term_chunks = torch.split(x, split_sizes, dim=1)

        # [ [B, frames, dim0], [B, frames, dim1], ... ]
        frame_terms_reshaped = [
            chunk.view(-1, frames, dim) 
            for chunk, dim in zip(term_chunks, term_dims)
        ]

        history_by_frame = []
        for i in range(frames):
            # [ [B, dim0], [B, dim1], ... ]
            terms_for_this_frame = [ftr[:, i, :] for ftr in frame_terms_reshaped]
            history_by_frame.append(torch.cat(terms_for_this_frame, dim=1))
        # [B, (Frame0_AllTerms), (Frame1_AllTerms), ...]
        history = torch.cat(history_by_frame, dim=1)
        return history, obs_dim
    
    def forward_ppo(self, x):  # x is stack observations by terms
        x = self.normalizer(x)
        history, obs_dim = self.flatten_obs(x)
        last_obs = history[:, -obs_dim:]
        return self.actor(last_obs)

    def forward_cts(self, x):  # x is stack observations by terms
        x = self.normalizer(x)
        history, obs_dim = self.flatten_obs(x)

        last_obs = history[:, -obs_dim:]
        latent = self.student_encoder(history)
        x = torch.cat([latent, last_obs], dim=1)
        
        return self.actor(x)

    def forward_moe_cts(self, x):
        x = self.normalizer(x)
        history, obs_dim = self.flatten_obs(x)

        last_obs = history[:, -obs_dim:]
        history_3d = history.view(-1, self.history_length, obs_dim)
        history_no_goal = history_3d[:, :, self.obs_no_goal_mask].reshape(x.shape[0], -1)

        latent, weights = self.student_moe_encoder(history, history_no_goal)
        x = torch.cat([latent, last_obs], dim=1)

        return self.actor(x), weights, latent
    
    def forward_mcp_cts(self, x):
        x = self.normalizer(x)
        history, obs_dim = self.flatten_obs(x)

        last_obs = history[:, -obs_dim:]
        obs_no_goal = last_obs[:, self.obs_no_goal_mask]
        latent = self.student_encoder(history)
        x_in = torch.cat([latent, last_obs], dim=1)
        x_no_goal_in = torch.cat([latent, obs_no_goal], dim=1)
        
        mean_action, _, weights = self.actor(x_in, x_no_goal_in)
        return mean_action, weights

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.input_dim)
        
        output_names = ["actions"]
        if self.forward == self.forward_moe_cts:
            output_names.append("weights")
            output_names.append("latent")
        if self.forward == self.forward_mcp_cts:
            output_names.append("weights")

        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=output_names,
            dynamic_axes={},
        )
