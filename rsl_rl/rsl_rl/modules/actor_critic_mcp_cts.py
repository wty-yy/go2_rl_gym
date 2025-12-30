# -*- coding: utf-8 -*-
'''
@File    : actor_critic_moe_cts.py
@Time    : 2025/12/30 21:06:46
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : Multiplicative Compositional Policies Concurrent Teacher Student Network
@Refer   : CTS https://arxiv.org/abs/2405.10830, MCP https://arxiv.org/abs/1905.09808
'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCriticMCPCTS(nn.Module):
    is_recurrent = False
    def __init__(self,  num_obs,
                        num_critic_obs,
                        num_actions,
                        num_envs,
                        history_length,
                        obs_no_goal_mask,
                        actor_hidden_dims=[512, 256],
                        critic_hidden_dims=[512, 256, 128],
                        teacher_encoder_hidden_dims=[512, 256],
                        student_encoder_hidden_dims=[512, 256],
                        student_expert_num=8,
                        activation='elu',
                        latent_dim=32,
                        norm_type='l2norm',
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        assert norm_type in ['l2norm', 'simnorm'], f"Normalization type {norm_type} not supported!"
        super().__init__()
        self.num_actions = num_actions
        self.history_length = history_length
        self.register_buffer("obs_no_goal_mask", torch.tensor(obs_no_goal_mask, dtype=torch.bool), persistent=False)
        self.num_obs_no_goal = torch.sum(self.obs_no_goal_mask).item()

        activation_str = activation
        activation = get_activation(activation)

        mlp_input_dim_t = num_critic_obs
        mlp_input_dim_s = num_obs * history_length
        mlp_input_dim_c = latent_dim + num_critic_obs
        actor_input_dim_g = latent_dim + num_obs
        actor_input_dim_p = latent_dim + self.num_obs_no_goal

        # History
        self.register_buffer("history", torch.zeros((num_envs, history_length, num_obs)), persistent=False)

        # Teacher encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(mlp_input_dim_t, teacher_encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(teacher_encoder_hidden_dims)):
            if l == len(teacher_encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(teacher_encoder_hidden_dims[l], latent_dim))
                if norm_type == 'l2norm':
                    encoder_layers.append(L2Norm())
                elif norm_type == 'simnorm':
                    encoder_layers.append(SimNorm())
            else:
                encoder_layers.append(nn.Linear(teacher_encoder_hidden_dims[l], teacher_encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.teacher_encoder = nn.Sequential(*encoder_layers)

        # Student encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(mlp_input_dim_s, student_encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(student_encoder_hidden_dims)):
            if l == len(student_encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(student_encoder_hidden_dims[l], latent_dim))
                if norm_type == 'l2norm':
                    encoder_layers.append(L2Norm())
                elif norm_type == 'simnorm':
                    encoder_layers.append(SimNorm())
            else:
                encoder_layers.append(nn.Linear(student_encoder_hidden_dims[l], student_encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.student_encoder = nn.Sequential(*encoder_layers)

        # MCP Actor
        self.actor_mcp = ActorMCP(
             input_dim=actor_input_dim_g,
             input_dim_no_goal=actor_input_dim_p,
             action_dim=num_actions,
             hidden_dims=actor_hidden_dims,
             expert_num=student_expert_num,
             activation=activation_str,
        )

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MCP: {self.actor_mcp}")
        print(f"Critic MLP: {self.critic}")
        print(f"Teacher Encoder: {self.teacher_encoder}")
        print(f"Student Encoder: {self.student_encoder}")

        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        self.history[dones > 0] = 0.0

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, x, x_no_goal):
        mean, std, _ = self.actor_mcp(x, x_no_goal)
        self.distribution = Normal(mean, std)

    def act(self, obs, privileged_obs, history, is_teacher, **kwargs):
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            with torch.no_grad():
                latent = self.student_encoder(history)
        x = torch.cat([latent, obs], dim=1)
        obs_no_goal = obs[:, self.obs_no_goal_mask]
        x_no_goal = torch.cat([latent, obs_no_goal], dim=1)
        self.update_distribution(x, x_no_goal)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs):
        self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
        latent = self.student_encoder(self.history.flatten(1))
        x = torch.cat([latent, obs], dim=1)
        obs_no_goal = obs[:, self.obs_no_goal_mask]
        x_no_goal = torch.cat([latent, obs_no_goal], dim=1)
        actions_mean, _, _ = self.actor_mcp(x, x_no_goal)
        return actions_mean

    def evaluate(self, privileged_obs, history, is_teacher, **kwargs):
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            latent = self.student_encoder(history)
        x = torch.cat([latent.detach(), privileged_obs], dim=1)
        value = self.critic(x)
        return value

class ActorMCP(nn.Module):
    def __init__(
        self,
        input_dim,          # latent + full obs
        input_dim_no_goal,  # latent + obs without goal
        action_dim,
        hidden_dims=[512, 256],
        expert_num=8,
        expert_hidden_dim=256,
        activation='elu',
    ):
        super().__init__()
        self.expert_num = expert_num
        self.action_dim = action_dim
        activation = get_activation(activation)

        # Gating network
        gating_layers = []
        last_dim = input_dim
        for l in hidden_dims:
            gating_layers.append(nn.Linear(last_dim, l))
            gating_layers.append(activation)
            last_dim = l
        gating_layers.append(nn.Linear(last_dim, expert_num))
        gating_layers.append(nn.Sigmoid())
        self.gating_network = nn.Sequential(*gating_layers)

        # Expert networks
        expert_layers = []
        last_dim = input_dim_no_goal
        for l in hidden_dims:
            expert_layers.append(nn.Linear(last_dim, l))
            expert_layers.append(activation)
            last_dim = l
        self.experts_backbone = nn.Sequential(*expert_layers)
        self.experts_hidden = nn.Sequential(
            nn.Linear(last_dim, expert_num * expert_hidden_dim),
            activation
        )
        self.experts_out = nn.Conv1d(
             in_channels=expert_num*expert_hidden_dim,
             out_channels=expert_num*action_dim*2,
             kernel_size=1,
             groups=expert_num
        )
    
    def forward(self, x, x_no_goal):
        """
        x: latent + full goal
        x_no_goal: latent + obs without goal
        """
        B = x.shape[0]
        weights = self.gating_network(x).unsqueeze(-1)  # (batch, expert_num, 1)
        shared_features = self.experts_backbone(x_no_goal)
        expert_hidden = self.experts_hidden(shared_features)
        expert_hidden = expert_hidden.unsqueeze(-1)  # (batch, channels, 1)
        expert_out = self.experts_out(expert_hidden)  # (batch, expert_num * action_dim * 2, 1)
        expert_out = expert_out.view(B, self.expert_num, self.action_dim * 2)
        mu, log_std = torch.chunk(expert_out, 2, dim=-1)  # (batch, expert_num, action_dim)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        var = torch.exp(2 * log_std) + 1e-9

        # MCP Composition
        # Formula: var_total = 1 / sum(w_i / var_i)
        #          mu_total = var_total * sum(w_i * mu_i / var_i)
        weighted_sum = torch.sum(weights / var, dim=1) + 1e-9  # (batch, action_dim)
        var_total = 1.0 / weighted_sum
        sigma_total = torch.sqrt(var_total)

        mu_weighted_sum = torch.sum(weights * mu / var, dim=1)  # (batch, action_dim)
        mu_total = var_total * mu_weighted_sum

        return mu_total, sigma_total, weights.squeeze(-1)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

class L2Norm(nn.Module):
    
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return F.normalize(x, p=2.0, dim=-1)

class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self):
		super().__init__()
		self.dim = 8  # for latent dim 512

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"
