# -*- coding: utf-8 -*-
'''
@File    : actor_critic_moe_cts.py
@Time    : 2025/12/30 21:06:46
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : Multiplicative Compositional Policies Concurrent Teacher Student Network
@Refer   :  CTS https://arxiv.org/abs/2405.10830,
            Switch Transformers (Load Balance) https://arxiv.org/abs/2101.03961
            MoE-Loco (AC MoE) http://arxiv.org/abs/2503.08564
'''
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules.utils import MLP, MoE, StudentMoEEncoder, Experts, L2Norm, SimNorm

class ActorCriticDualMoECTS(nn.Module):
    is_recurrent = False
    def __init__(self,  num_obs,
                        num_critic_obs,
                        num_actions,
                        num_envs,
                        history_length,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        teacher_encoder_hidden_dims=[512, 256],
                        student_encoder_hidden_dims=[512, 256, 256],  # last dim is expert hidden dim
                        expert_num=8,
                        activation='elu',
                        init_noise_std=1.0,
                        latent_dim=32,
                        norm_type='l2norm',
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        assert norm_type in ['l2norm', 'simnorm'], f"Normalization type {norm_type} not supported!"
        super().__init__()
        self.num_actions = num_actions
        self.history_length = history_length

        mlp_input_dim_t = num_critic_obs
        mlp_input_dim_s = num_obs * history_length
        mlp_input_dim_c = latent_dim + num_critic_obs
        mlp_input_dim_a = latent_dim + num_obs

        # History
        self.register_buffer("history", torch.zeros((num_envs, history_length, num_obs)), persistent=False)

        # Teacher encoder
        self.teacher_encoder = nn.Sequential(
            MLP([mlp_input_dim_t, *teacher_encoder_hidden_dims, latent_dim], activation),
            L2Norm() if norm_type == 'l2norm' else SimNorm()
        )

        # Student encoder
        self.student_moe_encoder = StudentMoEEncoder(
            expert_num=expert_num,
            input_dim=mlp_input_dim_s,
            hidden_dims=student_encoder_hidden_dims,
            output_dim=latent_dim,
            activation=activation,
            norm_type=norm_type,
        )

        # MCP Actor
        self.actor_moe = MoE(
            expert_num=expert_num,
            input_dim=mlp_input_dim_a,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation=activation,
        )

        # Value function
        self.critic_experts = Experts(
            expert_num=expert_num,
            input_dim=mlp_input_dim_c,
            backbone_hidden_dims=critic_hidden_dims[:-1],
            expert_hidden_dim=critic_hidden_dims[-1],
            output_dim=1,
            activation=activation,
        )

        print(f"Actor MoE: {self.actor_moe}")
        print(f"Critic Experts: {self.critic_experts}")
        print(f"Teacher Encoder: {self.teacher_encoder}")
        print(f"Student MoE Encoder: {self.student_moe_encoder}")

        self.distribution = None
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
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

    def update_distribution(self, x):
        mean, _ = self.actor_moe(x)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, obs, privileged_obs, history, is_teacher, **kwargs):
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            with torch.no_grad():
                latent, _ = self.student_moe_encoder(history)
        x = torch.cat([latent, obs], dim=1)
        self.update_distribution(x)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs):
        self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
        latent, _ = self.student_moe_encoder(self.history.flatten(1))
        x = torch.cat([latent, obs], dim=1)
        mean, _ = self.actor_moe(x)
        return mean

    def evaluate(self, obs, privileged_obs, history, is_teacher, **kwargs):
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            latent, _ = self.student_moe_encoder(history)
        x_actor = torch.cat([latent, obs], dim=1)
        weights = self.actor_moe.gating_network(x_actor)  # (B, expert_num)
        x_critic = torch.cat([latent.detach(), privileged_obs], dim=1)
        experts_value = self.critic_experts(x_critic)
        value = torch.sum(weights.unsqueeze(-1) * experts_value, dim=1)
        return value, weights
