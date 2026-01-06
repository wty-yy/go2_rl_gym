# -*- coding: utf-8 -*-
'''
@File    : actor_critic_moe_cts.py
@Time    : 2025/12/30 21:06:46
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : Mixture of Experts Concurrent Teacher Student Network
@Refer   : CTS https://arxiv.org/abs/2405.10830, Switch Transformers https://arxiv.org/abs/2101.03961
'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCriticMoECTS(nn.Module):
    is_recurrent = False
    def __init__(self,  num_obs,
                        num_critic_obs,
                        num_actions,
                        num_envs,
                        history_length,
                        obs_no_goal_mask,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        teacher_encoder_hidden_dims=[512, 256],
                        student_encoder_hidden_dims=[512, 256],
                        student_expert_num=8,
                        activation='elu',
                        init_noise_std=1.0,
                        latent_dim=32,
                        norm_type='l2norm',
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        assert norm_type in ['l2norm', 'simnorm'], f"Normalization type {norm_type} not supported!"
        super(ActorCriticMoECTS, self).__init__()
        self.num_actions = num_actions
        self.history_length = history_length
        self.register_buffer("obs_no_goal_mask", torch.tensor(obs_no_goal_mask, dtype=torch.bool), persistent=False)

        activation_str = activation
        activation = get_activation(activation)

        mlp_input_dim_t = num_critic_obs
        mlp_input_dim_e = torch.sum(self.obs_no_goal_mask).item() * history_length  # exclude command inputs for expert
        mlp_input_dim_g = num_obs * history_length  # all obs for gating
        mlp_input_dim_a = latent_dim + num_obs
        mlp_input_dim_c = latent_dim + num_critic_obs

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

        # Student MoE encoder
        self.student_moe_encoder = StudentMoEEncoder(
            expert_dim=mlp_input_dim_e,
            gating_dim=mlp_input_dim_g,
            hidden_dims=student_encoder_hidden_dims,
            expert_num=student_expert_num,
            latent_dim=latent_dim,
            activation=activation_str
        )

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

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

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Teacher Encoder: {self.teacher_encoder}")
        print(f"Student MoE Encoder: {self.student_moe_encoder}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


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

    def update_distribution(self, latent_and_obs):
        mean = self.actor(latent_and_obs)
        self.distribution = Normal(mean, mean*0. + self.std)
    
    def act(self, obs, privileged_obs, history, is_teacher, **kwargs):
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            with torch.no_grad():
                latent, _ = self.get_student_latent_and_weights(history)
        x = torch.cat([latent, obs], dim=1)
        self.update_distribution(x)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs):
        self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
        latent, _ = self.get_student_latent_and_weights(self.history.flatten(1))
        x = torch.cat([latent, obs], dim=1)
        actions_mean = self.actor(x)
        return actions_mean

    def evaluate(self, privileged_obs, history, is_teacher, **kwargs):
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            latent, _ = self.get_student_latent_and_weights(history)
        x = torch.cat([latent.detach(), privileged_obs], dim=1)
        value = self.critic(x)
        return value
    
    def get_student_latent_and_weights(self, history):
        B = history.shape[0]
        history_no_goal = history.reshape(B, self.history_length, -1)[:, :, self.obs_no_goal_mask].reshape(B, -1)
        return self.student_moe_encoder(history, history_no_goal)

class StudentMoEEncoder(nn.Module):
    def __init__(
        self,
        expert_dim,
        gating_dim,
        hidden_dims=[512, 256],
        expert_num=8,
        expert_hidden_dim=256,
        latent_dim=32,
        activation='elu',
        norm_type='l2norm',
    ):
        super().__init__()
        self.expert_num = expert_num
        self.latent_dim = latent_dim
        self.norm_layer = L2Norm() if norm_type == 'l2norm' else SimNorm()
        activation = get_activation(activation)

        # Expert networks
        self.experts = nn.ModuleList()
        for _ in range(expert_num):
            experts_layers = []
            last_dim = gating_dim
            for l in hidden_dims:
                experts_layers.append(nn.Linear(last_dim, l))
                experts_layers.append(activation)
                last_dim = l
            experts_layers.append(nn.Linear(last_dim, latent_dim))
            self.experts.append(nn.Sequential(*experts_layers))

        # Gating network
        gating_layers = []
        last_dim = gating_dim
        for l in hidden_dims:
            gating_layers.append(nn.Linear(last_dim, l))
            gating_layers.append(activation)
            last_dim = l
        gating_layers.append(nn.Linear(last_dim, expert_num))
        gating_layers.append(nn.Softmax(dim=-1))
        self.gating_network = nn.Sequential(*gating_layers)

    def forward(self, obs, obs_no_goal):
        weights = self.gating_network(obs)  # (batch, expert_num)
        
        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(obs))
        expert_latent = torch.stack(expert_outs, dim=1)  # (batch, expert_num, latent_dim)

        latent = torch.sum(weights.unsqueeze(-1) * expert_latent, dim=1)  # (batch, latent_dim)
        latent = self.norm_layer(latent)
        return latent, weights

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
