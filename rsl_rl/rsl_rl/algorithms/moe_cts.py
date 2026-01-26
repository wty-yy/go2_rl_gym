# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

import itertools
from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.storage import RolloutStorageCTS
from rsl_rl.algorithms.cts import CTS

class MoECTS(CTS):
    model: ActorCriticMoECTS
    def __init__(self,
                model,
                num_envs,
                history_length,
                num_learning_epochs=1,
                num_mini_batches=1,
                clip_param=0.2,
                gamma=0.998,
                lam=0.95,
                value_loss_coef=1.0,
                entropy_coef=0.0,
                load_balance_coef=0.01,
                learning_rate=1e-3,
                student_encoder_learning_rate=1e-3,
                max_grad_norm=1.0,
                use_clipped_value_loss=True,
                schedule="fixed",
                desired_kl=0.01,
                teacher_env_ratio=0.75,
                device='cpu',
                ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.history_length = history_length

        # CTS components
        self.model = model
        self.model.to(self.device)
        self.storage = None # initialized later
        params1 = [
            {"params": self.model.teacher_encoder.parameters()},
            {"params": self.model.critic.parameters()},
            {"params": self.model.actor.parameters()},
            {"params": self.model.std}
        ]
        self.optimizer1 = optim.Adam(params1, lr=learning_rate)
        self.optimizer2 = optim.Adam(self.model.student_moe_encoder.parameters(), lr=student_encoder_learning_rate)
        self.transition = RolloutStorageCTS.Transition()

        # CTS parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.load_balance_coef = load_balance_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.teacher_num_envs = max(int(num_envs * teacher_env_ratio), 1)
        self.student_num_envs = num_envs - self.teacher_num_envs
        student_env_ratio = 1 - teacher_env_ratio
        self.teacher_env_idxs = torch.tensor([i for i in range(num_envs) if i % int(1/student_env_ratio) != 0], device=self.device)
        self.student_env_idxs = torch.tensor([i for i in range(num_envs) if i % int(1/student_env_ratio) == 0], device=self.device)
        assert len(self.teacher_env_idxs) == self.teacher_num_envs, f"{len(self.teacher_env_idxs)=} != {self.teacher_num_envs=}"
        assert len(self.student_env_idxs) == self.student_num_envs, f"{len(self.student_env_idxs)=} != {self.student_num_envs=}"

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_latent_loss = 0
        mean_load_balance_loss = 0
        assert not self.model.is_recurrent
        data = list(self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs))
        teacher_samples = self.teacher_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        student_samples = self.student_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        for sample in data:
            (
                obs_batch, privileged_obs_batch, actions_batch, history_batch,
                target_values_batch, advantages_batch, returns_batch,
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
                hid_states_batch, masks_batch
            ) = sample
            def get_results(start, end, is_teacher):
                self.model.act(obs_batch[start:end], privileged_obs_batch[start:end], history_batch[start:end], is_teacher)
                actions_log_prob = self.model.get_actions_log_prob(actions_batch[start:end])
                value = self.model.evaluate(privileged_obs_batch[start:end], history_batch[start:end], is_teacher)
                mu = self.model.action_mean
                sigma = self.model.action_std
                entropy = self.model.entropy
                return actions_log_prob, value, mu, sigma, entropy
            teacher_results = get_results(0, teacher_samples, True)
            student_results = get_results(teacher_samples, teacher_samples + student_samples, False)
            results = []
            for x1, x2 in zip(teacher_results, student_results):
                results.append(torch.cat([x1, x2], dim=0))
            actions_log_prob_batch = results[0]
            value_batch = results[1]
            mu_batch = results[2]
            sigma_batch = results[3]
            entropy_batch = results[4]

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(
                            sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) +
                                torch.square(old_mu_batch - mu_batch)
                            ) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
                    for param_group in self.optimizer1.param_groups:
                        param_group['lr'] = self.learning_rate


            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                            1.0 + self.clip_param)
            surrogate_losses = torch.max(surrogate, surrogate_clipped)
            teacher_surrogate_loss = surrogate_losses[:teacher_samples].mean()
            student_surrogate_loss = surrogate_losses[teacher_samples:].mean()
            surrogate_loss = teacher_surrogate_loss + student_surrogate_loss
            # surrogate_loss = teacher_surrogate_loss

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            # teacher_value_loss = value_losses[:teacher_samples].mean()
            # student_value_loss = value_losses[teacher_samples:].mean()
            # value_loss = teacher_value_loss  # + student_value_loss

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer1.zero_grad()
            loss.backward()
            params_to_clip = itertools.chain.from_iterable(g['params'] for g in self.optimizer1.param_groups)
            nn.utils.clip_grad_norm_(params_to_clip, self.max_grad_norm)
            self.optimizer1.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()
        
        for sample in data:
            (
                obs_batch, privileged_obs_batch, actions_batch, history_batch,
                target_values_batch, advantages_batch, returns_batch,
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
                hid_states_batch, masks_batch
            ) = sample
            # Student encoder update
            student_latent, gating_weights = self.model.student_moe_encoder(history_batch[teacher_samples:])
            with torch.no_grad():
                teacher_latent = self.model.teacher_encoder(privileged_obs_batch[teacher_samples:])
            latent_loss = (teacher_latent - student_latent).pow(2).mean()

            # Load balance loss
            mean_usage = torch.mean(gating_weights, dim=0)
            target_usage = torch.full_like(mean_usage, 1.0 / gating_weights.shape[1])
            load_balance_loss = torch.mean((mean_usage - target_usage).pow(2))
            # load_balance_loss = torch.sum(mean_usage.pow(2)) * gating_weights.shape[1]  # Switch Transformer style

            student_loss = latent_loss + self.load_balance_coef * load_balance_loss

            self.optimizer2.zero_grad()
            student_loss.backward()
            nn.utils.clip_grad_norm_(self.model.student_moe_encoder.parameters(), self.max_grad_norm)
            self.optimizer2.step()

            mean_latent_loss += latent_loss.item()
            mean_load_balance_loss += load_balance_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_latent_loss /= num_updates
        mean_load_balance_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_latent_loss, mean_load_balance_loss
