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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import CTS, MoECTS, MCPCTS, ACMoECTS, DualMoECTS, REMCTS
from rsl_rl.modules import ActorCriticCTS, ActorCriticMoECTS, ActorCriticMCPCTS, ActorCriticACMoECTS, ActorCriticDualMoECTS, ActorCriticREMCTS
from rsl_rl.env import VecEnv

import yaml
import numpy as np
from pathlib import Path
from legged_gym.utils.helpers import class_to_dict
from typing import Union
from legged_gym.utils.exporter import export_policy_as_jit

def numpy_representer(dumper, data):
    return dumper.represent_float(float(data))

def numpy_int_representer(dumper, data):
    return dumper.represent_int(int(data))

# Add the numpy representer to yaml
yaml.add_representer(np.float32, numpy_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.float64, numpy_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.int32, numpy_int_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.int64, numpy_int_representer, Dumper=yaml.SafeDumper)


class OnPolicyRunnerCTS:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        history_length = train_cfg["history_length"]
        actor_critic_class = eval(self.cfg["policy_class_name"])
        model: Union[ActorCriticCTS, ActorCriticMoECTS, ActorCriticMCPCTS, ActorCriticACMoECTS, ActorCriticDualMoECTS, ActorCriticREMCTS] = actor_critic_class(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_actions,
            self.env.num_envs,
            history_length,
            **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: Union[CTS, MoECTS, MCPCTS, ACMoECTS, DualMoECTS, REMCTS] = alg_class(model, self.env.num_envs, history_length, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # init history
        self.history = torch.zeros((self.env.num_envs, history_length, self.env.num_obs), device=self.device)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
        if self.log_dir is not None and self.env.cfg.env.test is False:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            all_cfg = {"train_cfg": train_cfg, "env_cfg": class_to_dict(self.env.cfg)}
            yaml.safe_dump(all_cfg, open(os.path.join(self.log_dir, 'config.yaml'), 'w'))
        
        # robogauge client
        try:
            from robogauge.scripts.client import RoboGaugeClient
            self.robogauge_client = RoboGaugeClient("http://127.0.0.1:9973")  # Change PORT to your server port if needed, default is 9973
        except:
            self.robogauge_client = None
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        assert privileged_obs is not None
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
        self.alg.model.train() # switch to train mode (for dropout for example)

        ep_infos = []
        teacher_rewbuffer = deque(maxlen=100)
        teacher_lenbuffer = deque(maxlen=100)
        student_rewbuffer = deque(maxlen=100)
        student_lenbuffer = deque(maxlen=100)

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.start_learning_iteration = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs, self.history.flatten(1))
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    obs, privileged_obs, rewards, dones = obs.to(self.device), privileged_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.history[dones > 0] = 0.0
                    self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if new_ids.shape[0]:
                            ti = self.alg.teacher_env_idxs
                            teacher_ids = new_ids[torch.isin(new_ids, ti)]
                            student_ids = new_ids[~torch.isin(new_ids, ti)]
                            teacher_rewbuffer.extend(cur_reward_sum[teacher_ids].cpu().numpy().tolist())
                            teacher_lenbuffer.extend(cur_episode_length[teacher_ids].cpu().numpy().tolist())
                            student_rewbuffer.extend(cur_reward_sum[student_ids].cpu().numpy().tolist())
                            student_lenbuffer.extend(cur_episode_length[student_ids].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                if self.cfg["algorithm_class_name"] in ["ACMoECTS", "DualMoECTS"]:
                    self.alg.compute_returns(obs, privileged_obs, self.history.flatten(1))
                else:
                    self.alg.compute_returns(privileged_obs, self.history.flatten(1))
            
            if self.cfg["algorithm_class_name"] in ["CTS", "MCPCTS"]:
                mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_latent_loss = self.alg.update()
            elif self.cfg["algorithm_class_name"] in ["MoECTS", "ACMoECTS", "REMCTS"]:
                mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_latent_loss, mean_load_balance_loss = self.alg.update()
            elif self.cfg["algorithm_class_name"] == "DualMoECTS":
                mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_latent_loss, mean_load_balance_loss, mean_actor_load_balance_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration += 1
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), it, False)
            ep_infos.clear()
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)), it, True)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if 'terrain' in key:
                    self.writer.add_scalar('Terrain/' + key, value, locs['it'])
                else:
                    self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        if 'mcp' not in self.cfg["algorithm_class_name"].lower():
            mean_std = self.alg.model.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/entropy', locs['mean_entropy_loss'], locs['it'])
        self.writer.add_scalar('Loss/latent', locs['mean_latent_loss'], locs['it'])
        if 'mean_load_balance_loss' in locs:
            self.writer.add_scalar('Loss/load_balance', locs['mean_load_balance_loss'], locs['it'])
        if 'mean_actor_load_balance_loss' in locs:
            self.writer.add_scalar('Loss/actor_load_balance', locs['mean_actor_load_balance_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        if 'mcp' not in self.cfg["algorithm_class_name"].lower():
            self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['teacher_rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_teacher_reward', statistics.mean(locs['teacher_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_teacher_episode_length', statistics.mean(locs['teacher_lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_teacher_reward/time', statistics.mean(locs['teacher_rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_teacher_episode_length/time', statistics.mean(locs['teacher_lenbuffer']), self.tot_time)
        if len(locs['student_rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_student_reward', statistics.mean(locs['student_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_student_episode_length', statistics.mean(locs['student_lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_student_reward/time', statistics.mean(locs['student_rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_student_episode_length/time', statistics.mean(locs['student_lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033[0m "

        log_string = (f"""{'#' * width}\n"""
                      f"""{str.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                      'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                      f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                      f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                      f"""{'Entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                      f"""{'Latent loss:':>{pad}} {locs['mean_latent_loss']:.4f}\n""")
        if 'mean_load_balance_loss' in locs:
            log_string += f"""{'Load balance loss:':>{pad}} {locs['mean_load_balance_loss']:.4f}\n"""
        if 'mean_actor_load_balance_loss' in locs:
            log_string += f"""{'Actor load balance loss:':>{pad}} {locs['mean_actor_load_balance_loss']:.4f}\n"""
        if 'mcp' not in self.cfg["algorithm_class_name"].lower():
            log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
        if len(locs['teacher_rewbuffer']):
            log_string += (f"""{'Mean teacher reward:':>{pad}} {statistics.mean(locs['teacher_rewbuffer']):.2f}\n"""
                           f"""{'Mean teacher episode length:':>{pad}} {statistics.mean(locs['teacher_lenbuffer']):.2f}\n""")
        if len(locs['student_rewbuffer']):
            log_string += (f"""{'Mean student reward:':>{pad}} {statistics.mean(locs['student_rewbuffer']):.2f}\n"""
                           f"""{'Mean student episode length:':>{pad}} {statistics.mean(locs['student_lenbuffer']):.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (self.current_learning_iteration - self.start_learning_iteration) * (
                               locs['tot_iter'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, it, last_model, infos=None):
        torch.save({
            'model_state_dict': self.alg.model.state_dict(),
            'optimizer1_state_dict': self.alg.optimizer1.state_dict(),
            'optimizer2_state_dict': self.alg.optimizer2.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)
        self.update_robogauge(it, last_model)
    
    def update_robogauge(self, it, last_model):
        if self.robogauge_client is None:
            return

        if it % 500 == 0 or last_model:
            # export jit model
            jit_dir = os.path.join(self.log_dir, 'jit_models')
            jit_path = os.path.join(jit_dir, f'policy_jit_{it}.pt')
            export_policy_as_jit(self.alg.model, jit_dir, filename=f'policy_jit_{it}.pt')
            # upload to robogauge
            task_name = 'go2_moe'  # Both cts, moe-cts actor return a tuple `action, (latent, ...)`
            self.robogauge_client.submit_task(
                model_path=jit_path,
                step=it,
                task_name=task_name,
                experiment_name=self.cfg["experiment_name"]
            )
        check_times = 1
        if last_model:
            check_times = int(1e9)  # keep checking until manually stopped
        while check_times > 0:
            check_times -= 1
            self.robogauge_client.monitor_tasks()
            results_dir = os.path.join(self.log_dir, 'robogauge_results')
            os.makedirs(results_dir, exist_ok=True)
            result_received = False
            for task_id, resp in self.robogauge_client.response_data.items():
                scores = resp['results']['scores']
                step = resp['step']
                if step == it:
                    result_received = True
                for key, val in scores.items():
                    self.writer.add_scalar(f'RoboGauge/{key}', val, step)
                results_path = os.path.join(results_dir, f'results_{step}.yaml')
                with open(results_path, 'w', encoding='utf-8') as f:
                    yaml.dump(resp['results'], f, allow_unicode=True, sort_keys=False)
            
            if last_model and result_received:
                print(f"RoboGauge result for step {it} received. Exiting wait loop.")
                break

            if check_times > 0:
                print("Sleeping for 1 minute before checking RoboGauge results again...")
                time.sleep(60)  # wait for 1 minute before checking again

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.model.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer1.load_state_dict(loaded_dict['optimizer1_state_dict'])
            self.alg.optimizer2.load_state_dict(loaded_dict['optimizer2_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.model.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.model.to(device)
        return self.alg.model.act_inference
