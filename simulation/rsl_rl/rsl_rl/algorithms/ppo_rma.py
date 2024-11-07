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
import numpy as np

from rsl_rl.modules import ActorCriticRMA
from rsl_rl.storage.rollout_storage import RolloutStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.utils import unpad_trajectories
import time

class RMS(object):
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape, device=device)
        self.S = torch.ones(shape, device=device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S

class PPORMA:
    def __init__(self,
                 env, 
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 dagger_update_freq=20,
                 priv_reg_coef_schedual = [0, 0, 0],
                 grad_penalty_coef_schedule = [0, 0, 0],
                 num_hist=10,
                 **kwargs
                 ):

        self.env = env
        self.device = device
        self.num_hist = num_hist

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic # AC网络
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate) # 优化器
        self.transition = RolloutStorage.Transition() # 状态转移

        # PPO parameters
        self.clip_param = clip_param # clip参数
        self.num_learning_epochs = num_learning_epochs # 每次更新的学习周期
        self.num_mini_batches = num_mini_batches # 小批量数量 稳定训练
        self.value_loss_coef = value_loss_coef # 值函数损失的系数
        self.entropy_coef = entropy_coef # 熵损失的系数
        self.gamma = gamma # 折扣因子
        self.lam = lam # GAE的优势估计参数
        self.max_grad_norm = max_grad_norm # 梯度裁减的最大范数 防止梯度爆炸
        self.use_clipped_value_loss = use_clipped_value_loss # 是否使用裁减值函数损失

        # Adaptation
        self.hist_encoder_optimizer = optim.Adam(self.actor_critic.actor.history_encoder.parameters(), lr=learning_rate) # 历史编码器的优化器
        self.priv_reg_coef_schedual = priv_reg_coef_schedual # 正则化系数的调节计划
        self.gradient_penalty_coef_schedule = grad_penalty_coef_schedule # [0.002 0.002 700 1000] # 梯度惩罚系数的计划
        self.counter = 0 # 计数器
    
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape,  critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, info, hist_encoding=False):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach() # action值是过actor网络

        self.transition.values = self.actor_critic.evaluate(critic_obs).detach() # value值是过critic网络
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach() # 计算动作的对数概率
        self.transition.action_mean = self.actor_critic.action_mean.detach() # 动作分布的均值
        self.transition.action_sigma = self.actor_critic.action_std.detach() # 动作分布的标准差
        self.transition.observations = obs # 存储本体观测数据
        self.transition.critic_observations = critic_obs # 存储特权观测数据

        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones # 记录每个环境是否终止
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition) # 将当前的状态转移记录添加到存储中
        self.transition.clear() # 清空transition对象
        self.actor_critic.reset(dones)
        
        return rewards
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _calc_grad_penalty(self, obs_batch, actions_log_prob_batch):
        # 计算log_pi(a|s)对obs_batch的梯度
        grad_log_prob = torch.autograd.grad(actions_log_prob_batch.sum(), obs_batch, create_graph=True)[0]
        # 梯度平方运算 求和 再求均值
        gradient_penalty_loss = torch.sum(torch.square(grad_log_prob), dim=-1).mean()
        return gradient_penalty_loss
    
    def update(self):
        mean_value_loss = 0 # 价值损失
        mean_surrogate_loss = 0 # 代理损失
        mean_priv_reg_loss = 0 # 正则化损失
        mean_grad_penalty_loss = 0 # 梯度惩罚损失

        # 数据生成器
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for sample in generator:
                obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
                # 克隆观察数据 设置梯度计算
                obs_est_batch = obs_batch.clone()
                
                obs_est_batch.requires_grad_()
                # 使用策略网络对观测数据进行动作选择
                self.actor_critic.act(obs_est_batch, masks=masks_batch, hidden_states=hid_states_batch[0]) # match distribution dimension

                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                # 使用critic网络评估当前价值
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                # 当前策略的均值和标准差
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                # 计算当前的熵
                entropy_batch = self.actor_critic.entropy
                
                # Calculate the gradient penalty loss
                gradient_penalty_loss = self._calc_grad_penalty(obs_est_batch, actions_log_prob_batch)

                # 从 700开始动态调整到1000以后固定 在这里数值保持不变
                gradient_stage = min(max((self.counter - self.gradient_penalty_coef_schedule[2]), 0) / self.gradient_penalty_coef_schedule[3], 1)
                gradient_penalty_coef = gradient_stage * (self.gradient_penalty_coef_schedule[1] - self.gradient_penalty_coef_schedule[0]) + self.gradient_penalty_coef_schedule[0]
                
                # Adaptation module update
                priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch) # 推断隐变量
                with torch.inference_mode():
                    hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch) # 推理历史编码
                # 计算正则化损失
                priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
                # 调整正则化系数
                priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
                priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]
                
                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        # 计算KL散度
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)
                        # 更新lr
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                # 计算策略比例
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                # 计算代理损失
                surrogate = -torch.squeeze(advantages_batch) * ratio
                # 计算裁减后的代理损失
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                # 计算最终的价值损失
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                # 根据是否裁减来计算价值损失
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # 整合不同的损失项 形成最终的损失目标
                loss = surrogate_loss + \
                       self.value_loss_coef * value_loss - \
                       self.entropy_coef * entropy_batch.mean() + \
                       priv_reg_coef * priv_reg_loss + \
                       gradient_penalty_coef * gradient_penalty_loss
                # loss = self.teacher_alpha * imitation_loss + (1 - self.teacher_alpha) * loss

                # Gradient step
                self.optimizer.zero_grad() # 清除模型参数的梯度
                loss.backward() # 反向传播计算梯度
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm) # 防止梯度爆炸
                self.optimizer.step() # 更新参数

                # 累加当前损失值 计算整体的平均损失
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_priv_reg_loss += priv_reg_loss.item()
                mean_grad_penalty_loss += gradient_penalty_loss.item()
                

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        mean_grad_penalty_loss /= num_updates
        
        self.storage.clear()
        self.update_counter()
        return mean_value_loss, mean_surrogate_loss, mean_priv_reg_loss, priv_reg_coef, mean_grad_penalty_loss, gradient_penalty_coef

    # 优化历史编码器 计算hist_latent的loss
    def update_dagger(self):
        mean_hist_latent_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
                with torch.inference_mode():
                    self.actor_critic.act(obs_batch, hist_encoding=True, masks=masks_batch, hidden_states=hid_states_batch[0])

                # Adaptation module update
                with torch.inference_mode():
                    priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
                hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
                self.hist_encoder_optimizer.zero_grad()
                hist_latent_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm)
                self.hist_encoder_optimizer.step()
                
                mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss
    
    def update_counter(self):
        self.counter += 1
