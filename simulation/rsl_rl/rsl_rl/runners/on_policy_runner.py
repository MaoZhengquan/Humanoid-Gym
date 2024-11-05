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

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

import numpy as np
from rsl_rl.algorithms import PPORMA, PPO
from rsl_rl.modules import *
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings
from rsl_rl.utils.running_mean_std import RunningMeanStd

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 init_wandb=True,
                 device='cpu', **kwargs):

        self.cfg=train_cfg["runner"] # 训练配置文件
        self.alg_cfg = train_cfg["algorithm"] # 算法类型PPORMA
        self.policy_cfg = train_cfg["policy"] # 策略类型ACRMA
        self.device = device
        self.env = env
        self.normalize_obs = env.cfg.env.normalize_obs # 归一化观察值

        policy_class = eval(self.cfg["policy_class_name"]) # 动态根据配置中的policy_class_name获取策略类，赋值给policy_class
        # 创建具体的策略网络AC模型
        actor_critic = policy_class(num_prop=self.env.cfg.env.n_proprio,
                                    num_critic_obs=self.env.num_obs,
                                    num_priv_latent=self.env.cfg.env.n_priv_latent,
                                    num_hist=self.env.cfg.env.history_len,
                                    num_actions=self.env.num_actions,
                                    **self.policy_cfg).to(self.device)

        # 存储观察值均值和标准差的归一化器
        if self.normalize_obs:
            self.normalizer = RunningMeanStd(insize=self.env.num_obs).to(self.device)
        else:
            self.normalizer = None

        # 动态类获取算法类
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        # 创建RL算法对象
        self.alg = alg_class(self.env, 
                                  actor_critic,
                                  device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"] # 每个环境的步骤数
        self.save_interval = self.cfg["save_interval"] # 保存间隔
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]  # DAgger更新频率

        # 初始化存储转换的数据结构
        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env, 
            [self.env.num_obs], 
            [self.env.num_privileged_obs], 
            [self.env.num_actions],
        )

        self.learn = self.learn_RL
            
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0 # 总时间步数计数器
        self.tot_time = 0 # 总执行时间计数器
        self.current_learning_iteration = 0 # 当前学习迭代次数计数器
        

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0. # 价值损失
        mean_surrogate_loss = 0. # 代理损失
        mean_disc_loss = 0. # 辨别损失
        mean_disc_acc = 0. # 辨别器准确率
        mean_hist_latent_loss = 0. # 历史隐变量损失
        mean_priv_reg_loss = 0. # 正则化损失
        priv_reg_coef = 0. # 正则化系数
        entropy_coef = 0. # 熵系数
        grad_penalty_coef = 0. # 梯度惩罚系数

        # 在随机的回合长度上初始化（用于增强探索）
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        # 收集观测数据
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        # 观测数据移动到CPU
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        if self.normalize_obs:
            obs = self.normalizer(obs)
            # 归一化器评估模式
            self.normalizer.eval()
            critic_obs = self.normalizer(critic_obs)
            self.normalizer.train()
        infos = {} # 存储与环境交互时获取的额外信息
        # 策略网络设置为训练模式
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = [] # 存储每一个回合的信息
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        # 当前奖励累加器
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # 当前回合长度累加器
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        task_rew_buf = deque(maxlen=100)
        cur_task_rew_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 计算总学习迭代次数
        tot_iter = self.current_learning_iteration + num_learning_iterations
        # 保存当前迭代次数
        self.start_learning_iteration = copy(self.current_learning_iteration)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # 检测是否需要DAgger更新
            hist_encoding = it % self.dagger_update_freq == 0
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env): # 在每个环境中进行预定步数的交互
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding) # 选择动作
                    # 将动作传递给环境 获取下一个观察、奖励等信息
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # 检查是否归一化
                    if self.normalize_obs:
                        obs = self.normalizer(obs)
                        
                        self.normalizer.eval()
                        critic_obs = self.normalizer(critic_obs)
                        self.normalizer.train()

                    # 处理环境步骤 计算总奖励
                    total_rew = self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        # import ipdb; ipdb.set_trace()
                        cur_reward_sum += total_rew # 累加当前奖励
                        cur_reward_explr_sum += 0
                        cur_reward_entropy_sum += 0
                        cur_episode_length += 1 # 增加回合长度

                        # 结束回合标识
                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        # 将奖励和回合长度添加到对应的缓冲区
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        cur_reward_sum[new_ids] = 0
                        cur_reward_explr_sum[new_ids] = 0
                        cur_reward_entropy_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                stop = time.time()
                collection_time = stop - start # 收集时间

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs) # 计算回归值

            # 正则化系数
            regularization_scale = self.env.cfg.rewards.regularization_scale if hasattr(self.env.cfg.rewards, "regularization_scale") else 1
            # 平均回合长度
            average_episode_length = torch.mean(self.env.episode_length.float()).item() if hasattr(self.env, "episode_length") else 0
            # 返回损失值
            mean_value_loss, mean_surrogate_loss, mean_priv_reg_loss, priv_reg_coef, mean_grad_penalty_loss, grad_penalty_coef = self.alg.update()
            if hist_encoding and not self.cfg["algorithm_class_name"] == "PPO":
                print("Updating dagger...")
                mean_hist_latent_loss = self.alg.update_dagger()
            
            stop = time.time()
            learn_time = stop - start # 学习时间
            if self.log_dir is not None:
                self.log(locals())
            # 不同学习迭代阶段以不同频率保存模型
            if it < 2500:
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            elif it < 5000:
                if it % (2*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                if it % (5*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear() # 清空回合信息列表 准备新的回合记录
        
        # self.current_learning_iteration += num_learning_iterations
        # 回合结束后 保存模型
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        # 更新总时间步数
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        # 累加迭代时间 收集+学习
        self.tot_time += locs['collection_time'] + locs['learn_time']
        # 计算当前迭代时间总和
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]: # 遍历第一个回合信息中的每个键
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']: # 遍历所有的回合信息
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor): # 检测是否为张量
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0: # 检测当前信息是否为零维
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor) # 计算平均值
                # wandb_dict['Episode_rew/' + key] = value
                if "metric" in key:
                    wandb_dict['Episode_rew_metrics/' + key] = value
                else:
                    if "tracking" in key:
                        wandb_dict['Episode_rew_tracking/' + key] = value
                    elif "curriculum" in key:
                        wandb_dict['Episode_curriculum/' + key] = value
                    else:
                        wandb_dict['Episode_rew_regularization/' + key] = value
                    # 格式化并将平均值以字符串形式追加到ep_string中
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n""" # dont print metrics
        # 获取策略的标准差 反映探索程度
        mean_std = self.alg.actor_critic.std.mean()
        # 计算每秒的步骤数 反映算法的运行效率
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        # 将统计信息添加到WandB字典中
        wandb_dict['Loss/value_func'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict['Loss/grad_penalty_loss'] = locs['mean_grad_penalty_loss']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        # wandb_dict['Loss/discriminator'] = locs['mean_disc_loss']
        # wandb_dict['Loss/discriminator_accuracy'] = locs['mean_disc_acc']

        wandb_dict['Adaptation/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Adaptation/priv_reg_loss'] = locs['mean_priv_reg_loss']
        wandb_dict['Adaptation/priv_ref_lambda'] = locs['priv_reg_coef']

        wandb_dict['Scale/regularization_scale'] = locs["regularization_scale"]
        wandb_dict['Scale/grad_penalty_coef'] = locs["grad_penalty_coef"]

        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            # wandb_dict['Train/mean_reward_explr'] = statistics.mean(locs['rew_explr_buffer'])
            # wandb_dict['Train/mean_reward_task'] = statistics.mean(locs['task_rew_buf'])
            # wandb_dict['Train/mean_reward_entropy'] = statistics.mean(locs['rew_entropy_buffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            # wandb_dict['Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            # wandb_dict['Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        wandb.log(wandb_dict, step=locs['it'])

        # 当前迭代次数/总学习迭代次数 \033粗体文本
        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        # 获取正则化系数 平均回合长度 梯度惩罚系数
        scale_str = f"""{'Regularization_scale:':>{pad}} {locs['regularization_scale']:.4f}\n"""
        average_episode_length = f"""{'Average_episode_length:':>{pad}} {locs['average_episode_length']:.4f}\n"""
        gp_scale_str = f"""{'Grad_penalty_coef:':>{pad}} {locs['grad_penalty_coef']:.4f}\n"""
        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Experiment Name:':>{pad}} {os.path.basename(self.log_dir)}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Grad penalty loss:':>{pad}} {locs['mean_grad_penalty_loss']:.4f}\n"""
                        #   f"""{'Discriminator loss:':>{pad}} {locs['mean_disc_loss']:.4f}\n"""
                        #   f"""{'Discriminator accuracy:':>{pad}} {locs['mean_disc_acc']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                        #   f"""{'Mean reward (task):':>{pad}} {statistics.mean(locs['task_rew_buf']):.2f}\n"""
                        #   f"""{'Mean reward (exploration):':>{pad}} {statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                        #   f"""{'Mean reward (entropy):':>{pad}} {statistics.mean(locs['rew_entropy_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += f"""{'-' * width}\n""" # 分隔线
        log_string += ep_string
        log_string += f"""{'-' * width}\n"""
        log_string += scale_str
        log_string += average_episode_length
        log_string += gp_scale_str
        curr_it = locs['it'] - self.start_learning_iteration
        # 当前总迭代时间与迭代步骤相比 * 总迭代步骤 推车ETA
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def save(self, path, infos=None):
        if self.normalize_obs:
            state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(), # 策略和价值网络参数
            'optimizer_state_dict': self.alg.optimizer.state_dict(), # 优化器参数
            'iter': self.current_learning_iteration, # 当前学习迭代次数
            'normalizer': self.normalizer, # 观测归一化器
            'infos': infos,
            }
        else:
            state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict']) # 加载模型参数
        if self.normalize_obs:
            self.normalizer = loaded_dict['normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict']) # 恢复优化器模型
        # self.current_learning_iteration = loaded_dict['iter']
        self.current_learning_iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
        self.env.global_counter = self.current_learning_iteration * 24 # 更新环境步骤的计数器
        self.env.total_env_steps_counter = self.current_learning_iteration * 24
        print("*" * 80)
        return loaded_dict['infos']

    # 获取当前策略网络推理的函数
    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    # 获取当前的策略和价值网络
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic

    # 获取当前的观测归一化器
    def get_normalizer(self, device=None):
        self.normalizer.eval()
        if device is not None:
            self.normalizer.to(device)
        return self.normalizer
    