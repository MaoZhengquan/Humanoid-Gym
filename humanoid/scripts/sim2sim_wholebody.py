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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import os.path
import csv
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque

from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
# from humanoid.envs import XBotLCfg
from humanoid.envs import GRCfg
import onnxruntime as ort
import torch


class cmd:
    vx = 0.
    vy = 0.0
    dyaw = 0.0


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def init_csv(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path}文件已删除")

def record_obs_to_csv(obs,path):
    with open(path,mode='a',newline='')as file:
        writer = csv.writer(file)
        for row in obs:
            formatted_row = [f"{x:.3f}" for x in row]
            writer.writerow(formatted_row)

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt

    data = mujoco.MjData(model)

    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    action_startup = np.zeros(cfg.env.num_dofs, dtype=np.float32)
    default_joint_pos = np.zeros(cfg.env.num_dofs, dtype=np.float32)
    for index, value in enumerate(cfg.init_state.default_joint_angles.values()):
        action_startup[index] = value * (1 // cfg.control.action_scale)
        default_joint_pos[index] = value
    print(data.qpos)
    data.qpos[7:] = default_joint_pos[:]

    target_q = np.zeros((cfg.env.num_actions))
    action = np.zeros((cfg.env.num_dofs))
    action[:] = action_startup[:]
    proprio_history_buf = deque(maxlen=cfg.env.history_len)
    for _ in range(cfg.env.history_len):
        proprio_history_buf.append(np.zeros(cfg.env.n_proprio))
    priv_latent = np.zeros(cfg.env.n_priv_latent, dtype=np.float32)

    count_lowlevel = 0
    count_max_merge = 50
    last_action = np.zeros(cfg.env.num_actions, dtype=np.float32)
    obs = np.zeros([1, cfg.env.num_observations], dtype=np.float32)  # 47

    init_csv('../utils/obs.csv')
    init_csv('../utils/tau.csv')

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        phase = count_lowlevel * 0.002 / cfg.rewards.cycle_time
        # phase = (_ // 10 ) * 0.02 / 0.8

        mask_right = (math.floor(phase) + 1) % 2
        mask_left = math.floor(phase) % 2

        cos_pos = (1 - math.cos(2 * math.pi * phase)) / 2  # 得到一条从0开始增加，频率为step_freq，振幅0～1的曲线，接地比较平滑
        # print("cfg.sim_config.dt",cfg.sim_config.dt)
        right_leg_phase = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.64)
        left_leg_phase =  math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.64)

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_dofs:]
        dq = dq[-cfg.env.num_dofs:]

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros(cfg.env.n_proprio, dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            obs[0] = right_leg_phase
            obs[1] = left_leg_phase
            obs[2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            obs[5:8] = omega * cfg.normalization.obs_scales.ang_vel
            obs[8:10] = eu_ang[:2]
            obs[10:22] = (q - default_joint_pos) * cfg.normalization.obs_scales.dof_pos
            obs[22:34] = dq * cfg.normalization.obs_scales.dof_vel
            print(last_action.shape)
            obs[34:44] = last_action
            record_obs_to_csv(obs.reshape(1,44),'../utils/obs.csv')
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            obs_hist = np.array(proprio_history_buf).flatten()
            proprio_history_buf.append(obs)
            obs_buf = np.concatenate([obs,priv_latent,obs_hist],dtype=np.float32)
            obs_buf = obs_buf.reshape(1,516)
            input_name = sess.get_inputs()[0].name
            raw_action = sess.run(None,{input_name: obs_buf})
            # obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0)
            # with torch.no_grad():
            #     raw_action = policy(obs_tensor).cpu().numpy().squeeze()
            last_action = np.array(raw_action.copy())
            scaled_actions = np.array(raw_action) * 0.5
            step_actions = np.zeros(12)
            step_actions[cfg.env.control_indices] = scaled_actions
            if count_lowlevel < count_max_merge:
                step_actions[:] = (action_startup[:] / count_max_merge * (count_max_merge - count_lowlevel)
                             + action[:] / count_max_merge * count_lowlevel) * 0.5
            # action_scaled = action*cfg.control.action_scale
            # action = np.clip(action ,cfg.normalization.clip_actions_min,cfg.normalization.clip_actions_max)
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            target_q = step_actions + default_joint_pos


            # print("action_scaled",action * cfg.control.action_scale)
        target_dq = np.zeros((cfg.env.num_dofs), dtype=np.double)
        # Generate PD control

        # print("target_q",target_q /3.14 *180.0)
        # print("q", q/3.14 *180.0)
        # print("target_q - q",target_q- q)
        # print("target_dq - q",- dq)

        tau = pd_control(target_q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques

        # print("target_q",target_q /3.14 *180.0)
            # tau = np.round(tau,4)
            # print("right_leg_phase",right_leg_phase)
            # print("left_leg_phase", left_leg_phase)
            # print("joint_offset",obs[0,5:17])
            # print("joint_vel",obs[0, 17:29])
            # print("omega",obs[0, 29:32])
            # print('quat', quat)
            # print("eu_ang", obs[0, 32:35])
            # print("action", obs[0, 35:47])
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        record_obs_to_csv(tau.reshape(1, 12), '../utils/tau.csv')

            # print("action_clip",action)
            # print("tau",tau)
        data.ctrl = tau
        # if count_lowlevel > 200:
        #     print('tau', tau)
        #     data.ctrl = tau
        # else:
        #     target_q = default_angle.copy()
        #     tau = pd_control(target_q, q, cfg.robot_config.kps,
        #                      target_dq, dq, cfg.robot_config.kds)  # Calc torques
        #     tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        #     data.ctrl = tau
        # print('count',count_lowlevel)
        # print("joint_vel", obs[0, 19:21])
        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(GRCfg):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/gr1t1/mjcf/gr1t1-terrain.xml'
            else:
                # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L.xml'
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/gr1t1/mjcf/GR1T1_6DoF.xml'
                print("mujoco_model_path",mujoco_model_path)
            sim_duration = 60.0
            dt = 0.002
            decimation = 10

        class robot_config:
            kps = np.array([200, 200, 350, 350, 10.98,0, 200, 200, 350, 350, 10.98,0], dtype=np.double)
            kds = np.array([20, 20, 20, 20, 0.6,0.1,20, 20, 20, 20, 0.6,0.1], dtype=np.double)
            tau_limit = np.array([48,60,160,160,16,8,48,60,160,160,16,8], dtype=np.double)

    policy = torch.jit.load(args.load_model)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 20
    sess = ort.InferenceSession('/home/mao/Github_Project/humanoid-gym/humanoid/utils/policy.onnx',
                                session_options)
    run_mujoco(policy, Sim2simCfg())
