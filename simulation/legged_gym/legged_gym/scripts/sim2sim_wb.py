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
from legged_gym.envs.gr1.gr1_walk_phase_config import GR1WalkPhaseCfg
from scipy.spatial.transform import Rotation as R
from legged_gym.envs import LEGGED_GYM_ROOT_DIR
import torch


class cmd:
    vx = 0.0
    vy = 0.
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

def record_obs_to_csv(obs):
    with open('../utils/obs.csv',mode='a',newline='')as file:
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

    action_startup = np.zeros(cfg.env.num_actions, dtype=np.float32)
    # default_joint_pos = np.zeros(cfg.env.num_actions, dtype=np.float32)
    # for index, value in enumerate(cfg.init_state.default_joint_angles.values()):
    #     action_startup[index] = value * (1 // cfg.control.action_scale)
    #     default_joint_pos[index] = value
    print(data.qpos)
    default_dof_pos = np.array([
        0.0, 0.0, -0.4, 0.8, -0.4, 0.0,  # left leg (6)
        0.0, 0.0, -0.4, 0.8, -0.4, 0.0,  # right leg (6)
        0.0, -0.0, 0.0,  # waist (3)
        0.0, 0.2, 0.0, -0.3,
        0.0, -0.2, 0.0, -0.3,
    ])
    data.qpos[7:] = default_dof_pos[:]

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    action[:] = action_startup[:]
    device = "cuda"

    def extract_data():
        device = "cuda"
        dof_pos = data.qpos.astype(np.float32)[-21:]
        dof_vel = data.qvel.astype(np.float32)[-21:]
        quat = data.sensor('orientation').data.astype(np.float32)
        ang_vel = data.sensor('angular-velocity').data.astype(np.float32)
        print("2")
        dof_vel = torch.from_numpy(dof_vel).float().unsqueeze(0).to(device)
        print("3")
        return (dof_pos, dof_vel, quat, ang_vel)

    count_lowlevel = 0
    count_max_merge = 50

    obs = np.zeros([1, cfg.env.num_observations], dtype=np.float32)  # 47

    init_csv('../utils/obs.csv')
    commands = np.zeros(3)
    history_len = 10
    n_proprio = 71
    n_priv_latent = 50
    proprio_history_buf = deque(maxlen=history_len)
    priv_latent = np.zeros(n_priv_latent, dtype=np.float32)
    control_indices = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])  # all dofs
    policy_jit = torch.jit.load(policy, map_location=device)
    dof_pos, dof_vel, quat, ang_vel = extract_data()

    for _ in range(history_len):
        proprio_history_buf.append(np.zeros(n_proprio))
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        dof_pos, dof_vel, quat, ang_vel = extract_data()

        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]
        control_indices = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18, 19, 20, 21, 22])
        obs_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22])
        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            phase = (count_lowlevel // 20) * 0.001 / 0.8
            sin_pos = [np.sin(2 * np.pi * phase)]
            cos_pos = [np.cos(2 * np.pi * phase)]
            obs_prop = np.concatenate([
                sin_pos, cos_pos,
                commands,
                omega * 0.25,
                omega[:2],
                (dof_pos - default_dof_pos[obs_indices]) * 1.0,
                dof_vel[obs_indices] * 0.05,
                action,
            ])
            print("1")
            obs_hist = np.array(proprio_history_buf).flatten()
            proprio_history_buf.append(obs_prop)
            obs_buf = np.concatenate([obs_prop, priv_latent, obs_hist])
            obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(device)
            with torch.no_grad():
                raw_action = policy_jit(obs_tensor).cpu().numpy().squeeze()
            action = raw_action.copy()
            raw_action = np.clip(raw_action, -10., 10.)
            scaled_actions = raw_action * 0.5
            step_actions = np.zeros(21)
            step_actions[control_indices] = scaled_actions

            pd_target = step_actions + default_dof_pos

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

        tau = pd_control(pd_target, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques

        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques

        data.ctrl = tau

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

    class Sim2simCfg(GR1WalkPhaseCfg):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/gr1t1/urdf/GR1T1.xml"
            else:
                # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L.xml'
                mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/gr1t1/urdf/GR1T1.xml"
                print("mujoco_model_path",mujoco_model_path)
            sim_duration = 60.0
            dt = 0.001
            decimation = 20
            cycle_time = 0.8

        class robot_config:
            kps = np.array([251.625, 362.52, 200, 200, 10.98, 0.0,
                251.625, 362.52, 200, 200, 10.98, 0.0,
                362.52, 362.52*2, 362.52*2,
                40, 40, 40, 40,
                40, 40, 40, 40], dtype=np.double)
            kds = np.array([14.72, 10.08, 11, 11, 0.60, 0.1,
                14.72, 10.08, 11, 11, 0.60, 0.1,
                10.08, 10.08, 10.08,
                2.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0, 2.0,], dtype=np.double)
            tau_limit = np.array([48, 60, 160, 160, 16, 8,
                48, 60, 160, 160, 16, 8,
                82.5, 82.5, 82.5,
                18, 18, 18, 18,
                18, 18, 18, 18,], dtype=np.double)

    # policy = torch.jit.load(args.load_model)
    policy = args.load_model
    run_mujoco(policy, Sim2simCfg())
