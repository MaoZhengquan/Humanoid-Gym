import os
import queue
from pathlib import Path

from prompt_toolkit.key_binding.bindings.named_commands import self_insert
from pynput import keyboard
import time
import math
import numpy as np
import typer
import csv
from collections import deque
from fourier_grx_client import ControlGroup, RobotClient
import pandas as pd
from ischedule import run_loop, schedule
from scipy.spatial.transform import Rotation as R
from triton.language import dtype

from utils import *
import onnxruntime as ort
import threading
from Joy.joystick import Joystick

joint_target_queue = queue.Queue()


def init_csv(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path}文件已删除")


class Sim2realCfg:
    num_dofs = 12
    num_actions = 10
    n_proprio = 2 + 3 + 3 + 2 + 2 * (num_dofs) + num_actions
    n_priv_latent = 4 + 1 + 2 * (num_dofs) + 3
    history_len = 10
    control_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]

    sim_duration = 60
    dt = 0.01
    step_freq = 2
    cycle_time = 0.8
    decimation = 10
    action_scale = 0.5

    class obs_scale:
        lin_vel = 1.
        ang_vel = 0.25
        dof_pos = 1.
        dof_vel = 0.05
        quat = 0.5


class DemoNohlaRLWalk:
    """
    Reinforcement Learning Walker
    """

    def __init__(
            self,
            step_freq: int = 500,
            act: bool = True,
            cfg=Sim2realCfg()
    ):
        """
        Initialize the RL Walker

        Input:
        - comm_freq: communication frequency, in Hz
        - step_freq: step frequency, in Hz
        """

        # setup RobotClient
        self.client = RobotClient(namespace="gr/1", server_ip="localhost")
        self.act = act
        time.sleep(1.0)
        self.cfg = cfg
        self.client.enable()
        self.set_gains()
        self.start_time = time.time()
        self.last_target_pos = np.zeros(32)
        self.count_lowlevel = 0
        self.last_action = np.zeros(self.cfg.num_actions)
        self.action = np.zeros(self.cfg.num_actions)
        self.commands = np.zeros(3, dtype=np.float32)
        self.joystick = Joystick()
        # 关节的默认角度
        self.joint_default_position = np.array([
            0.0, 0.0, -0.4, 0.8, -0.4, -0.0,  # left leg (6)
            -0.0, 0.0, -0.4, 0.8, -0.4, 0.0,  # right leg (6)
            0.0, -0.3, 0.0,  # waist (3)
            0.0, 0.0, 0.0,  # waist (3)
            0.0, 0.2, 0.0, -0.3, 0.0, 0.0, 0.0,  # left arm (4)
            0.0, -0.2, 0.0, -0.3, 0.0, 0.0, 0.0,  # right arm (4)
        ])

        self.proprio_history_buf = deque(maxlen=cfg.history_len)
        for _ in range(self.cfg.history_len):
            self.proprio_history_buf.append(np.zeros(self.cfg.n_proprio))
        self.priv_latent = np.zeros(self.cfg.n_priv_latent, dtype=np.float32)

        self.last_action = np.zeros(self.cfg.num_actions)
        self.action = np.zeros(self.cfg.num_actions)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 20
        self.sess = ort.InferenceSession(
            '/home/gr124ja0052/Fourier/fourier-grx-client/examples/data/1122/gr1_policy.onnx',
            session_options)
        # warm up
        time.sleep(5)

    def GetRemoteStates(self):
        data = self.joystick.read()
        if data is None:
            raise ValueError("Joystick not connected!")
        left_joy, right_joy = data["joystick"]
        btns = data["buttons"]
        # A, B, X, Y
        return left_joy, right_joy, btns

    def set_gains(self):
        """
        Set gains for the robot
        """
        pos_kp = [
            0.997, 1.023, 1.061, 1.061, 0.508, 0.508,
            0.997, 1.023, 1.061, 1.061, 0.508, 0.508,
            1.023, 1.023 * 5, 1.023 * 5,
            0.556, 0.556, 0.556,
            0.556, 0.556, 0.556, 0.556, 0, 0, 0,
            0.556, 0.556, 0.556, 0.556, 0, 0, 0,
        ]

        vel_kp = [
            0.044, 0.03, 0.263, 0.263, 0.004, 0.004,
            0.044, 0.03, 0.263, 0.263, 0.004, 0.004,
            0.03, 0.03, 0.03,
            0.03, 0.03, 0.03,
            0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
            0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
        ]
        # fmt: off
        vel_ki = [
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ]
        pd_kp = [
            251.625, 330.52, 200, 200, 10.98, 0,  # left leg
            251.625, 330.52, 200, 200, 10.98, 0,  # right leg
            251.625, 251.625 * 5, 251.625 * 5,  # waist
            112.06, 112.06, 112.06,  # head
            92.85, 92.85, 112.06, 112.06, 112.06, 10, 10,  # left arm
            92.85, 92.85, 112.06, 112.06, 112.06, 10, 10,  # right arm
        ]
        pd_kd = [
            14.72, 10.08, 11, 11, 0.6, 0.1,  # left leg
            14.72, 10.08, 11, 11, 0.6, 0.1,  # right leg
            14.72, 14.72, 14.72,  # waist
            3.1, 3.1, 3.1,  # head
            2.575, 2.575, 3.1, 3.1, 3.1, 1.0, 1.0,  # left arm
            2.575, 2.575, 3.1, 3.1, 3.1, 1.0, 1.0,  # rig
            # ht arm
        ]
        # pd_kp = [
        #     250., 250.0, 350, 350, 10.98, 0,  # left leg
        #     250., 250.0, 350, 350, 10.98, 0,  # right leg
        #     251.625, 251.625 * 5, 251.625 * 5,  # waist
        #     112.06, 112.06, 112.06,  # head
        #     92.85, 92.85, 112.06, 112.06, 112.06, 10, 10,  # left arm
        #     92.85, 92.85, 112.06, 112.06, 112.06, 10, 10,  # right arm
        # ]
        # pd_kd = [
        #     25, 25, 30, 30, 0.6, 0.1,  # left leg
        #     25, 25, 30, 30, 0.6, 0.1,  # right leg
        #     14.72, 14.72, 14.72,  # waist
        #     3.1, 3.1, 3.1,  # head
        #     2.575, 2.575, 3.1, 3.1, 3.1, 1.0, 1.0,  # left arm
        #     2.575, 2.575, 3.1, 3.1, 3.1, 1.0, 1.0,  # rig
        #     # ht arm
        # ]
        # fmt: on
        print(self.client.get_gains())
        self.client.set_gains(position_control_kp=pos_kp, velocity_control_kp=vel_kp, velocity_control_ki=vel_ki,
                              pd_control_kp=pd_kp, pd_control_kd=pd_kd)

    def compute_pd_tau(self, target_q, q, dq):
        pd_kp = np.array([
            251.625, 362.52, 300, 300, 10.98, 0,  # left leg
            251.625, 362.52, 300, 300, 10.98, 0,  # right leg
        ])
        pd_kd = np.array([
            14.72, 10.08, 11, 11, 0.6, 0.1,  # left leg
            14.72, 10.08, 11, 11, 0.6, 0.1,  # right leg
        ])
        tau_limit = np.array([48, 60, 160, 160, 16, 8, 48, 60, 160, 160, 16, 8])
        tau = (target_q - q) * pd_kp - pd_kd * dq
        tau_clip = np.clip(tau, -tau_limit, tau_limit)
        return tau_clip

    def process_remote(self):
        left_joy, right_joy, btns = self.GetRemoteStates()
        lx, ly = left_joy
        rx, ry = right_joy

        self.commands[0] = ly  if abs(ly) > 0.1 else 0
        self.commands[1] = -lx * 0.2 if abs(lx) > 0.1 else 0
        self.commands[2] = -rx * 0.2 if abs(rx) > 0.1 else 0
        # print("Joystick Read ",self.commands[0])
    def sensor_data_thread(self):
        commands = np.zeros(3, dtype=np.float32)

        while True:
            global hist_obs, joint_target_position_deg, joint_target_position_deg_fake, previous_command_x, previous_command_y
            start_time = time.time()

            def record_to_csv(obs, path):
                with open(path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    for row in obs:
                        formatted_row = [f"{x:.3f}" for x in row]
                        writer.writerow(formatted_row)

            """
            Step function for the RL Walker

            Input:
            - act: whether to actuate the robot or not
            """

            # 获取状态值
            imu_quat = self.client.imu_quaternion

            imu_angular_velocity_deg = self.client.imu_angular_velocity
            imu_angular_velocity = imu_angular_velocity_deg / 180.0 * math.pi  # unit : deg/s

            joint_measured_position_urdf = self.client.joint_positions
            joint_measured_velocity_urdf = self.client.joint_velocities
            imu_euler_ang_deg = self.client.imu_angles
            imu_euler_ang = imu_euler_ang_deg / 180.0 * math.pi

            joint_offset_position = joint_measured_position_urdf[0:12] - self.joint_default_position[0:12]

            vel_norm = np.sqrt(self.commands[0] ** 2 + self.commands[1] ** 2 + self.commands[2] ** 2)
            if vel_norm < 0.05:
                stand_flag = 0
            else:
                stand_flag = 1
            right_leg_phase = math.sin(
                2 * math.pi * self.count_lowlevel * stand_flag * self.cfg.dt / self.cfg.cycle_time)
            left_leg_phase = math.cos(
                2 * math.pi * self.count_lowlevel * stand_flag * self.cfg.dt / self.cfg.cycle_time)

            obs = np.zeros(self.cfg.n_proprio, dtype=np.float32)
            obs[0] = right_leg_phase
            obs[1] = left_leg_phase
            obs[2] = self.commands[0]
            obs[3] = self.commands[1]
            obs[4] = self.commands[2]
            obs[5:8] = imu_angular_velocity * self.cfg.obs_scale.ang_vel
            obs[8:10] = imu_euler_ang[:2] / 20.0
            obs[10:22] = joint_offset_position[0:12] * self.cfg.obs_scale.dof_pos
            obs[22:34] = joint_measured_velocity_urdf[0:12] * self.cfg.obs_scale.dof_vel
            obs[34:44] = self.last_action
            print("x_vel", self.commands[0])
            # print("joint_offset_position[0:12]", joint_offset_position[0:12])
            obs_hist = np.array(self.proprio_history_buf).flatten()
            self.proprio_history_buf.append(obs)
            obs_buf = np.concatenate([obs, self.priv_latent, obs_hist], dtype=np.float32)
            obs_buf = obs_buf.reshape(1, 516)
            input_name = self.sess.get_inputs()[0].name

            raw_actions = self.sess.run(None, {input_name: obs_buf})[0][0]
            self.last_action = raw_actions.copy()
            raw_actions = np.clip(raw_actions, -3, 3)
            scaled_actions = raw_actions * self.cfg.action_scale  # rad
            step_actions = np.zeros(self.cfg.num_dofs)
            step_actions[self.cfg.control_indices] = scaled_actions

            target_pos = step_actions
            joint_target_position = target_pos + self.joint_default_position[0:12]  # rad

            # parse to numpy
            # 32 dims
            joint_target_position_deg = np.rad2deg(self.joint_default_position)
            joint_target_position_deg[0:12] = np.rad2deg(joint_target_position)
            joint_target_position_deg[[5, 11]] = 0
            joint_target_position_deg_fake = np.rad2deg(self.joint_default_position)

            # 计算理想力矩
            ideal_torque = self.compute_pd_tau(joint_target_position, joint_measured_position_urdf[0:12]
                                               , joint_measured_velocity_urdf[0:12])
            cur_tau = self.client.joint_efforts[0:12]
            if self.count_lowlevel > 50:
                record_to_csv(ideal_torque.reshape(1, 12),
                              '/home/gr124ja0052/Fourier/fourier-grx-client/examples/obs/ideal_tau.csv')
                record_to_csv(cur_tau.reshape(1, 12),
                              '/home/gr124ja0052/Fourier/fourier-grx-client/examples/obs/cur_tau.csv')

            # print("队列是否为空",joint_target_queue.empty())
            if self.count_lowlevel > 50:
                self.client.move_joints(ControlGroup.ALL, joint_target_position_deg, 0.0, degrees=True)
                record_to_csv(obs.reshape(1, 44), '/home/gr124ja0052/Fourier/fourier-grx-client/examples/obs/obs.csv')

            # joint_target_queue.put(joint_target_position_deg)

            # print("joint_target_position_deg",joint_target_position_deg[0:12])
            end_time = time.time()
            duration_time = end_time - start_time
            # print("policy_time",duration_time)
            self.count_lowlevel += 1
            if duration_time > 0.01:
                # print("policy timeout")
                continue
            else:
                time.sleep(0.01 - duration_time)

            self.process_remote()
def main(
        step_freq: int = 500, act: bool = True
):
    walker = DemoNohlaRLWalk(
        step_freq=step_freq, act=act
    )

    thread_sensor_data = threading.Thread(target=walker.sensor_data_thread)
    thread_sensor_data.start()
    thread_sensor_data.join()


if __name__ == "__main__":
    init_csv('/home/gr124ja0052/Fourier/fourier-grx-client/examples/obs/obs.csv')
    init_csv('/home/gr124ja0052/Fourier/fourier-grx-client/examples/obs/ideal_tau.csv')
    init_csv('/home/gr124ja0052/Fourier/fourier-grx-client/examples/obs/cur_tau.csv')
    typer.run(main)
