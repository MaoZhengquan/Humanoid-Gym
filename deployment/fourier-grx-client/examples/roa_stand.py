import time
from rich.table import Table
import numpy as np
import typer
from fourier_grx_client import ControlGroup, RobotClient, ControlMode
from ischedule import run_loop, schedule
from scipy.signal import freqs
from torch.fx.experimental.unification.dispatch import namespace


class DemoNohlaStand:
    """
    Reinforcement Learning Walker
    """

    def __init__(self, step_freq=100, act=True):
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

        self.set_gains()
        self.client.enable()

        # algorithm
        self.move_count = 0
        self.joint_start_position = None

    def set_gains(self):
        """
        Set gains for the robot
        """

        # fmt: off

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
        pd_kp = [
            251.625, 362.52, 400, 400, 10.98, 0,  # left leg
            251.625, 362.52, 400, 400, 10.98, 0,  # right leg
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
        print("control mode",self.client.get_control_modes())

        # fmt: on
        self.client.set_gains(position_control_kp=pos_kp, velocity_control_kp=vel_kp, pd_control_kp=pd_kp,
                              pd_control_kd=pd_kd)
        print("control mode",self.client.get_control_modes())

    def stand(self):
        move_period = 100

        global joint_start_position
        # get states 32 dims
        table = Table("Type", "Data", title="Current :robot: states (in radians)")

        # print("joint_velocity", self.client.joint_velocities[0:12])
        # print("joint_position", self.client.joint_positions[0:12])
        # print("imu_angular_velocity", self.client.imu_angular_velocity)


        joint_measured_position_rad = self.client.joint_positions
        joint_measured_velocity_rad = self.client.joint_velocities

        if self.joint_start_position is None:
            self.joint_start_position = np.array(joint_measured_position_rad)
            # print("joint_start_position = \n", np.round(self.joint_start_position, 1))

        # set end position
        # deg
        joint_end_position = np.array([
            0.0, 0.0, -0.4, 0.8, -0.4, -0.0,  # left leg (6)
            -0.0, 0.0, -0.4, 0.8, -0.4, 0.0,  # right leg (6)
            0.0, -0.3, 0.0,  # waist (3)
            0.0, 0.0, 0.0,  # waist (3)
            0.0, 0.20, 0.0, -0.3, 0.0, 0.0, 0.0,  # left arm (4)
            0.0, -0.20, 0.0, -0.3, 0.0, 0.0, 0.0, # right arm (4)
        ])

        # update move ratio
        move_ratio = min(self.move_count / move_period, 1)

        # update target position 32 dim
        joint_target_position_controlled = self.joint_start_position + \
                                           (joint_end_position - self.joint_start_position) * move_ratio
        # print("joint_end_position",joint_end_position)
        # print("joint_start_position",self.joint_start_position)
        # print("act",self.act)
        # update count
        self.move_count += 1

        # print info
        # print("move_ratio = ", np.round(move_ratio * 100, 1), "%")

        if move_ratio < 1:
            finish_flag = False
        else:
            finish_flag = True
        if self.act:
            if finish_flag == False:
                self.client.move_joints(ControlGroup.ALL, joint_target_position_controlled, 0.0,degrees=False)

        return joint_target_position_controlled, finish_flag


def main(step_freq: int = 100, act: bool = True):
    walker = DemoNohlaStand(step_freq=step_freq, act=act)

    # start the scheduler
    # schedule(walker.step, interval=1 / step_freq)
    schedule(walker.stand, interval=1 / step_freq)

    # run the scheduler
    run_loop()


if __name__ == "__main__":
    typer.run(main)
