import numpy as np

# 欧拉角 (roll, pitch, yaw) in radians
roll = 180 / 180.0 * 3.14  # 180 degrees
pitch = 0     # 0 degrees
yaw = 180 / 180.0 * 3.14  # 180 degrees

# 计算四元数 (qx, qy, qz, qw)
qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

print(qw, qx, qy, qz)
