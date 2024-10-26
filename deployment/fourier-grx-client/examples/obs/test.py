import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
joint_default_position = np.array([
            0.0, 0.0, -0.4, 0.8, -0.4, -0.0,  # left leg (6)
            -0.0, 0.0, -0.4, 0.8, -0.4, 0.0,  # right leg (6)
            0.0, 0.0, 0.0,  # waist (3)
            0.0, 0.0, 0.0,  # waist (3)
            0.0, 0.2, 0.0, -0.2, 0.0, 0.0, 0.0,  # left arm (4)
            0.0, -0.2, 0.0, -0.2, 0.0, 0.0, 0.0, # right arm (4)
        ])
data = pd.read_csv('/home/gr124ja0052/Fourier/fourier-grx-client/examples/obs/obs_mujoco.csv', skiprows=92,
                               header=None)

mujoco_ang_left = data.iloc[:, 34:39].values / 3.0
mujoco_ang_right = data.iloc[:, 39:44].values / 3.0
new_data = []
print(len(mujoco_ang_left))
for index in range(len(mujoco_ang_left)):
    # 从joint_default_position复制ascii码
    joint_target_position_deg = joint_default_position.copy()
    # 更新特定关节位置值（示例更新）
    joint_target_position_deg[0:5] += mujoco_ang_left[index] * 0.5
    joint_target_position_deg[6:11] += mujoco_ang_right[index] * 0.5

    # 将计算结果追加到列表中
    new_data.append(joint_target_position_deg)

# 将新数据转换为DataFrame
new_df = pd.DataFrame(new_data)
new_df.to_csv('/home/gr124ja0052/Fourier/fourier-grx-client/examples/obs/new_obs_mujoco.csv', index=False, header=False)

new_data = pd.read_csv('/home/gr124ja0052/Fourier/fourier-grx-client/examples/obs/new_obs_mujoco.csv', header=None)

# 提取第三列的数据
third_column_values = new_data.iloc[:200, 2].values

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(third_column_values, label="Third Value Curve")
plt.title("Curve of the Third Value")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()