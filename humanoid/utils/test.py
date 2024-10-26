import numpy as np
import pandas as pd

# 设置起点q1的值
q1 = 78  # 你可以更改为任何起点值

def compute_trajectory(q1, T=10, q2=0):
    A = np.zeros((8, 8))  # 使用8x8的方阵
    b = np.zeros(8)

    # 1. q(0) = q1
    A[0, 0] = 1
    b[0] = q1

    # 2. q(10) = q2
    A[1, 0] = 1
    A[1, 1] = T
    A[1, 2] = T**2
    A[1, 3] = T**3
    A[1, 4] = T**4
    A[1, 5] = T**5
    A[1, 6] = T**6
    A[1, 7] = T**7
    b[1] = q2

    # 3. dot(q)(0) = 0
    A[2, 1] = 1

    # 4. dot(q)(10) = 0
    A[3, 1] = 1
    A[3, 2] = 2 * T
    A[3, 3] = 3 * T**2
    A[3, 4] = 4 * T**3
    A[3, 5] = 5 * T**4
    A[3, 6] = 6 * T**5
    A[3, 7] = 7 * T**6

    # 5. dot(dot(q))(0) = 0
    A[4, 2] = 2

    # 6. dot(dot(q))(10) = 0
    A[5, 2] = 2
    A[5, 3] = 6 * T
    A[5, 4] = 12 * T**2
    A[5, 5] = 20 * T**3
    A[5, 6] = 30 * T**4
    A[5, 7] = 42 * T**5

    # 7. dot(dot(dot(q)))(0) = 0
    A[6, 3] = 6

    # 8. dot(dot(dot(q)))(10) = 0
    A[7, 3] = 6
    A[7, 4] = 12 * T
    A[7, 5] = 20 * T**2
    A[7, 6] = 30 * T**3
    A[7, 7] = 42 * T**4

    # 打印行列式以检查奇异性
    print("行列式:", np.linalg.det(A))

    coeffs = np.linalg.solve(A, b)
    return coeffs

def trajectory(t, coeffs):
    return sum(c * t**i for i, c in enumerate(coeffs))

def velocity(t, coeffs):
    return sum(i * c * t**(i - 1) for i, c in enumerate(coeffs) if i > 0)

def acceleration(t, coeffs):
    return sum(i * (i - 1) * c * t**(i - 2) for i, c in enumerate(coeffs) if i > 1)

# 调用compute_trajectory，使用设定的q1
coeffs = compute_trajectory(q1)

# 打印多项式系数
print("多项式系数:", coeffs)

# 计算和绘制曲线
time_points = np.linspace(0, 10, num=50)
trajectory_values = [trajectory(t, coeffs) for t in time_points]
velocity_values = [velocity(t, coeffs) for t in time_points]
acceleration_values = [acceleration(t, coeffs) for t in time_points]

# 保存数据
data = {
    '位移 q/°': trajectory_values
}
df = pd.DataFrame(data)
print(data)