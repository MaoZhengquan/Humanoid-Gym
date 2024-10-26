import numpy as np
import torch

def append_data_to_csv(file_name,data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    elif isinstance(data, float):
        data = np.array([[data]], dtype=np.float32)
    else:
        data = np.array(data, dtype=np.float32)
    # data = data.astype(np.float32)
    # if isinstance(data,(int,float,np.float32)):
    #     print(1)
    #     data = np.array([[data]],dtype=np.float32)
    # elif isinstance(data,(list,np.array)) and np.ndim(data) == 1:
    #     print(1)
    #     data = np.array([data],dtype=np.float32)
    # print(1)
    # if np.ndim(data) != 2:
    #     raise ValueError("Data must be a scalar")
    with open(file_name,'a') as f:
        np.savetxt(f,data,delimiter=",",fmt='%.6f')
def quat_to_rot_matrix(q):
    x, y, z, w = q
    return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])
def rot_matrix_to_quat(R):
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw])


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

