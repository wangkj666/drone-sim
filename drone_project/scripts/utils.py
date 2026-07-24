"""工具函数"""
import numpy as np


def euler_to_quaternion(roll, pitch, yaw):
    """Euler角 (rad) → 四元数 [x, y, z, w]"""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def quaternion_to_euler(q):
    """四元数 [x, y, z, w] → Euler角 [roll, pitch, yaw] (rad)"""
    x, y, z, w = q
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])


def body_to_world(body_vec, quaternion):
    """机体坐标系 → 世界坐标系"""
    q = quaternion
    rot = np.array([
        [1-2*(q[1]**2+q[2]**2), 2*(q[0]*q[1]-q[2]*q[3]), 2*(q[0]*q[2]+q[1]*q[3])],
        [2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[0]**2+q[2]**2), 2*(q[1]*q[2]-q[0]*q[3])],
        [2*(q[0]*q[2]-q[1]*q[3]), 2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[0]**2+q[1]**2)],
    ])
    return rot @ np.array(body_vec)
