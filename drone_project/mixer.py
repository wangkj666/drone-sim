"""
四旋翼混控器 — 将 [推力, 滚转, 俯仰, 偏航] 映射到四个电机转速
X型 / 十字型 两种配置
"""
import numpy as np


class QuadMixer:
    """
    X型四旋翼混控器

    电机布局（俯视，机头朝前 = +Y方向）:
           M1(前右)        M2(前左)
                \        /
                  [机身]
                /        \
           M4(后左)        M3(后右)
    """

    def __init__(self, config: dict):
        """根据配置初始化混控矩阵"""
        self.arm_length = config["drone"]["arm_length"]
        self.torque_const = config["control"]["thrust_to_torque"]
        mixer_type = config["control"]["mixer_type"]
        self.min_thrust = config["control"]["min_thrust"]
        self.max_thrust = config["control"]["max_thrust"]

        L = self.arm_length
        k = self.torque_const

        if mixer_type == "x":
            # X型: 机臂与Y轴夹角45度
            # M1(FR, CW) M2(FL, CCW) M3(RL, CCW) M4(RR, CW)
            self.matrix = np.array([
                [1/4,  1/(4*L), -1/(4*L), -1/(4*k)],  # M1 FR
                [1/4,  1/(4*L),  1/(4*L),  1/(4*k)],  # M2 FL
                [1/4, -1/(4*L),  1/(4*L), -1/(4*k)],  # M3 RL
                [1/4, -1/(4*L), -1/(4*L),  1/(4*k)],  # M4 RR
            ])
        else:
            self.matrix = np.array([
                [1/4,  0,       1/(2*L), -1/(2*k)],
                [1/4, -1/(2*L), 0,        1/(2*k)],
                [1/4,  0,      -1/(2*L), -1/(2*k)],
                [1/4,  1/(2*L), 0,        1/(2*k)],
            ])

    def mix(self, thrust: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        输入: 总推力T(N), roll_moment, pitch_moment, yaw_moment
        输出: 4个电机推力 [M1, M2, M3, M4] (N)
        """
        cmd = np.array([thrust, roll, pitch, yaw])
        motor_thrusts = self.matrix @ cmd
        return np.clip(motor_thrusts, self.min_thrust, self.max_thrust)
