"""
级联PID飞行控制器
外环: 位置误差 → 期望姿态 + 总推力
内环: 姿态误差 → 期望力矩 → 混控器 → 电机推力
"""
import numpy as np


class PID:
    """单轴PID控制器"""
    def __init__(self, kp: float, ki: float, kd: float, integral_max: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, error: float, dt: float) -> float:
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.integral_max, self.integral_max)
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative


class FlightController:
    """
    四旋翼级联PID飞行控制器

    控制流程:
      des_pos(x,y,z) + des_yaw
          │
      [位置PID] ──→ accel_xyz ──→ thrust + desired_roll/pitch
          │
      [姿态PID] ──→ roll/pitch/yaw moments
          │
      [混控器]   ──→ 4×motor_thrust
    """

    def __init__(self, config: dict, mixer):
        cfg = config["control"]
        d = config["drone"]
        sim = config["sim"]

        self.mass = d["mass"]
        self.g = sim["gravity"]
        self.max_tilt = cfg["max_tilt"]
        self.dt = sim["physics_dt"]

        # 外环位置PID (x, y, z 各一组)
        kp = cfg["pos_pid"]["kp"]
        ki = cfg["pos_pid"]["ki"]
        kd = cfg["pos_pid"]["kd"]
        imax = cfg["pos_pid"]["integral_max"]
        self.pid_x = PID(kp[0], ki[0], kd[0], imax)
        self.pid_y = PID(kp[1], ki[1], kd[1], imax)
        self.pid_z = PID(kp[2], ki[2], kd[2], imax)

        # 内环姿态PID (roll, pitch, yaw 各一组)
        akp = cfg["att_pid"]["kp"]
        aki = cfg["att_pid"]["ki"]
        akd = cfg["att_pid"]["kd"]
        aimax = cfg["att_pid"]["integral_max"]
        self.pid_roll  = PID(akp[0], aki[0], akd[0], aimax)
        self.pid_pitch = PID(akp[1], aki[1], akd[1], aimax)
        self.pid_yaw   = PID(akp[2], aki[2], akd[2], aimax)

        self.mixer = mixer

        # 状态
        self.desired_pos = np.zeros(3)
        self.desired_yaw = 0.0
        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.current_att = np.zeros(3)  # [roll, pitch, yaw]
        self.current_angvel = np.zeros(3)

    def reset(self):
        for pid in [self.pid_x, self.pid_y, self.pid_z,
                     self.pid_roll, self.pid_pitch, self.pid_yaw]:
            pid.reset()
        self.desired_pos = np.zeros(3)
        self.desired_yaw = 0.0

    def set_target(self, position: np.ndarray, yaw: float = 0.0):
        self.desired_pos = np.array(position)
        self.desired_yaw = yaw

    def update_state(self, position: np.ndarray, velocity: np.ndarray,
                     attitude: np.ndarray, angular_vel: np.ndarray):
        self.current_pos = np.array(position)
        self.current_vel = np.array(velocity)
        self.current_att = np.array(attitude)
        self.current_angvel = np.array(angular_vel)

    def compute_control(self) -> np.ndarray:
        """返回 [M1, M2, M3, M4] 电机推力 (N)"""
        dt = self.dt
        g = self.g
        m = self.mass

        # ── 外环：位置误差 → 加速度 ──
        ax = self.pid_x.step(self.desired_pos[0] - self.current_pos[0], dt)
        ay = self.pid_y.step(self.desired_pos[1] - self.current_pos[1], dt)
        az = self.pid_z.step(self.desired_pos[2] - self.current_pos[2], dt)

        # 速度阻尼
        ax -= 0.5 * self.current_vel[0]
        ay -= 0.5 * self.current_vel[1]
        az -= 0.5 * self.current_vel[2]

        # ── 重力补偿 + PID → 总推力 + 期望姿态 ──
        total_thrust = m * np.sqrt(max(ax**2 + ay**2, 1e-6) + (g + max(az, -g*0.5))**2)
        total_thrust = np.clip(total_thrust, 0, m * g * 3)  # 上限3倍重力

        desired_roll  = np.clip(np.arctan2(ay, g + az), -self.max_tilt, self.max_tilt)
        desired_pitch = np.clip(np.arctan2(-ax, np.sqrt(max(ay**2 + (g + az)**2, 1e-6))),
                                 -self.max_tilt, self.max_tilt)

        # ── 内环：姿态误差 → 力矩 ──
        roll_err  = self._angle_error(desired_roll,  self.current_att[0])
        pitch_err = self._angle_error(desired_pitch, self.current_att[1])
        yaw_err   = self._angle_error(self.desired_yaw, self.current_att[2])

        roll_moment  = self.pid_roll.step(roll_err, dt)   - 0.3 * self.current_angvel[0]
        pitch_moment = self.pid_pitch.step(pitch_err, dt) - 0.3 * self.current_angvel[1]
        yaw_moment   = self.pid_yaw.step(yaw_err, dt)    - 0.3 * self.current_angvel[2]

        # ── 混控器 → 电机推力 ──
        return self.mixer.mix(total_thrust, roll_moment, pitch_moment, yaw_moment)

    @staticmethod
    def _angle_error(target: float, current: float) -> float:
        err = target - current
        return np.arctan2(np.sin(err), np.cos(err))
