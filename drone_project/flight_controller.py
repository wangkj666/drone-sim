"""
级联PID飞行控制器
外环: 位置误差 → 期望姿态 + 总推力
内环: 姿态误差 → 期望力矩 → 混控器 → 电机推力
"""
import numpy as np


class PID:
    """单轴 PID 控制器。

    当能够获得被控量的速度时，D 项使用测量速度而不是误差差分，避免
    目标位置突变时产生较大的 derivative kick。
    """
    def __init__(self, kp: float, ki: float, kd: float, integral_max: float,
                 output_limit: float | None = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max
        self.output_limit = output_limit
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, error: float, dt: float, measurement_rate: float | None = None) -> float:
        candidate_integral = np.clip(
            self._integral + error * dt,
            -self.integral_max,
            self.integral_max,
        )
        derivative = ((error - self._prev_error) / dt if dt > 0 else 0.0)
        if measurement_rate is not None:
            derivative = -measurement_rate

        unclamped = self.kp * error + self.ki * candidate_integral + self.kd * derivative
        output = unclamped
        if self.output_limit is not None:
            output = np.clip(unclamped, -self.output_limit, self.output_limit)
            # 饱和时仅允许积分项帮助控制量回到可用范围，避免积分饱和。
            if output != unclamped and error * unclamped > 0:
                candidate_integral = self._integral
                unclamped = self.kp * error + self.ki * candidate_integral + self.kd * derivative
                output = np.clip(unclamped, -self.output_limit, self.output_limit)

        self._integral = candidate_integral
        self._prev_error = error
        return output


class PositionMPC:
    """基于离散双积分模型的逐轴有限时域 MPC 外环。

    使用投影梯度法求解带加速度上下限的二次规划，避免引入额外求解器依赖。
    """

    def __init__(self, dt: float, horizon: int, position_weight: float,
                 velocity_weight: float, control_weight: float,
                 acceleration_limit: float):
        self.dt = dt
        self.horizon = horizon
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.control_weight = control_weight
        self.acceleration_limit = acceleration_limit
        self._warm_start = np.zeros(horizon)
        self._position_matrix, self._velocity_matrix = self._prediction_matrices()
        self._hessian = (
            position_weight * self._position_matrix.T @ self._position_matrix
            + velocity_weight * self._velocity_matrix.T @ self._velocity_matrix
            + control_weight * np.eye(horizon)
        )
        self._gradient_step = 1.0 / (2.0 * np.linalg.eigvalsh(self._hessian).max())

    def _prediction_matrices(self):
        position_matrix = np.zeros((self.horizon, self.horizon))
        velocity_matrix = np.zeros((self.horizon, self.horizon))
        for row in range(self.horizon):
            for column in range(row + 1):
                position_matrix[row, column] = self.dt**2 * (row - column + 0.5)
                velocity_matrix[row, column] = self.dt
        return position_matrix, velocity_matrix

    def step(self, position: float, velocity: float, target: float) -> float:
        """求解未来 horizon 步的加速度序列，仅执行第一步。"""
        steps = np.arange(1, self.horizon + 1)
        free_position = position + steps * self.dt * velocity
        free_velocity = np.full(self.horizon, velocity)
        linear_term = (
            self.position_weight * self._position_matrix.T @ (free_position - target)
            + self.velocity_weight * self._velocity_matrix.T @ free_velocity
        )

        controls = self._warm_start.copy()
        for _ in range(24):
            gradient = 2.0 * (self._hessian @ controls + linear_term)
            controls = np.clip(
                controls - self._gradient_step * gradient,
                -self.acceleration_limit,
                self.acceleration_limit,
            )
        self._warm_start = np.concatenate((controls[1:], controls[-1:]))
        return float(controls[0])


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
        self.outer_loop = cfg.get("outer_loop", "pid")

        # 外环位置PID (x, y, z 各一组)
        kp = cfg["pos_pid"]["kp"]
        ki = cfg["pos_pid"]["ki"]
        kd = cfg["pos_pid"]["kd"]
        imax = cfg["pos_pid"]["integral_max"]
        self.pid_x = PID(kp[0], ki[0], kd[0], imax, cfg["max_xy_accel"])
        self.pid_y = PID(kp[1], ki[1], kd[1], imax, cfg["max_xy_accel"])
        self.pid_z = PID(kp[2], ki[2], kd[2], imax, cfg["max_z_accel"])

        self._mpc_update_period = 1
        self._mpc_frame = 0
        self._mpc_accel = np.zeros(3)
        self._mpc = None
        if self.outer_loop == "mpc":
            mpc_cfg = cfg["mpc"]
            self._mpc_update_period = mpc_cfg["update_period"]
            control_dt = self.dt * self._mpc_update_period
            self._mpc = [
                PositionMPC(control_dt, mpc_cfg["horizon"],
                            mpc_cfg["position_weight"][axis],
                            mpc_cfg["velocity_weight"][axis],
                            mpc_cfg["control_weight"][axis],
                            cfg["max_xy_accel"] if axis < 2 else cfg["max_z_accel"])
                for axis in range(3)
            ]

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
        self._mpc_frame = 0
        self._mpc_accel = np.zeros(3)
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
        if self._mpc is None:
            ax = self.pid_x.step(self.desired_pos[0] - self.current_pos[0], dt,
                                 self.current_vel[0])
            ay = self.pid_y.step(self.desired_pos[1] - self.current_pos[1], dt,
                                 self.current_vel[1])
            az = self.pid_z.step(self.desired_pos[2] - self.current_pos[2], dt,
                                 self.current_vel[2])
        else:
            if self._mpc_frame % self._mpc_update_period == 0:
                self._mpc_accel = np.array([
                    controller.step(self.current_pos[axis], self.current_vel[axis],
                                    self.desired_pos[axis])
                    for axis, controller in enumerate(self._mpc)
                ])
            self._mpc_frame += 1
            ax, ay, az = self._mpc_accel

        # ── 重力补偿 + PID → 总推力 + 期望姿态 ──
        total_thrust = m * np.sqrt(max(ax**2 + ay**2, 1e-6) + (g + max(az, -g*0.5))**2)
        total_thrust = np.clip(total_thrust, 0, m * g * 3)  # 上限3倍重力

        # 机体系为 X 向右、Y 向前、Z 向上。正 roll 产生 -Y 水平推力，
        # 正 pitch 产生 +X 水平推力，因此这里的符号必须与混控器一致。
        desired_roll  = np.clip(np.arctan2(-ay, g + az), -self.max_tilt, self.max_tilt)
        desired_pitch = np.clip(np.arctan2(ax, np.sqrt(max(ay**2 + (g + az)**2, 1e-6))),
                                 -self.max_tilt, self.max_tilt)

        # ── 内环：姿态误差 → 力矩 ──
        roll_err  = self._angle_error(desired_roll,  self.current_att[0])
        pitch_err = self._angle_error(desired_pitch, self.current_att[1])
        yaw_err   = self._angle_error(self.desired_yaw, self.current_att[2])

        roll_moment  = self.pid_roll.step(roll_err, dt, self.current_angvel[0])
        pitch_moment = self.pid_pitch.step(pitch_err, dt, self.current_angvel[1])
        yaw_moment   = self.pid_yaw.step(yaw_err, dt, self.current_angvel[2])

        # ── 混控器 → 电机推力 ──
        return self.mixer.mix(total_thrust, roll_moment, pitch_moment, yaw_moment)

    @staticmethod
    def _angle_error(target: float, current: float) -> float:
        err = target - current
        return np.arctan2(np.sin(err), np.cos(err))
