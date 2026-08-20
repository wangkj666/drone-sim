"""
侧装旋转夹爪 — Pivot绕Y轴旋转, 平滑动画, 可夹物体
"""
import numpy as np


class Gripper:
    def __init__(self, prim_path: str = "/World/Drone/Gripper"):
        self.prim_path = prim_path
        self._pivot_ops = []
        self._current_angle = 0.0
        self._target_angle = 0.0
        self._anim_speed = 60.0  # 度/秒
        self._claw_paths = []    # 夹爪末端路径, 用于位置参考
        self._built = False

    def build(self):
        from omni.isaac.core.utils.prims import create_prim, set_prim_property
        from pxr import UsdGeom, Gf
        import omni.usd

        stage = omni.usd.get_context().get_stage()

        for name, x_off, rot_dir in [("Left", 0.14, -1), ("Right", -0.14, 1)]:
            base = f"{self.prim_path}/{name}"

            # 固定安装座
            create_prim(f"{base}/Bracket", "Cube",
                        position=np.array([x_off, 0, -0.02]),
                        scale=np.array([0.03, 0.04, 0.03]))
            set_prim_property(f"{base}/Bracket", "primvars:displayColor", [(0.3, 0.3, 0.3)])

            # Pivot (旋转节点)
            pivot_path = f"{base}/Pivot"
            create_prim(pivot_path, "Xform",
                        position=np.array([x_off, 0, 0.01]))

            pivot_prim = stage.GetPrimAtPath(pivot_path)
            pivot_xf = UsdGeom.Xformable(pivot_prim)
            pivot_xf.ClearXformOpOrder()
            pivot_xf.AddTranslateOp().Set(Gf.Vec3d(x_off, 0, 0.01))
            rot_op = pivot_xf.AddRotateYOp()
            rot_op.Set(0.0)
            self._pivot_ops.append(rot_op)

            # 上臂 (pivot向下延伸)
            create_prim(f"{pivot_path}/Arm", "Cylinder",
                        position=np.array([0, 0, -0.10]),
                        scale=np.array([0.012, 0.012, 0.12]))
            set_prim_property(f"{pivot_path}/Arm", "primvars:displayColor", [(0.25, 0.25, 0.25)])

            # 夹爪 (末端)
            claw_color = [(1, 0.2, 0.2)] if "Left" in name else [(0.2, 0.2, 1)]
            claw_path = f"{pivot_path}/Claw"
            create_prim(claw_path, "Cube",
                        position=np.array([0, 0, -0.22]),
                        scale=np.array([0.03, 0.06, 0.06]))
            set_prim_property(claw_path, "primvars:displayColor", claw_color)
            self._claw_paths.append(claw_path)

        self._built = True
        print(f"[Gripper] 双侧旋转臂已安装")
        return self

    def update(self, dt: float):
        """每帧调用, 平滑动画"""
        if abs(self._target_angle - self._current_angle) < 0.5:
            return
        step = self._anim_speed * dt
        if self._target_angle > self._current_angle:
            self._current_angle = min(self._current_angle + step, self._target_angle)
        else:
            self._current_angle = max(self._current_angle - step, self._target_angle)

        from pxr import Gf
        for op in self._pivot_ops:
            op.Set(float(self._current_angle))

    def set_angle(self, angle_deg: float):
        self._target_angle = angle_deg

    def open(self):
        """臂向外张开"""
        self.set_angle(80)

    def close(self):
        """臂向内夹紧"""
        self.set_angle(-30)

    def reset(self):
        self.set_angle(0)
        self._current_angle = 0
        for op in self._pivot_ops:
            op.Set(0.0)
