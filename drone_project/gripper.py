"""
双侧弯齿条夹爪 — 按结构图实现

      (  [机身]  )      ← 左右各一条弯齿条(括号状), 齿在内侧
         ╲  ╱
          ●             ← 中心连杆关节

运动: 两条弯齿条绕机身两侧的轴转动
      - 正角度 = 尖端向中间合拢(夹紧)
      - 整体可翻转 180°: 朝下抓地面 / 朝上抓上方
"""
import numpy as np


class Gripper:
    def __init__(self, prim_path: str = "/World/Drone/Gripper"):
        self.prim_path = prim_path
        self._pivot_data = []      # [(rotate_op, sign), ...]
        self._mode = 0.0           # 0=朝下, 180=朝上
        self._grip = -6.0          # 正=夹紧, 负=张开
        self._cur = -6.0           # 当前实际角度
        self._anim_speed = 90.0    # 度/秒
        self._current_angle = self._cur
        self._built = False

    # ── 构建 ────────────────────────────────────────────────
    def build(self):
        from omni.isaac.core.utils.prims import create_prim, set_prim_property
        from pxr import UsdGeom, Gf
        import omni.usd

        stage = omni.usd.get_context().get_stage()

        def part(path, prim_type, translate, scale, color=None, rot_y=0.0, rot_x=0.0):
            """建零件并写【局部】变换"""
            create_prim(path, prim_type)
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            xf.ClearXformOpOrder()
            xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate))
            if rot_x:
                xf.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble).Set(float(rot_x))
            if rot_y:
                xf.AddRotateYOp(UsdGeom.XformOp.PrecisionDouble).Set(float(rot_y))
            xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))
            if color:
                set_prim_property(path, "primvars:displayColor", [color])

        DARK = (0.09, 0.09, 0.10)
        GREY = (0.34, 0.34, 0.37)
        RED = (0.85, 0.10, 0.10)

        # 夹爪装在机身正中央 (y=0), 两条臂在左右两侧上下摆动。
        # 避免穿模的办法: 臂的弧段【从外侧绕上去】——
        # 弧段若经过"正朝内"(α=180°)方向, 会正好压在机身板上;
        # 因此朝上姿态用负角度旋转(绕外侧), 全程避开该方向。
        GRIP_Y = 0.0

        # ── 1. 连接梁 (把机构连到机身) ──
        part(f"{self.prim_path}/Mount", "Cube",
             (0.0, -0.122, 0.0), (0.060, 0.030, 0.014), GREY)

        # ── 2. 舵机盒 + 齿轮组 (在机构平面内, 两轮啮合) ──
        part(f"{self.prim_path}/ServoBox", "Cube",
             (0.0, GRIP_Y, -0.028), (0.036, 0.026, 0.026), DARK)
        self._add_gear(part, f"{self.prim_path}/GearR",
                       (0.030, GRIP_Y, -0.058), 0.028, 0.012, 12, GREY)
        self._add_gear(part, f"{self.prim_path}/GearL",
                       (-0.030, GRIP_Y, -0.058), 0.028, 0.012, 12, GREY)

        # ── 3. 两侧: 弯齿条臂 ──
        for name, s in [("Right", 1), ("Left", -1)]:
            x_off = s * 0.115

            # 枢轴 (转轴沿Y)
            part(f"{self.prim_path}/{name}/Axle", "Cylinder",
                 (x_off, GRIP_Y, 0.0), (0.016, 0.016, 0.030), GREY, rot_x=90.0)

            pivot_path = f"{self.prim_path}/{name}/Pivot"
            create_prim(pivot_path, "Xform")
            pivot_xf = UsdGeom.Xformable(stage.GetPrimAtPath(pivot_path))
            pivot_xf.ClearXformOpOrder()
            pivot_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(x_off, GRIP_Y, 0.0))
            rot_op = pivot_xf.AddRotateYOp(UsdGeom.XformOp.PrecisionDouble)
            rot_op.Set(-45.0 * s)
            self._pivot_data.append((rot_op, s))

            # 弯齿条
            self._add_curved_rack(part, pivot_path, s)

        # ── 4. 中心连杆 + 关节 (固定在机构底座上, 不会甩飞) ──
        for s, nm in ((1, "R"), (-1, "L")):
            part(f"{self.prim_path}/Link{nm}", "Cube",
                 (s * 0.038, GRIP_Y, -0.098), (0.005, 0.010, 0.045),
                 (0.30, 0.30, 0.62), rot_y=-s * 42.0)
        part(f"{self.prim_path}/Joint", "Sphere",
             (0.0, GRIP_Y, -0.128), (0.015, 0.015, 0.015), RED)

        self._built = True
        print("[Gripper] 双侧弯齿条夹爪已安装 (可朝下/朝上)")
        return self

    @staticmethod
    def _add_gear(part, base_path, center, radius, half_thick, n_teeth, color):
        part(f"{base_path}/Disc", "Cylinder",
             center, (radius, radius, half_thick), color, rot_x=90.0)
        for k in range(n_teeth):
            a = 2.0 * np.pi * k / n_teeth
            part(f"{base_path}/T{k}", "Cube",
                 (center[0] + radius * np.cos(a), center[1],
                  center[2] + radius * np.sin(a)),
                 (0.0045, half_thick * 0.85, 0.006), color,
                 rot_y=90.0 - np.degrees(a))

    @staticmethod
    def _add_curved_rack(part, pivot_path, s):
        """弯齿条: 从枢轴外侧绕过, 尖端弯向中间; 齿在凹面(朝枢轴一侧)"""
        # 注意: 曲率半径必须小于【枢轴到中线的距离】(0.115m),
        # 否则臂转到朝上时会扫过中线, 两条臂互相穿插。
        R = 0.105                       # 曲率半径
        th0, th1 = np.radians(8.0), np.radians(125.0)
        n_seg = 12
        arm_half_t = 0.010              # 臂厚(半径方向)
        arm_half_w = 0.022              # 臂宽(Y方向)
        tooth_h = 0.010
        dth = (th1 - th0) / n_seg
        seg_len = R * dth

        for i in range(n_seg):
            th = th0 + (i + 0.5) * dth
            if i in (0, n_seg - 1):     # 首尾稍短, 收口好看
                L = seg_len * 0.62
            else:
                L = seg_len * 0.72
            px = s * R * np.cos(th)
            pz = -R * np.sin(th)
            part(f"{pivot_path}/Seg{i}", "Cube",
                 (px, 0.0, pz), (arm_half_t, arm_half_w, L),
                 (0.15, 0.15, 0.16), rot_y=np.degrees(s * th))

            # 齿: 朝枢轴(凹面)一侧
            d = arm_half_t + tooth_h
            tx = s * (R - d) * np.cos(th)
            tz = -(R - d) * np.sin(th)
            part(f"{pivot_path}/Tooth{i}", "Cube",
                 (tx, 0.0, tz), (tooth_h, arm_half_w * 0.5, seg_len * 0.24),
                 (0.55, 0.55, 0.58), rot_y=np.degrees(s * th))

    # ── 运动 ────────────────────────────────────────────────
    def update(self, dt: float):
        """每帧调用, 平滑插值"""
        # 朝上时臂转过90°以上, 张合方向随之反向
        sign = 1.0 if self._mode == 0.0 else -1.0
        tgt = self._mode + self._grip * sign
        if abs(tgt - self._cur) < 0.3:
            return
        step = self._anim_speed * dt
        self._cur += float(np.clip(tgt - self._cur, -step, step))
        self._current_angle = self._cur
        for op, s in self._pivot_data:
            op.Set(float(self._cur * s))

    def open(self):
        """两臂张开 (爪口约0.230m)"""
        self._grip = -30.0

    def close(self):
        """两臂合拢夹紧 (爪口约0.095m, 夹住0.10m的目标)"""
        self._grip = 10.0

    def point_down(self):
        """整体朝下 (抓地面目标)"""
        self._mode = 0.0

    def point_up(self):
        """整体朝上 (向上抓取/栖息)。
        用负角度(从外侧绕上去)而不是正角度(从内侧穿过机身)——
        否则臂的弧段会正好压在机身板上形成穿模。
        -150 是实测值: 配上 close() 后合拢到 -160, 臂表面正好贴住 r=0.022 的横杆
        (verify_perch.py 量出间隙 -0.7mm, 不穿模)。"""
        self._mode = -150.0

    def reset(self):
        """初始姿态: 两臂微微朝上张开 (备战状态)"""
        self._mode = 0.0
        self._grip = -45.0          # 角度越小臂越朝上
        self._cur = -45.0
        self._current_angle = self._cur
        for op, s in self._pivot_data:
            op.Set(float(self._cur * s))
