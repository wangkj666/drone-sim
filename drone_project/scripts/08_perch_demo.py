"""
自主栖息演示 — 无人机飞到横杆下方, 夹爪翻转朝上, 抱住横杆悬挂

流程: 起飞 → 飞到栖枝正下方 → 夹爪转朝上+张开 → 上升让横杆进入爪口
      → 闭臂夹住 → 栖息保持 → 松开脱离

横杆沿 Y 方向布置(穿过旋翼之间), 这是真实无人机停树枝的姿态。
"""
import torch
from isaacsim import SimulationApp
import os
_HEADLESS = os.environ.get("DRONE_HEADLESS", "0") == "1"
simulation_app = SimulationApp({"headless": _HEADLESS})

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim, set_prim_property

from drone_model import DroneModel
from gripper import Gripper

cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
with open(cfg_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

world = World(physics_dt=config["sim"]["physics_dt"])
world.scene.add_default_ground_plane()

import omni.usd
from pxr import UsdGeom, Gf
stage = omni.usd.get_context().get_stage()

# ── 栖枝: 立柱 + 横杆 (横杆沿Y, 穿过旋翼之间) ──
PERCH_X, PERCH_Y, PERCH_Z = 0.5, 0.0, 1.30
BAR_R, BAR_HALF = 0.020, 0.45    # 杆径4cm, 配合爪口尺寸

def make_cyl(path, translate, scale, color, rot_x=0.0):
    """圆柱: 需要 [平移, 旋转, 缩放] 的顺序, 否则旋转后形状会被压扁"""
    create_prim(path, "Cylinder")
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate))
    if rot_x:
        xf.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble).Set(float(rot_x))
    xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))
    set_prim_property(path, "primvars:displayColor", [color])

# 立柱 (在横杆一端, 不会挡到无人机)
make_cyl("/World/Perch/Pole", (PERCH_X, BAR_HALF, PERCH_Z / 2),
         (0.030, 0.030, PERCH_Z / 2), (0.35, 0.28, 0.18))
# 横杆 (绕X转90° → 轴向变成Y)
make_cyl("/World/Perch/Bar", (PERCH_X, PERCH_Y, PERCH_Z),
         (BAR_R, BAR_R, BAR_HALF), (0.45, 0.36, 0.22), rot_x=90.0)

# ── 无人机 + 夹爪 ──
drone = DroneModel(config).build(world, kinematic=True)
world.reset()
gripper = Gripper().build()

drone_prim = stage.GetPrimAtPath("/World/Drone")
xf = UsdGeom.Xformable(drone_prim)
translate_op = None
for op in xf.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        translate_op = op
        break

dt = config["sim"]["physics_dt"]
speed = 0.45
current = np.array(config["drone"]["init_position"], dtype=float)

# 栖息高度: 夹爪朝上时爪口在机心上方约6cm, 让横杆正好落进爪口。
# (再低会让横杆插进机身顶板, 实测机身顶面在机心上方0.02)
PERCH_DRONE_Z = PERCH_Z - 0.060

# 状态机
# 0起飞 1飞到杆下 2夹爪转朝上 3上升入爪口 4闭臂 5栖息 6脱离 7完成
phase = 0
hover_target = np.array([0.0, 0.0, 0.80])
hold = 0

print("=" * 58)
print("  自主栖息演示")
print("  起飞 → 飞到栖枝下 → 夹爪朝上 → 上升抱杆 → 栖息 → 脱离")
print("=" * 58)

frame = 0
while simulation_app.is_running():
    d = np.linalg.norm(hover_target - current)
    if d > 0.004:
        current += (hover_target - current) * min(speed * dt, d) / d

    move = hover_target - current
    roll  = np.clip(-move[1] * 0.5, -0.2, 0.2)
    pitch = np.clip( move[0] * 0.5, -0.2, 0.2)
    if translate_op:
        translate_op.Set(Gf.Vec3d(*current))

    gripper.update(dt)
    world.step(render=True)
    frame += 1

    if phase == 0:
        if d < 0.08:
            hover_target[:] = [PERCH_X, PERCH_Y, 0.80]
            phase = 1
            print(f"  [{frame:4d}] 起飞完成 → 飞向栖枝下方")

    elif phase == 1:
        if d < 0.10:
            gripper.point_up()      # 夹爪翻转朝上
            gripper.open()          # 张开等待横杆进入
            phase, hold = 2, 90     # 等1.5秒让翻转动画走完
            print(f"  [{frame:4d}] 到达栖枝下方, 夹爪翻转朝上")

    elif phase == 2:
        hold -= 1
        if hold <= 0:
            hover_target[2] = PERCH_DRONE_Z
            phase = 3
            print(f"  [{frame:4d}] 缓慢上升, 让横杆进入爪口")

    elif phase == 3:
        if d < 0.02:
            gripper.close()         # 夹住横杆
            phase, hold = 4, 45
            print(f"  [{frame:4d}] 横杆到位, 闭臂夹住!")

    elif phase == 4:
        hold -= 1
        if hold <= 0:
            phase, hold = 5, 360    # 栖息保持6秒
            print(f"  [{frame:4d}] ★ 栖息中 (夹爪抱住横杆, 位置锁定)")

    elif phase == 5:
        hold -= 1
        if hold <= 0:
            gripper.open()          # 松开
            hover_target[2] = 0.80
            phase, hold = 6, 60
            print(f"  [{frame:4d}] 松开, 脱离横杆")

    elif phase == 6:
        hold -= 1
        if d < 0.08 and hold < 30:
            phase = 7
            print(f"  [{frame:4d}] 完成! 栖息+脱离全流程成功")

    if frame % 120 == 0:
        names = ["起飞", "飞向栖枝", "夹爪朝上", "上升入爪", "闭臂", "栖息中", "脱离", "完成"]
        print(f"  [{frame:5d}] pos=({current[0]:+.2f},{current[1]:+.2f},{current[2]:.2f}) "
              f"横杆z={PERCH_Z:.2f} [{names[phase]}]")

simulation_app.close()
