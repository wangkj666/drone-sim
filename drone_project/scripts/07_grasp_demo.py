"""
侧装旋转臂抓取演示
起飞→搜索→接近→开臂→下降→闭臂夹取→携物返航
"""
import torch
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim, set_prim_property

from drone_model import DroneModel
from camera import DroneCamera
from perception import ColorDetector
from gripper import Gripper

cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
with open(cfg_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

world = World(physics_dt=config["sim"]["physics_dt"])
world.scene.add_default_ground_plane()

# 目标方块
create_prim("/World/Target", "Cube",
            position=np.array([0.3, 0.0, 0.2]),
            scale=np.array([0.12, 0.12, 0.25]))
set_prim_property("/World/Target", "primvars:displayColor", [(1, 0.2, 0.2)])

drone = DroneModel(config).build(world)
world.reset()
camera = DroneCamera(prim_path="/World/Drone/Camera", resolution=(640, 480))
camera.initialize()
detector = ColorDetector(camera_matrix=camera.get_intrinsics())
gripper = Gripper().build()

# USD 操控
import omni.usd
from pxr import UsdGeom, Gf
stage = omni.usd.get_context().get_stage()
drone_prim = stage.GetPrimAtPath("/World/Drone")
xf = UsdGeom.Xformable(drone_prim)
translate_op = None
for op in xf.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        translate_op = op
        break

# 目标物体的 Xform (抓取后跟着动)
target_xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Target"))
target_translate = None
for op in target_xf.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        target_translate = op
        break

dt = config["sim"]["physics_dt"]
speed = 0.5
current = np.array(config["drone"]["init_position"], dtype=float)

# 状态机
phase = 0  # 0起飞 1搜索 2接近 3开臂下降 4闭臂 5返航 6完成
target_pos = None
hover_target = np.array([0.0, 0.0, 2.5])
hold = 0
grabbed = False

print("=" * 55)
print("  侧臂抓取演示")
print("  起飞→搜索→接近→开臂下降→闭臂夹取→返航")
print("=" * 55)

frame = 0
while simulation_app.is_running():
    d = np.linalg.norm(hover_target - current)
    if d > 0.005:
        current += (hover_target - current) * min(speed * dt, d) / d

    move = hover_target - current
    roll  = np.clip(-move[1] * 0.5, -0.2, 0.2)
    pitch = np.clip( move[0] * 0.5, -0.2, 0.2)
    if translate_op:
        translate_op.Set(Gf.Vec3d(*current))

    # 夹爪平滑动画
    gripper.update(dt)

    # 抓取后物体跟着无人机走
    if grabbed and target_translate:
        target_translate.Set(Gf.Vec3d(current[0] + 0.05, current[1], 0.2))

    world.step(render=True)
    frame += 1

    # 感知
    if frame % 20 == 0 and target_pos is None:
        rgb = camera.get_rgb()
        if rgb is not None:
            for _, xyz, _ in detector.detect(rgb, drone_pos=current):
                target_pos = xyz
                break

    # 状态机
    if phase == 0 and d < 0.1:
        phase, hold = 1, 60

    elif phase == 1:
        hold -= 1
        if target_pos is not None:
            hover_target[:2] = target_pos[:2]
            phase = 2
            print(f"  [{frame:4d}] 发现目标 ({target_pos[0]:.2f},{target_pos[1]:.2f})")

    elif phase == 2:
        if target_pos is not None:
            hover_target[:2] = target_pos[:2]
        if d < 0.3:
            gripper.open()  # 先张开臂
            hover_target[2] = 0.55
            phase, hold = 3, 60
            print(f"  [{frame:4d}] 开臂, 下降抓取")

    elif phase == 3:
        hold -= 1
        if d < 0.12 and hold < 20:
            gripper.close()  # 夹紧!
            phase, hold = 4, 40
            print(f"  [{frame:4d}] 闭臂夹取!")

    elif phase == 4:
        hold -= 1
        if hold <= 0:
            grabbed = True
            hover_target[:2] = [0, 0]
            hover_target[2] = 2.0
            phase = 5
            print(f"  [{frame:4d}] 携物返航")

    elif phase == 5 and d < 0.1:
        phase = 6
        print(f"  [{frame:4d}] 完成!")

    if frame % 90 == 0:
        names = ["起飞","搜索","接近","开臂下降","闭臂","返航","完成"]
        t = f"({target_pos[0]:.2f},{target_pos[1]:.2f})" if target_pos is not None else "-"
        print(f"  [{frame:4d}] ({current[0]:.2f},{current[1]:.2f},{current[2]:.2f}) "
              f"[{names[phase]}] arm={gripper._current_angle:.0f}°")

simulation_app.close()
