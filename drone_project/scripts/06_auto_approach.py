"""
自主目标接近 — 感知+飞行整合
看到彩色方块 → 自动飞过去 → 悬停在上方
"""
import torch
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
import cv2
import time
from scipy.spatial.transform import Rotation as R
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim, set_prim_property

from drone_model import DroneModel
from camera import DroneCamera
from perception import ColorDetector

# ── 配置 ──
cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
with open(cfg_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

world = World(physics_dt=config["sim"]["physics_dt"])
world.scene.add_default_ground_plane()

# 目标方块
colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0)]
for i, c in enumerate(colors):
    x, y = i * 0.3 - 0.45, 0.0
    create_prim(f"/World/Target{i}", "Cube",
                position=np.array([x, y, 0.2]), scale=np.array([0.2, 0.2, 0.2]))
    set_prim_property(f"/World/Target{i}", "primvars:displayColor", [c])

# 无人机 + 相机 + 感知
drone = DroneModel(config).build(world)
world.reset()
camera = DroneCamera(prim_path="/World/Drone/Camera", resolution=(640, 480))
camera.initialize()
detector = ColorDetector(camera_matrix=camera.get_intrinsics())

# ── USD 操控 ──
import omni.usd
from pxr import UsdGeom, Gf
stage = omni.usd.get_context().get_stage()
drone_prim = stage.GetPrimAtPath("/World/Drone")
xf = UsdGeom.Xformable(drone_prim)
translate_op = rotate_op = None
for op in xf.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        translate_op = op
    elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
        rotate_op = op

dt = config["sim"]["physics_dt"]
speed = 0.5
current = np.array(config["drone"]["init_position"], dtype=float)

# ── 任务状态机 ──
# phase 0: 起飞到搜索高度
# phase 1: 搜索目标 (原地旋转或悬停)
# phase 2: 飞向目标
# phase 3: 下降接近
# phase 4: 完成
phase = 0
target_pos = None           # 检测到的目标世界坐标
search_height = 2.5
approach_xy_offset = np.array([0.0, 0.0])  # 目标上方悬停
approach_z = 0.6                          # 下降接近高度

hover_target = np.array([0.0, 0.0, search_height])
hold_counter = 0

print("=" * 55)
print("  自主目标接近")
print("  阶段: 起飞→搜索→接近→下降")
print("=" * 55)

frame = 0
while simulation_app.is_running():
    # ── 飞控：平滑飞到 current_target ──
    d = np.linalg.norm(hover_target - current)
    if d > 0.005:
        current += (hover_target - current) * min(speed * dt, d) / d

    move = hover_target - current
    roll  = np.clip(-move[1] * 0.5, -0.2, 0.2)
    pitch = np.clip( move[0] * 0.5, -0.2, 0.2)
    if translate_op:
        translate_op.Set(Gf.Vec3d(*current))
    if rotate_op:
        rotate_op.Set(Gf.Vec3f(np.degrees(roll), np.degrees(pitch), 0))

    world.step(render=True)
    frame += 1

    # ── 感知：每30帧检测一次 ──
    detected = None
    if frame % 30 == 0:
        rgb = camera.get_rgb()
        if rgb is not None and rgb.size > 0:
            detections = detector.detect(rgb, drone_pos=current)
            for name, xyz, _ in detections:
                detected = xyz  # 取第一个检测到的目标
                break

    # ── 状态切换 ──
    if phase == 0:
        # 起飞到搜索高度
        if d < 0.1:
            phase = 1
            hold_counter = 60  # 先悬停1秒
            print(f"  [frame {frame}] 到达搜索高度 {search_height}m")

    elif phase == 1:
        # 搜索目标
        hold_counter -= 1
        if detected is not None:
            target_pos = detected
            hover_target = np.array([target_pos[0], target_pos[1], search_height])
            phase = 2
            print(f"  [frame {frame}] 发现目标! pos=({target_pos[0]:.2f},{target_pos[1]:.2f}) 飞过去...")

    elif phase == 2:
        # 飞向目标上方
        if target_pos is not None:
            hover_target[:2] = target_pos[:2]
        if d < 0.15:
            hover_target[2] = approach_z
            phase = 3
            print(f"  [frame {frame}] 到达目标上方, 下降至 {approach_z}m")

    elif phase == 3:
        # 下降接近
        if d < 0.1:
            phase = 4
            print(f"  [frame {frame}] 悬停在目标上方 z={current[2]:.2f}m")

    # ── 打印状态 ──
    if frame % 60 == 0:
        status = ["起飞","搜索","接近","下降","完成"][phase]
        tinfo = f"target=({target_pos[0]:.2f},{target_pos[1]:.2f})" if target_pos is not None else "搜索中"
        print(f"  [{frame:4d}] pos=({current[0]:.2f},{current[1]:.2f},{current[2]:.2f}) "
              f"[{status}] {tinfo}")

simulation_app.close()
