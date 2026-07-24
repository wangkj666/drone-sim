"""
相机测试 — 无人机飞行 + 实时下视图
"""
import torch
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from omni.isaac.core import World
from scipy.spatial.transform import Rotation as R

from drone_model import DroneModel
from camera import DroneCamera

# ── 配置 ──
cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
with open(cfg_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

world = World(physics_dt=config["sim"]["physics_dt"])
world.scene.add_default_ground_plane()

# 放几个彩色方块做目标
from omni.isaac.core.utils.prims import create_prim, set_prim_property
colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0)]
for i, c in enumerate(colors):
    x, y = i * 0.3 - 0.45, 0.0
    create_prim(f"/World/Target{i}", "Cube",
                position=np.array([x, y, 0.15]),
                scale=np.array([0.15, 0.15, 0.15]))
    set_prim_property(f"/World/Target{i}", "primvars:displayColor", [c])

# ── 无人机 + 相机 ──
drone = DroneModel(config).build(world)
world.reset()

# 在下视位置装相机 (底面下方 10cm)
camera = DroneCamera(prim_path="/World/Drone/Camera",
                     resolution=(640, 480),
                     frequency=30)
camera.initialize()

import omni.usd
from pxr import UsdGeom, Gf
stage = omni.usd.get_context().get_stage()

# ── 飞控 ──
dt = config["sim"]["physics_dt"]
speed = 0.6
current = np.array(config["drone"]["init_position"], dtype=float)

waypoints = [
    (np.array([0.0, 0.0, 2.5]), 60, "起飞"),
    (np.array([0.0, 0.5, 2.5]), 60, "前移"),
    (np.array([0.0, 0.0, 2.5]), 60, "回中"),
    (np.array([0.0, 0.0, 0.6]), 999, "降落"),
]
wi, target, hold, label = 0, waypoints[0][0].copy(), waypoints[0][1], waypoints[0][2]

# 获取USD transform ops
drone_prim = stage.GetPrimAtPath("/World/Drone")
xf = UsdGeom.Xformable(drone_prim)
translate_op = rotate_op = None
for op in xf.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        translate_op = op
    elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
        rotate_op = op

print("=" * 50)
print("  相机测试 — 飞行 + 下视图")
print("  按 Q 退出")
print("=" * 50)

frame = 0
while simulation_app.is_running():
    # ── 飞行逻辑 ──
    d = np.linalg.norm(target - current)
    if hold > 0:
        hold -= 1
    if hold <= 0 and d < 0.05 and wi < len(waypoints) - 1:
        wi += 1
        target, hold, label = waypoints[wi][0].copy(), waypoints[wi][1], waypoints[wi][2]

    if d > 0.005:
        current += (target - current) * min(speed * dt, d) / d

    move = target - current
    roll  = np.clip(-move[1] * 0.5, -0.2, 0.2)
    pitch = np.clip( move[0] * 0.5, -0.2, 0.2)

    if translate_op:
        translate_op.Set(Gf.Vec3d(*current))
    if rotate_op:
        rotate_op.Set(Gf.Vec3f(np.degrees(roll), np.degrees(pitch), 0))

    world.step(render=True)

    # ── 保存相机图像（每60帧一张） ──
    if frame > 10 and frame % 60 == 0:
        rgb = camera.get_rgb()
        if rgb is not None and rgb.size > 0:
            import cv2
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
            cv2.putText(bgr, f"z={current[2]:.1f}m [{label}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            save_dir = "G:/isaac/drone_project/cam_frames"
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(f"{save_dir}/frame_{frame:04d}.png", bgr)
            print(f"  [frame {frame}] saved camera image ({save_dir}/frame_{frame:04d}.png)")

    frame += 1
    if frame % 120 == 0:
        print(f"  [{frame:4d}] ({current[0]:4.1f},{current[1]:4.1f},{current[2]:4.1f}) [{label}]")

simulation_app.close()
