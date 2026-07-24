"""
感知测试 — 颜色检测 + 3D定位
"""
import torch
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
import cv2
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

# ── 放彩色方块（红绿蓝黄，散布开来） ──
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
pos_list = [(-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4), (0.4, 0.4)]
for i, (c, (px, py)) in enumerate(zip(colors, pos_list)):
    create_prim(f"/World/Target{i}", "Cube",
                position=np.array([px, py, 0.15]),
                scale=np.array([0.2, 0.2, 0.15]))
    set_prim_property(f"/World/Target{i}", "primvars:displayColor", [c])
print("Placed 4 colored blocks: red, green, blue, yellow")

# ── 无人机 + 相机 ──
drone = DroneModel(config).build(world)
world.reset()
camera = DroneCamera(prim_path="/World/Drone/Camera", resolution=(640, 480))
camera.initialize()

# ── 飞行控制 ──
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
speed = 0.6
current = np.array(config["drone"]["init_position"], dtype=float)

# 飞行路径
waypoints = [
    (np.array([0.0, 0.0, 2.5]), 60,  "起飞"),
    (np.array([0.0, 0.0, 1.8]), 60,  "下降观察"),
    (np.array([0.3, 0.0, 1.8]), 60,  "右移"),
    (np.array([0.3, 0.3, 1.8]), 60,  "前移"),
    (np.array([0.0, 0.3, 1.8]), 60,  "左移"),
    (np.array([0.0, 0.0, 1.8]), 60,  "回中"),
    (np.array([0.0, 0.0, 0.5]), 999, "降落"),
]
wi, target, hold, label = 0, waypoints[0][0].copy(), waypoints[0][1], waypoints[0][2]

detector = ColorDetector(camera_height=1.8, camera_matrix=camera.get_intrinsics())

print("=" * 55)
print("  颜色检测感知测试")
print("  检测红绿蓝黄方块 → 显示3D位置")
print("=" * 55)

frame = 0
while simulation_app.is_running():
    d = np.linalg.norm(target - current)
    if hold > 0:
        hold -= 1
    if hold <= 0 and d < 0.05 and wi < len(waypoints) - 1:
        wi += 1
        target, hold, label = waypoints[wi][0].copy(), waypoints[wi][1], waypoints[wi][2]
        print(f"  >>> {label} ({target[0]:.1f},{target[1]:.1f},{target[2]:.1f})")

    if d > 0.005:
        current += (target - current) * min(speed * dt, d) / d

    move = target - current
    roll = np.clip(-move[1] * 0.5, -0.2, 0.2)
    pitch = np.clip(move[0] * 0.5, -0.2, 0.2)
    if translate_op:
        translate_op.Set(Gf.Vec3d(*current))
    if rotate_op:
        rotate_op.Set(Gf.Vec3f(np.degrees(roll), np.degrees(pitch), 0))

    world.step(render=True)

    # ── 颜色检测 (每15帧) ──
    if frame > 10 and frame % 15 == 0:
        rgb = camera.get_rgb()
        if rgb is not None and rgb.size > 0:
            detections = detector.detect(rgb, drone_pos=current)
            if detections:
                names = [d[0] for d in detections]
                pos_3d = [f"({d[1][0]:.2f},{d[1][1]:.2f})" for d in detections]
                print(f"  [frame {frame}] 检测到: {list(zip(names, pos_3d))}")

            # 保存检测结果图
            vis = detector.draw(rgb, detections)
            save_dir = "G:/isaac/drone_project/cam_frames"
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(f"{save_dir}/detect_{frame:04d}.png",
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    frame += 1

cv2.destroyAllWindows()
simulation_app.close()
