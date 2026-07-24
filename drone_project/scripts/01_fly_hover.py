"""
全自动飞行 — USD直接设位姿
"""
import torch
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from omni.isaac.core import World
from drone_model import DroneModel
from pxr import UsdGeom, Gf
import omni.usd

cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
with open(cfg_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

world = World(physics_dt=config["sim"]["physics_dt"])
world.scene.add_default_ground_plane()
drone = DroneModel(config).build(world)
world.reset()

# 移除物理 (我们直接控制位姿，不需要物理引擎)
stage = omni.usd.get_context().get_stage()
drone_prim = stage.GetPrimAtPath("/World/Drone")
drone_prim.RemoveAPI(UsdGeom.PhysicsRigidBodyAPI) if hasattr(UsdGeom, 'PhysicsRigidBodyAPI') else None

xform = UsdGeom.Xformable(drone_prim)

dt = config["sim"]["physics_dt"]
speed = 0.8
current = np.array(config["drone"]["init_position"], dtype=float)

waypoints = [
    (np.array([0.0, 0.0, 2.0]), 60,  "1.起飞"),
    (np.array([0.8, 0.0, 2.0]), 60,  "2.右移"),
    (np.array([0.8, 0.8, 2.0]), 60,  "3.前移"),
    (np.array([0.0, 0.8, 2.0]), 60,  "4.左移"),
    (np.array([0.0, 0.0, 2.0]), 60,  "5.回中"),
    (np.array([0.0, 0.0, 0.5]), 999, "6.降落"),
]
wi, target, hold, label = 0, waypoints[0][0].copy(), waypoints[0][1], waypoints[0][2]

print("=" * 50)
print(f"  {label} — 全自动飞行")
print("=" * 50)

# 拿到translate/rotate ops
translate_op = None
rotate_op = None
for op in xform.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        translate_op = op
    elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
        rotate_op = op

frame = 0
while simulation_app.is_running():
    d = np.linalg.norm(target - current)

    if hold > 0:
        hold -= 1
    if hold <= 0 and d < 0.05 and wi < len(waypoints) - 1:
        wi += 1
        target, hold, label = waypoints[wi][0].copy(), waypoints[wi][1], waypoints[wi][2]
        print(f"  [{frame:4d}] >>> {label} ({target[0]:.1f},{target[1]:.1f},{target[2]:.1f})")

    if d > 0.005:
        current += (target - current) * min(speed * dt, d) / d

    # 微倾
    move = target - current
    roll  = np.clip(-move[1] * 0.5, -0.2, 0.2)
    pitch = np.clip( move[0] * 0.5, -0.2, 0.2)

    # 直接用USD设位姿
    if translate_op:
        translate_op.Set(Gf.Vec3d(*current))
    if rotate_op:
        rotate_op.Set(Gf.Vec3f(np.degrees(roll), np.degrees(pitch), 0))

    world.step(render=True)
    frame += 1

    if frame % 90 == 0:
        print(f"  [{frame:4d}] ({current[0]:4.1f},{current[1]:4.1f},{current[2]:4.1f}) [{label}]")

simulation_app.close()
