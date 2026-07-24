"""
简单悬停测试 — 验证基础力学
"""
import torch
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from omni.isaac.core import World
from drone_model import DroneModel

cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
with open(cfg_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

world = World(physics_dt=config["sim"]["physics_dt"])
world.scene.add_default_ground_plane()
drone = DroneModel(config).build(world)
world.reset()

# 起始就悬停推力
hover = config["drone"]["mass"] * config["sim"]["gravity"]
per = hover / 4
print(f"恒定推力: {hover:.1f}N total, {per:.1f}N/电机")
print("无人机应该悬停或缓慢上升，不翻倒")

for i in range(300):
    drone.apply_rotor_forces(np.array([per, per, per, per]))
    world.step(render=True)
    if i % 60 == 0:
        s = drone.get_state()
        if s:
            print(f"  z={s['position'][2]:.2f}  roll={np.degrees(s['attitude'][0]):.0f}d  "
                  f"pitch={np.degrees(s['attitude'][1]):.0f}d  angvel_z={s['angular_velocity'][2]:.2f}")

simulation_app.close()
