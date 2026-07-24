"""
Isaac Sim 无人机 Demo — 按 R 键重新掉落，按 ESC 退出
"""
import torch
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim, set_prim_property
from omni.isaac.core.prims import RigidPrim
import omni.timeline
import omni.kit.commands
from pxr import Usd, UsdGeom

# 时间轴控制
timeline = omni.timeline.get_timeline_interface()

def build_drone(world):
    """搭建无人机所有零件"""
    parts = []

    # 机身
    create_prim("/World/Drone/Body", "Cube",
                position=np.array([0, 0, 1.5]),
                scale=np.array([0.2, 0.15, 0.06]))
    set_prim_property("/World/Drone/Body", "primvars:displayColor", [(0.2, 0.2, 0.3)])
    parts.append("/World/Drone/Body")

    # 机臂
    arm_dirs = [(0.15, 0, 0), (-0.15, 0, 0), (0, 0.15, 0), (0, -0.15, 0)]
    for i, (dx, dy, dz) in enumerate(arm_dirs):
        create_prim(f"/World/Drone/Arm{i}", "Cylinder",
                    position=np.array([dx, dy, 1.5]),
                    scale=np.array([0.02, 0.02, 0.25]))
        parts.append(f"/World/Drone/Arm{i}")

    # 电机 + 螺旋桨
    motor_xy = [(0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25)]
    for i, (x, y) in enumerate(motor_xy):
        create_prim(f"/World/Drone/Motor{i}", "Cylinder",
                    position=np.array([x, y, 1.5]),
                    scale=np.array([0.04, 0.04, 0.03]))
        set_prim_property(f"/World/Drone/Motor{i}", "primvars:displayColor", [(0.1, 0.1, 0.1)])
        parts.append(f"/World/Drone/Motor{i}")

        create_prim(f"/World/Drone/Prop{i}", "Cube",
                    position=np.array([x, y, 1.53]),
                    scale=np.array([0.12, 0.02, 0.005]))
        set_prim_property(f"/World/Drone/Prop{i}", "primvars:displayColor", [(0.9, 0.9, 0.9)])
        parts.append(f"/World/Drone/Prop{i}")

    # 起落架
    leg_xy = [(0.1, 0.08), (-0.1, 0.08), (0.1, -0.08), (-0.1, -0.08)]
    for i, (x, y) in enumerate(leg_xy):
        create_prim(f"/World/Drone/Leg{i}", "Cylinder",
                    position=np.array([x, y, 1.42]),
                    scale=np.array([0.005, 0.005, 0.06]))
        set_prim_property(f"/World/Drone/Leg{i}", "primvars:displayColor", [(0.15, 0.15, 0.15)])
        parts.append(f"/World/Drone/Leg{i}")

    # 注册为刚体
    for i, path in enumerate(parts):
        rp = RigidPrim(prim_path=path, name=f"part_{i}", mass=0.1)
        world.scene.add(rp)

    return parts


# 1. 创建世界
world = World()
world.scene.add_default_ground_plane()

# 2. 搭无人机 + 障碍物
parts = build_drone(world)

colors = [(0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8), (0.8, 0.8, 0.2)]
cube_pos = [(0.8, 0, 0.3), (-0.6, 0.4, 0.3), (0, 0.8, 0.3), (-0.5, -0.5, 0.3)]
for i, ((x, y, z), c) in enumerate(zip(cube_pos, colors)):
    create_prim(f"/World/Cube{i}", "Cube", position=np.array([x, y, z]), scale=np.array([0.2, 0.2, 0.2]))
    set_prim_property(f"/World/Cube{i}", "primvars:displayColor", [c])

# 3. 初始重置
timeline.stop()
world.reset()
timeline.play()

print("=" * 50)
print("  R 键 = 重新掉落（回到初始位置再掉一次）")
print("  ESC = 退出")
print("=" * 50)

# 4. 主循环
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
print("Done!")
