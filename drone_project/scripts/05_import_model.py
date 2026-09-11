"""
加载 F570 低面版无人机 — 重试
"""
import torch
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from omni.isaac.core.utils.prims import create_prim, set_prim_property
import numpy as np
import time

stage = omni.usd.get_context().get_stage()

create_prim("/World/Ground", "Plane", scale=np.array([10, 10, 1]))
set_prim_property("/World/Ground", "primvars:displayColor", [(0.3, 0.3, 0.3)])

# 直接引用低面版 GLB
prim = stage.DefinePrim("/World/F570", "Xform")
prim.GetReferences().AddReference("G:/isaac/f570_low.glb")

print("F570 低面模型加载中 (5.4MB)...")
print("=" * 50)
print("  滚轮缩放找模型")
print("=" * 50)

while simulation_app.is_running():
    time.sleep(0.1)
simulation_app.close()
