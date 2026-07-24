"""
Isaac Sim 入门脚本 - 打开一个带物理仿真的 3D 场景
使用方法:
    1. 打开终端
    2. source isaacsim_env/Scripts/activate
    3. export OMNI_KIT_ACCEPT_EULA=YES
    4. python hello_isaac.py
"""

# ============================================================
# 第1步: 必须先导入 torch！（否则 DLL 加载会失败）
# ============================================================
import torch
print(f"[1/5] PyTorch {torch.__version__} loaded, CUDA: {torch.cuda.is_available()}")

# ============================================================
# 第2步: 启动 Isaac Sim 仿真引擎
# ============================================================
from isaacsim import SimulationApp

# headless=False 表示显示 GUI 窗口（能看到 3D 画面）
# headless=True  表示后台运行（无窗口，适合服务器/训练）
simulation_app = SimulationApp({"headless": False})
print("[2/5] SimulationApp started")

# ============================================================
# 第3步: 创建仿真世界
# ============================================================
from omni.isaac.core import World

world = World()
print("[3/5] World created")

# ============================================================
# 第4步: 加载一个地面，让场景不是空的
# ============================================================
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import GeometryPrim

# 从 Nucleus 服务器加载一个地面
world.scene.add_default_ground_plane()
print("[4/5] Ground plane added")

# ============================================================
# 第5步: 运行仿真循环
# ============================================================
print("[5/5] Simulation running! Close the window to exit.")
print("      Tip: 用鼠标中键旋转视角，右键平移，滚轮缩放")

# 重置世界
world.reset()

# 仿真主循环：每帧步进一次物理
while simulation_app.is_running():
    world.step(render=True)  # render=True 表示刷新画面

# 关闭
simulation_app.close()
print("Simulation closed. Bye!")
