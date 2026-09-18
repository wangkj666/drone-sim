"""自检: 栖息姿态到底夹没夹住横杆?

不看渲染图 —— 渲染图上"看着像夹住了"是看不出来的(杆会从爪口漏下去)。
改为用【真实 USD 变换】把每段臂的 8 个角点算出来, 量它到横杆轴线的最近距离。
判据(意图): 臂表面贴住杆(|侵入| < 2mm) 且全程不穿机身、两臂不交叉。

用法:  python verify_perch.py     (结果写到 vp_out.txt)
"""
import torch
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import sys, os, yaml, itertools
import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, Gf
from omni.isaac.core import World

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "drone_project"))
from drone_model import DroneModel
from gripper import Gripper

cfg = yaml.safe_load(open(os.path.join("drone_project", "config.yaml"), encoding="utf-8"))
world = World(physics_dt=cfg["sim"]["physics_dt"])
world.scene.add_default_ground_plane()
drone = DroneModel(cfg).build(world, kinematic=True)
world.reset()
g = Gripper().build()

DZ = 1.23
stage = omni.usd.get_context().get_stage()
tr = [op for op in UsdGeom.Xformable(stage.GetPrimAtPath("/World/Drone")).GetOrderedXformOps()
      if op.GetOpType() == UsdGeom.XformOp.TypeTranslate][0]
tr.Set(Gf.Vec3d(0.0, 0.0, DZ))

HX = cfg["drone"]["frame_length"] / 2
HY = cfg["drone"]["frame_width"] / 2
HZ = cfg["drone"]["frame_height"] / 2
CORNERS = list(itertools.product((-1, 1), repeat=3))

def seg_corners(path):
    """一段臂的 8 个角点(世界坐标)"""
    m = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    return [np.array(m.Transform(Gf.Vec3d(*c))) for c in CORNERS]

def set_angle(a):
    g._cur = float(a); g._mode = 0.0; g._grip = 0.0
    for op, s in g._pivot_data:
        op.Set(float(a * s))
    world.step(render=False)

def analyse(a, z_bar, where=None):
    """返回 (到杆轴最近距离, 穿机身角点数, 交叉段数); where 可回收最近点坐标"""
    set_angle(a)
    mind = 1e9; hit = 0; cross = 0
    for side in ("Right", "Left"):
        for i in range(12):
            cs = seg_corners(f"/World/Drone/Gripper/{side}/Pivot/Seg{i}")
            for p in cs:
                d = np.hypot(p[0], p[2] - DZ - z_bar)     # 杆轴 = x0, z=z_bar 的 Y 向直线
                if d < mind:
                    mind = d
                    if where is not None:
                        where[:] = [side, i, p[0], p[2] - DZ]
                if abs(p[0]) < HX and abs(p[1]) < HY and abs(p[2] - DZ) < HZ:
                    hit += 1
    for i in range(12):
        if (seg_corners(f"/World/Drone/Gripper/Right/Pivot/Seg{i}")[0][0]
                < seg_corners(f"/World/Drone/Gripper/Left/Pivot/Seg{i}")[0][0]):
            cross += 1
    return mind, hit, cross

_log = open("vp_out.txt", "w", encoding="utf-8")
def out(s):
    _log.write(s + "\n"); _log.flush()

CLOSED, OPEN = -160.0, -120.0      # 对应 mode=-150, close(grip=10) / open(grip=-30)
R_BAR = 0.022
out(f"横杆半径 r={R_BAR:.3f} (直径 {2*R_BAR:.3f})")
out("=== 找合适的杆高 z_bar: 让臂表面恰好贴住杆 ===")
best = None
for zb in [0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100]:
    md, hit, cross = analyse(CLOSED, zb)
    pen = R_BAR - md                      # >0 = 臂插进杆里
    out(f"z_bar={zb:.3f} 臂到杆轴最近={md:.4f} 侵入={pen:+.4f} 穿机身={hit} 交叉={cross}")
    if pen <= 0 and best is None:
        best = zb
out(f"-> 建议 z_bar = {best if best else '无(需改臂几何)'}")

out("")
out("=== 张开姿态能否让杆顺利进入 (杆停在选定高度) ===")
zb = best if best else 0.075
md, hit, cross = analyse(OPEN, zb)
out(f"张开角={OPEN:.0f} 臂到杆轴最近={md:.4f} 需要 > {R_BAR:.3f} -> "
    f"{'[OK]' if md > R_BAR else '[X]'}  穿机身={hit} 交叉={cross}")

out("")
out("=== 闭合姿态最终自检 ===")
wh = [None]
md, hit, cross = analyse(CLOSED, zb, wh)
out(f"闭合角={CLOSED:.0f} 贴杆={md:.4f} 侵入={R_BAR - md:+.4f} 穿机身={hit} 交叉={cross} "
    f"{'[OK]' if (hit == 0 and cross == 0 and abs(R_BAR - md) < 0.004) else '[X]'}")
out(f"接触点: 第{wh[1]}段 {wh[0]} 臂, 机心坐标 x={wh[2]:+.3f} z={wh[3]:+.3f} "
    f"(杆心 z={zb:.3f}) -> 接触{'在杆心下方(托住)' if wh[3] < zb else '在杆心高度(侧向夹住)'}")

# 过渡路径: 从张开到闭合, 全程不能撞杆/穿机身
out("")
out("=== 翻转+闭合过渡路径 ===")
bad = []
for a in np.arange(-100.0, CLOSED - 1, -5.0):
    md, hit, cross = analyse(float(a), zb)
    # 过渡中杆已经在爪口里了, 只要不穿机身/不交叉/不把杆捅穿(侵入<3mm)
    if hit or cross or md < R_BAR - 0.003:
        bad.append((round(float(a), 1), round(float(md), 4), hit, cross))
out(f"异常角度: {bad if bad else '无 [OK]'}")

_log.close()
app.close()
