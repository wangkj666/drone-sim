"""
完整任务链演示 — 栖息待机 → 脱离 → 抓取目标 → 携物返航

    起飞 → 飞到栖枝下方 → 夹爪朝上 → 上升抱杆 → 栖息待机
         → 松开脱离 → 夹爪转回朝下 → 搜索地面目标 → 接近
         → 下降抓取 → 闭臂 → 携物返航

为什么是这个顺序: 机上只有一副夹爪, "朝上抱杆"和"朝下抓物"不能同时进行,
所以两种能力只能按先后串成一条任务链, 而不是并行 —— 这是实机结构决定的。

夹爪的可转角度范围是实测的: verify_gripper.py 扫出 -170°…+40° 之外
臂会扫到机身, 本任务里用到的 -160°(抱杆) 和 -30°…+10°(抓取) 都在安全区内。
"""
import os
import torch
from isaacsim import SimulationApp
_HEADLESS = os.environ.get("DRONE_HEADLESS", "0") == "1"
simulation_app = SimulationApp({"headless": _HEADLESS})

import sys
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

import omni.usd
from pxr import UsdGeom, Gf
stage = omni.usd.get_context().get_stage()

# ── 栖枝: 立柱 + 横杆 (沿Y穿过旋翼之间, 沿X会撞桨叶) ──
PERCH_X, PERCH_Y, PERCH_Z = 0.5, 0.0, 1.30
BAR_R, BAR_HALF = 0.022, 0.45

def make_cyl(path, translate, scale, color, rot_x=0.0):
    """圆柱: op 顺序必须是 [平移, 旋转, 缩放], 否则旋转后形状会被压扁"""
    create_prim(path, "Cylinder")
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate))
    if rot_x:
        xf.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble).Set(float(rot_x))
    xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))
    set_prim_property(path, "primvars:displayColor", [color])

make_cyl("/World/Perch/Pole", (PERCH_X, BAR_HALF, PERCH_Z / 2),
         (0.030, 0.030, PERCH_Z / 2), (0.35, 0.28, 0.18))
make_cyl("/World/Perch/Bar", (PERCH_X, PERCH_Y, PERCH_Z),
         (BAR_R, BAR_R, BAR_HALF), (0.45, 0.36, 0.22), rot_x=90.0)

# ── 地面目标 ──
TARGET_POS = np.array([0.30, 0.0, 0.07])
CAMERA_OFFSET = 0.15   # 相机装在机心下方 0.15m
TARGET_TOP = 0.14      # 目标顶面离地高度 (修正测距用)
create_prim("/World/Target", "Cube", position=TARGET_POS,
            scale=np.array([0.05, 0.05, 0.07]))   # 0.10×0.10×0.14, 窄于闭合爪口
set_prim_property("/World/Target", "primvars:displayColor", [(1, 0.1, 0.1)])
target_translate = None
for op in UsdGeom.Xformable(stage.GetPrimAtPath("/World/Target")).GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        target_translate = op
        break

# ── 无人机 + 相机 + 夹爪 ──
drone = DroneModel(config).build(world, kinematic=True)
world.reset()
camera = DroneCamera(prim_path="/World/Drone/Camera", resolution=(640, 480))
camera.initialize()
detector = ColorDetector(camera_matrix=camera.get_intrinsics())
gripper = Gripper().build()

# 位姿 ops: 平移 + 机身倾角。rotateXYZ 不存在就补一个 (追加到 op 列表末尾,
# 即"先转再平移", 顺序正确)
drone_prim = stage.GetPrimAtPath("/World/Drone")
xf = UsdGeom.Xformable(drone_prim)
translate_op = rotate_op = None
for op in xf.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        translate_op = op
    elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
        rotate_op = op
if rotate_op is None:
    rotate_op = xf.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat)

dt = config["sim"]["physics_dt"]
SPEED = 0.5
GRIPPER_Y = 0.0                     # 夹爪在机身正中央, 目标停在正下方即可
PERCH_DRONE_Z = PERCH_Z - 0.075     # 合拢 -160° 时臂面正好贴住杆面
GRASP_Z = 0.15                      # 爪口咬合高度(实测爪口在机心下方 0.078)
CRUISE_Z = 1.50

current = np.array(config["drone"]["init_position"], dtype=float)

NAMES = ["起飞", "飞向栖枝", "夹爪朝上", "上升入爪", "闭臂", "栖息中", "脱离下降",
         "夹爪转朝下", "搜索目标", "接近目标", "下降抓取", "闭臂夹取", "携物返航", "完成"]
# 需要精细对准的阶段: 期间把机身倾角收平, 否则夹爪会横向偏出几厘米
LEVEL_PHASES = {2, 3, 4, 5, 8, 9, 10, 11}

phase, hold = 0, 0
hover_target = np.array([0.0, 0.0, 0.80])
target_pos = None
grabbed = False
grab_offset = None
tilt_amt = 0.0

print("=" * 60)
print("  完整任务链: 栖息待机 → 脱离 → 抓取目标 → 携物返航")
print("=" * 60)

frame = 0
while simulation_app.is_running():
    d = np.linalg.norm(hover_target - current)
    if d > 0.004:
        current += (hover_target - current) * min(SPEED * dt, d) / d

    # 机身随运动方向微倾; 精细对准阶段平滑收平
    move = hover_target - current
    tilt_amt += float(np.clip((0.0 if phase in LEVEL_PHASES else 1.0) - tilt_amt,
                              -2.0 * dt, 2.0 * dt))
    roll = np.clip(-move[1] * 0.5, -0.2, 0.2) * tilt_amt
    pitch = np.clip(move[0] * 0.5, -0.2, 0.2) * tilt_amt

    translate_op.Set(Gf.Vec3d(*current))
    rotate_op.Set(Gf.Vec3f(np.degrees(roll), np.degrees(pitch), 0.0))

    gripper.update(dt)

    # 抓取后物体保持"抓取瞬间"的相对位置, 随无人机一起移动
    if grabbed and target_translate is not None and grab_offset is not None:
        p = current + grab_offset
        target_translate.Set(Gf.Vec3d(p[0], p[1], p[2]))

    world.step(render=True)
    frame += 1

    # 感知: 只在搜索阶段开, 且此时机身已收平, 相机正对地面
    if phase == 8 and frame % 20 == 0 and target_pos is None:
        rgb = camera.get_rgb()
        if rgb is not None:
            z_eff = current[2] - CAMERA_OFFSET - TARGET_TOP
            seen = np.array([current[0], current[1], z_eff])
            for _, xyz, _ in detector.detect(rgb, drone_pos=seen):
                target_pos = xyz
                break

    # ── 状态机 ──
    if phase == 0:                                    # 起飞
        if d < 0.08:
            hover_target[:] = [PERCH_X, PERCH_Y, 0.80]
            phase = 1
            print(f"  [{frame:4d}] 起飞完成 → 飞向栖枝下方")

    elif phase == 1:                                  # 飞到栖枝下方
        if d < 0.10:
            gripper.point_up()
            gripper.open()
            phase, hold = 2, 90
            print(f"  [{frame:4d}] 到达栖枝下方, 夹爪翻转朝上")

    elif phase == 2:                                  # 等翻转动画走完
        hold -= 1
        if hold <= 0:
            hover_target[2] = PERCH_DRONE_Z
            phase = 3
            print(f"  [{frame:4d}] 缓慢上升, 让横杆进入爪口")

    elif phase == 3:                                  # 上升入爪口
        if d < 0.02:
            gripper.close()
            phase, hold = 4, 45
            print(f"  [{frame:4d}] 横杆到位, 闭臂夹住!")

    elif phase == 4:
        hold -= 1
        if hold <= 0:
            phase, hold = 5, 240                      # 栖息待机 4 秒
            print(f"  [{frame:4d}] ★ 栖息中 (夹爪抱住横杆, 位置锁定)")

    elif phase == 5:                                  # 栖息保持
        hold -= 1
        if hold <= 0:
            gripper.open()
            hover_target[2] = 0.80
            phase, hold = 6, 60
            print(f"  [{frame:4d}] 松开, 脱离横杆")

    elif phase == 6:                                  # 下降脱离
        hold -= 1
        if d < 0.08 and hold < 30:
            gripper.point_down()                      # 转回朝下, 准备抓取
            gripper.open()
            phase, hold = 7, 120                      # -160°→-30° 约需 1.4s
            print(f"  [{frame:4d}] 脱离完成, 夹爪转回朝下")

    elif phase == 7:                                  # 等翻转动画走完
        hold -= 1
        if hold <= 0:
            hover_target[:] = [0.0, 0.0, CRUISE_Z]
            phase, hold = 8, 60
            print(f"  [{frame:4d}] 飞向目标区域, 开始搜索")

    elif phase == 8:                                  # 搜索目标
        hold -= 1
        if target_pos is not None:
            hover_target[:2] = target_pos[:2] + np.array([0.0, GRIPPER_Y])
            phase = 9
            print(f"  [{frame:4d}] 发现目标 ({target_pos[0]:.2f},{target_pos[1]:.2f}), 飞至正上方")

    elif phase == 9:                                  # 接近目标正上方
        if target_pos is not None:
            hover_target[:2] = target_pos[:2] + np.array([0.0, GRIPPER_Y])
        if d < 0.30:
            gripper.open()
            hover_target[2] = GRASP_Z
            phase, hold = 10, 60
            print(f"  [{frame:4d}] 开臂, 下降抓取")

    elif phase == 10:                                 # 下降抓取
        hold -= 1
        if d < 0.12 and hold < 20:
            gripper.close()
            phase, hold = 11, 40
            print(f"  [{frame:4d}] 闭臂夹取!")

    elif phase == 11:                                 # 等夹紧
        hold -= 1
        if hold <= 0:
            grabbed = True
            if target_translate is not None:
                op = target_translate.Get()
                grab_offset = np.array([op[0] - current[0],
                                        op[1] - current[1],
                                        op[2] - current[2]])
                print(f"  [{frame:4d}] 抓取偏移量记录: {np.round(grab_offset, 3)}")
            hover_target[:] = [0.0, 0.0, CRUISE_Z]
            phase = 12
            print(f"  [{frame:4d}] 携物返航")

    elif phase == 12 and d < 0.10:
        phase, hold = 13, 180       # 停 3 秒展示最终状态, 然后正常退出
        print(f"  [{frame:4d}] 完成! 栖息 + 抓取全任务链成功")

    elif phase == 13:
        hold -= 1
        if hold <= 0:
            break                    # 退出循环, 让 SimulationApp 正常收尾

    if frame % 120 == 0:
        st = drone.get_state()
        act_z = f"{st['position'][2]:.2f}" if st else "--"
        obj_z = "--"
        if grabbed and target_translate is not None:
            ot = target_translate.Get()
            if ot is not None:
                obj_z = f"{ot[2]:.2f}"
        tp = f"({target_pos[0]:.2f},{target_pos[1]:.2f})" if target_pos is not None else "-"
        print(f"  [{frame:5d}] pos=({current[0]:+.2f},{current[1]:+.2f},{current[2]:.2f}) "
              f"drone_z={act_z} obj_z={obj_z} 目标={tp} [{NAMES[phase]}]")

simulation_app.close()
