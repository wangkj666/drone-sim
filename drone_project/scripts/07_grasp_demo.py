"""
侧装旋转臂抓取演示
起飞→搜索→接近→开臂→下降→闭臂夹取→携物返航
"""
import torch
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys, os
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

# USD 操控 (先拿到 stage)
import omni.usd
from pxr import UsdGeom, Gf
stage = omni.usd.get_context().get_stage()

# ── 目标方块: 尺寸匹配夹爪闭合间距(0.125m), 保证能夹住 ──
TARGET_POS = np.array([0.30, 0.0, 0.07])  # 0.07 = 半高, 底面刚好贴地
CAMERA_OFFSET = 0.15   # 相机装在机心下方 0.15m
TARGET_TOP = 0.14      # 目标顶面离地高度 (用于修正测距)
GRIPPER_X = 0.0        # 弯齿条左右对称, 目标停在机身正下方即可
create_prim("/World/Target", "Cube",
            position=TARGET_POS,
            scale=np.array([0.05, 0.05, 0.07]))   # 0.10×0.10×0.14, 宽<0.125
set_prim_property("/World/Target", "primvars:displayColor", [(1, 0.1, 0.1)])

# 复用 Cube 自带的 translate op 移动目标
target_xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Target"))
target_translate = None
for op in target_xf.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        target_translate = op
        break

drone = DroneModel(config).build(world, kinematic=True)
world.reset()
camera = DroneCamera(prim_path="/World/Drone/Camera", resolution=(640, 480))
camera.initialize()
detector = ColorDetector(camera_matrix=camera.get_intrinsics())
gripper = Gripper().build()

# 无人机 translate op
drone_prim = stage.GetPrimAtPath("/World/Drone")
xf = UsdGeom.Xformable(drone_prim)
translate_op = None
for op in xf.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        translate_op = op
        break

dt = config["sim"]["physics_dt"]
speed = 0.5
current = np.array(config["drone"]["init_position"], dtype=float)

# 状态机
phase = 0  # 0起飞 1搜索 2接近 3开臂下降 4闭臂 5返航 6完成
target_pos = None
hover_target = np.array([0.0, 0.0, 1.5])
hold = 0
grabbed = False
grab_offset = None    # 抓取瞬间物体相对无人机的偏移

print("=" * 55)
print("  侧臂抓取演示")
print("  起飞→搜索→接近→开臂下降→闭臂夹取→返航")
print("=" * 55)

frame = 0
while simulation_app.is_running():
    d = np.linalg.norm(hover_target - current)
    if d > 0.005:
        current += (hover_target - current) * min(speed * dt, d) / d

    move = hover_target - current
    roll  = np.clip(-move[1] * 0.5, -0.2, 0.2)
    pitch = np.clip( move[0] * 0.5, -0.2, 0.2)
    if translate_op:
        translate_op.Set(Gf.Vec3d(*current))

    # 夹爪平滑动画
    gripper.update(dt)

    # 抓取后物体保持"抓取瞬间"与无人机的相对位置, 随无人机一起移动
    if grabbed and target_translate is not None and grab_offset is not None:
        p = current + grab_offset
        target_translate.Set(Gf.Vec3d(p[0], p[1], p[2]))

    world.step(render=True)
    frame += 1

    # 感知 (距离要扣掉相机在机心下方的偏移和目标高度, 否则算出的坐标会偏大)
    if frame % 20 == 0 and target_pos is None:
        rgb = camera.get_rgb()
        if rgb is not None:
            z_eff = current[2] - CAMERA_OFFSET - TARGET_TOP
            seen = np.array([current[0], current[1], z_eff])
            for _, xyz, _ in detector.detect(rgb, drone_pos=seen):
                target_pos = xyz
                break

    # 状态机
    if phase == 0 and d < 0.1:
        phase, hold = 1, 60

    elif phase == 1:
        hold -= 1
        if target_pos is not None:
            # 夹爪在机身右侧, 所以无人机要停在目标左侧, 让目标落在两臂之间
            hover_target[:2] = target_pos[:2] + np.array([-GRIPPER_X, 0.0])
            phase = 2
            print(f"  [{frame:4d}] 发现目标 ({target_pos[0]:.2f},{target_pos[1]:.2f}), 飞至夹爪对准")

    elif phase == 2:
        if target_pos is not None:
            hover_target[:2] = target_pos[:2] + np.array([-GRIPPER_X, 0.0])
        if d < 0.3:
            gripper.open()  # 先张开臂
            hover_target[2] = 0.26   # 降到弯齿条能包住目标的高度
            phase, hold = 3, 60
            print(f"  [{frame:4d}] 开臂, 下降抓取")

    elif phase == 3:
        hold -= 1
        if d < 0.12 and hold < 20:
            gripper.close()  # 夹紧!
            phase, hold = 4, 40
            print(f"  [{frame:4d}] 闭臂夹取!")

    elif phase == 4:
        hold -= 1
        if hold <= 0:
            grabbed = True
            # 记录抓取瞬间物体相对无人机的位置, 之后严格保持
            if target_translate is not None:
                op = target_translate.Get()
                grab_offset = np.array([op[0] - current[0],
                                        op[1] - current[1],
                                        op[2] - current[2]])
                print(f"  [{frame:4d}] 抓取偏移量记录: {np.round(grab_offset, 3)}")
            hover_target[:2] = [0, 0]
            hover_target[2] = 1.5
            phase = 5
            print(f"  [{frame:4d}] 携物返航")

    elif phase == 5 and d < 0.1:
        phase = 6
        print(f"  [{frame:4d}] 完成!")

    if frame % 90 == 0:
        names = ["起飞","搜索","接近","开臂下降","闭臂","返航","完成"]
        t = f"({target_pos[0]:.2f},{target_pos[1]:.2f})" if target_pos is not None else "-"
        st = drone.get_state()
        act_z = f"{st['position'][2]:.2f}" if st else "--"
        obj_z = "--"
        if grabbed and target_translate is not None:
            ot = target_translate.Get()
            if ot is not None:
                obj_z = f"{ot[2]:.2f}"
        print(f"  [{frame:4d}] drone_z={act_z} obj_z={obj_z} [{names[phase]}] {t}")

simulation_app.close()
