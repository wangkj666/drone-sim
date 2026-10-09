"""生成组会用图: 栖息抱杆 / 下降抓取 / 携物返航

不跑任务链, 直接把无人机和夹爪摆到三个关键姿态, 各拍一张。
相机只用一台 —— 场景里同时存在无人机下视相机时, Camera.set_world_pose 不生效
(拍出来是下视相机的画面), 所以这里不建 DroneCamera。

用法:  python capture_shots.py     (输出到 shots/*.png)
"""
import os
import torch
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import sys
import numpy as np
import yaml
import cv2
import omni.usd
from pxr import UsdGeom, Gf
from scipy.spatial.transform import Rotation as Rot
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim, set_prim_property
from isaacsim.sensors.camera import Camera

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "drone_project"))
from drone_model import DroneModel
from gripper import Gripper

os.makedirs("shots", exist_ok=True)

cfg = yaml.safe_load(open(os.path.join("drone_project", "config.yaml"), encoding="utf-8"))
world = World(physics_dt=cfg["sim"]["physics_dt"])
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()

def make_cyl(path, translate, scale, color, rot_x=0.0):
    create_prim(path, "Cylinder")
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate))
    if rot_x:
        xf.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble).Set(float(rot_x))
    xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))
    set_prim_property(path, "primvars:displayColor", [color])

PERCH_X, PERCH_Y, PERCH_Z = 0.5, 0.0, 1.30
BAR_R, BAR_HALF = 0.022, 0.45
make_cyl("/World/Perch/Pole", (PERCH_X, BAR_HALF, PERCH_Z / 2),
         (0.030, 0.030, PERCH_Z / 2), (0.35, 0.28, 0.18))
make_cyl("/World/Perch/Bar", (PERCH_X, PERCH_Y, PERCH_Z),
         (BAR_R, BAR_R, BAR_HALF), (0.45, 0.36, 0.22), rot_x=90.0)

create_prim("/World/Target", "Cube", position=np.array([0.30, 0.0, 0.07]),
            scale=np.array([0.05, 0.05, 0.07]))
set_prim_property("/World/Target", "primvars:displayColor", [(1, 0.1, 0.1)])

drone = DroneModel(cfg).build(world, kinematic=True)
world.reset()
g = Gripper().build()

def tr_op(path):
    for op in UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op

drone_tr = tr_op("/World/Drone")
tgt_tr = tr_op("/World/Target")

# 补光: 场景自带光照很暗, 高空拍出来机身下侧全黑
from pxr import UsdLux
create_prim("/World/ShotLight", "SphereLight", position=np.array([1.2, -1.2, 2.0]))
UsdLux.SphereLight(stage.GetPrimAtPath("/World/ShotLight")).GetIntensityAttr().Set(12000.0)
UsdLux.SphereLight(stage.GetPrimAtPath("/World/ShotLight")).GetRadiusAttr().Set(0.4)
light_tr = tr_op("/World/ShotLight")

cam = Camera(prim_path="/World/PC", name="pc", resolution=(900, 680), frequency=30)
cam.initialize()
cam.set_clipping_range(0.02, 100.0)

def look_at(eye, tgt, up=np.array([0.0, 0.0, 1.0])):
    f = tgt - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    q = Rot.from_matrix(np.stack([r, u, -f], axis=1)).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

def shot(tag, twist, light_dz=0.4):
    """相机从斜前方看向无人机当前位置"""
    eye = np.array(drone_tr.Get()) + np.array(twist)
    light_tr.Set(Gf.Vec3d(*(eye + np.array([0.0, 0.0, light_dz]))))   # 跟随相机的补光
    cam.set_world_pose(position=eye, orientation=look_at(eye, np.array(drone_tr.Get())),
                       camera_axes="usd")
    for _ in range(20):          # 渲染需要几步才刷新
        world.step(render=True)
    img = cam.get_rgb()
    if img is None or not img.size or img[..., :3].max() == 0:
        print(f"[X] {tag}: 画面全黑, 没存")
        return
    cv2.imwrite(f"shots/{tag}.png", cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR))
    print(f"[OK] shots/{tag}.png")

def settle(n=200):
    for _ in range(n):
        g.update(1 / 60); world.step(render=True)

# 1) 栖息抱杆 (机心在杆下 0.075, 夹爪朝上合拢 -160°)
drone_tr.Set(Gf.Vec3d(PERCH_X, PERCH_Y, PERCH_Z - 0.075))
g.point_up(); g.open(); settle()
g.close(); settle(220)
shot("1_perch", (0.62, -0.80, 0.22), light_dz=-0.15)   # 靠近+压低补光, 否则被横杆挡住

# 2) 下降抓取 (机心 0.15, 夹爪朝下合拢, 咬住 0.14 高的目标)
drone_tr.Set(Gf.Vec3d(0.30, 0.0, 0.15))
g.point_down(); g.open(); settle()
g.close(); settle(220)
shot("2_grasp", (0.68, -1.10, 0.06))   # 相机压到与机心齐平, 否则目标被机身挡住

# 3) 携物返航 (物体按抓取瞬间的相对位置 -0.08 跟着升空)
tgt_tr.Set(Gf.Vec3d(0.10, 0.0, 1.42))
drone_tr.Set(Gf.Vec3d(0.10, 0.0, 1.50))
settle(60)
shot("3_return", (0.85, -1.25, -0.02))  # 略低于机身, 才能看见下面吊着的物体

app.close()
