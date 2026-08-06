# 无人机自主栖息与抓取 — 仿真项目

在 Isaac Sim 4.5 中搭建四旋翼无人机，实现**全自动飞行 + 下视目标感知 + 自主目标接近**。

## 环境

- Windows 11 + Isaac Sim 4.5 + Python 3.10 + RTX 4060 Laptop
- 无人机模型：WHEELTEC F570（轴距 570mm，1.5kg）

## 快速开始

```powershell
cd G:\isaac
.\isaacsim_env\Scripts\Activate.ps1
$env:OMNI_KIT_ACCEPT_EULA = "YES"

# 全自动飞行演示
python drone_project\scripts\01_fly_hover.py

# 相机测试
python drone_project\scripts\02_camera_test.py

# 感知测试（颜色检测+3D定位）
python drone_project\scripts\03_perception_test.py

# 自主目标接近（感知驱动飞行）
python drone_project\scripts\06_auto_approach.py
```

## 项目结构

```
drone_project/
├── config.yaml                 # 全局参数（无人机尺寸、PID增益等）
├── drone_model.py              # 无人机模型（几何体搭建）
├── flight_controller.py        # 级联 PID 控制器（含抗饱和）
├── mixer.py                    # X 型四旋翼混控器
├── camera.py                   # 下视 RGB 相机模块
├── perception.py               # 颜色检测 + 3D 世界坐标定位
├── report.md                   # 阶段性成果报告
└── scripts/
    ├── 01_fly_hover.py         # 全自动飞行（方形航线）
    ├── 02_camera_test.py       # 下视相机实时保存图像
    ├── 03_perception_test.py   # 颜色感知 + 3D 定位测试
    ├── 04_closed_loop_hover.py # 力控 PID 闭环测试（理论验证）
    ├── 05_import_model.py      # CAD 模型导入（开发中）
    └── 06_auto_approach.py     # 感知驱动自主目标接近
```

## 已实现功能

- [x] 全自动飞行（起飞、方形航线、降落）
- [x] 下视相机实时拍摄
- [x] 颜色检测 + 3D 世界坐标定位（误差 < 0.15m）
- [x] 级联 PID 控制器设计（理论方案）
- [x] X 型混控器
- [x] 感知驱动自主目标接近

## 注意事项

- `import torch` 必须在 `from isaacsim import SimulationApp` 之前
- 必须设置环境变量 `OMNI_KIT_ACCEPT_EULA=YES`
- 首次启动需从 NVIDIA 下载约 800 个扩展文件（缓存后不再下载）
- 力控 PID 在当前仿真环境下受 60Hz + 低惯量限制，演示使用运动学控制
- 实物部署建议使用 PX4/ArduPilot 飞控固件，通过 MAVLink 发送目标坐标

## 实物迁移

| 仿真 | 实物 |
|------|------|
| 运动学飞控演示 | → 高层逻辑直接迁移 |
| 感知模块（颜色/位姿） | → 部署到机载电脑（Jetson/RPi） |
| PID 控制器设计 | → 用 PX4 飞控替代底层 PID |
| 目标坐标指令 | → 通过 MAVLink 发送给飞控 |
