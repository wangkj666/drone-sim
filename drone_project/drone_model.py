"""
无人机模型 — 在 Isaac Sim 中构建四旋翼无人机
提供: 模型构建、状态读取、力/力矩施加
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


class DroneModel:
    """
    X型四旋翼无人机

    机身坐标系: X=右, Y=前, Z=上
    电机: M1(FR), M2(FL), M3(RL), M4(RR)
    """

    def __init__(self, config: dict):
        self.config = config
        d = config["drone"]

        self.mass = d["mass"]
        self.arm_length = d["arm_length"]
        self.init_pos = np.array(d["init_position"])
        self.gravity = config["sim"]["gravity"]
        self.dt = config["sim"]["physics_dt"]

        self._s = self.arm_length / np.sqrt(2)
        self.rotor_positions = np.array([
            [ self._s,  self._s, 0],
            [-self._s,  self._s, 0],
            [-self._s, -self._s, 0],
            [ self._s, -self._s, 0],
        ])

        self._prim_path = "/World/Drone"
        self._rigid_view = None  # RigidPrimView (支持施力)
        self._built = False

    def build(self, world):
        """在仿真世界中构建无人机"""
        from omni.isaac.core.utils.prims import create_prim, set_prim_property
        from omni.isaac.core.prims import RigidPrimView
        from pxr import UsdPhysics, UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()

        # ── 1. 根节点 ──
        create_prim(
            prim_path=self._prim_path,
            prim_type="Xform",
            position=self.init_pos,
        )

        # ── 2. 机身 ──
        create_prim(
            prim_path=f"{self._prim_path}/Body",
            prim_type="Cube",
            position=np.array([0, 0, 0]),
            scale=np.array([self.config["drone"]["frame_length"],
                           self.config["drone"]["frame_width"],
                           self.config["drone"]["frame_height"]]),
        )
        set_prim_property(f"{self._prim_path}/Body", "primvars:displayColor",
                          [(0.2, 0.2, 0.35)])

        # ── 3. 机臂 ──
        arm_dirs = [(self._s, self._s), (-self._s, self._s),
                     (-self._s, -self._s), (self._s, -self._s)]
        for i, (x, y) in enumerate(arm_dirs):
            create_prim(
                prim_path=f"{self._prim_path}/Arm{i}",
                prim_type="Cylinder",
                position=np.array([x * 0.5, y * 0.5, 0]),
                scale=np.array([0.015, 0.015, self.arm_length / 2]),
            )
            arm_prim = stage.GetPrimAtPath(f"{self._prim_path}/Arm{i}")
            if arm_prim:
                UsdGeom.Xformable(arm_prim).AddRotateXOp().Set(90.0)

        # ── 4. 电机 + 螺旋桨 ──
        for i, (rx, ry, rz) in enumerate(self.rotor_positions):
            create_prim(f"{self._prim_path}/Motor{i}", "Cylinder",
                        position=np.array([rx, ry, self.config["drone"]["motor_height"] / 2]),
                        scale=np.array([self.config["drone"]["motor_dia"] / 2,
                                       self.config["drone"]["motor_dia"] / 2,
                                       self.config["drone"]["motor_height"] / 2]))
            set_prim_property(f"{self._prim_path}/Motor{i}", "primvars:displayColor", [(0.15, 0.15, 0.15)])

            create_prim(f"{self._prim_path}/Prop{i}", "Cube",
                        position=np.array([rx, ry, self.config["drone"]["motor_height"] + 0.01]),
                        scale=np.array([self.config["drone"]["prop_length"],
                                       0.005, self.config["drone"]["prop_width"]]))
            set_prim_property(f"{self._prim_path}/Prop{i}", "primvars:displayColor", [(0.9, 0.9, 0.9)])

        # ── 5. 起落架 ──
        leg_offsets = [(0.08, 0.06), (-0.08, 0.06), (-0.08, -0.06), (0.08, -0.06)]
        for i, (lx, ly) in enumerate(leg_offsets):
            create_prim(f"{self._prim_path}/Leg{i}", "Cylinder",
                        position=np.array([lx, ly, -self.config["drone"]["frame_height"] / 2 - 0.04]),
                        scale=np.array([0.004, 0.004, 0.05]))
            set_prim_property(f"{self._prim_path}/Leg{i}", "primvars:displayColor", [(0.12, 0.12, 0.12)])

        # ── 6. 物理属性 ──
        drone_prim = stage.GetPrimAtPath(self._prim_path)
        UsdPhysics.RigidBodyAPI.Apply(drone_prim)
        mass_api = UsdPhysics.MassAPI.Apply(drone_prim)
        mass_api.GetMassAttr().Set(self.mass)

        # 角速度阻尼 (配合速度控制稳定)
        from pxr import PhysxSchema
        physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(drone_prim)
        physx_api.GetAngularDampingAttr().Set(0.5)

        # 球形碰撞体（不会像薄盒子那样碰地翻倒）
        coll_path = f"{self._prim_path}/Collision"
        create_prim(coll_path, "Sphere",
                    position=np.array([0, 0, 0]),
                    scale=np.array([self.arm_length * 0.8,
                                   self.arm_length * 0.8,
                                   self.arm_length * 0.8]))
        coll_prim = stage.GetPrimAtPath(coll_path)
        UsdPhysics.CollisionAPI.Apply(coll_prim)
        try:
            UsdGeom.Imageable(coll_prim).MakeInvisible()
        except Exception:
            pass

        # ── 7. 使用 RigidPrimView（支持 apply_forces） ──
        self._rigid_view = RigidPrimView(
            prim_paths_expr=self._prim_path,
            name="drone_view",
        )
        world.scene.add(self._rigid_view)

        self._built = True
        print(f"Drone built at {self._prim_path}")
        return self

    def get_state(self) -> dict:
        """读取无人机当前状态"""
        if self._rigid_view is None:
            return None

        positions, orientations = self._rigid_view.get_world_poses()
        lin_vels = self._rigid_view.get_linear_velocities()
        ang_vels = self._rigid_view.get_angular_velocities()

        pos = np.array(positions[0])
        orient_quat = np.array(orientations[0])
        lin_vel = np.array(lin_vels[0])
        ang_vel = np.array(ang_vels[0])

        r = R.from_quat([orient_quat[1], orient_quat[2], orient_quat[3], orient_quat[0]])
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)

        return {
            "position": pos,
            "velocity": lin_vel,
            "attitude": np.array([roll, pitch, yaw]),
            "angular_velocity": ang_vel,
            "quaternion": orient_quat,
        }

    def apply_rotor_forces(self, motor_thrusts: np.ndarray):
        """
        施加四路电机推力
        motor_thrusts: [M1, M2, M3, M4] (N)
        """
        if self._rigid_view is None:
            return

        state = self.get_state()
        if state is None:
            return

        quat = state["quaternion"]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        body_z = r.apply(np.array([0, 0, 1]))

        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        for i, thrust in enumerate(motor_thrusts):
            rotor_pos_world = r.apply(self.rotor_positions[i])
            force_world = body_z * thrust
            total_force += force_world
            total_torque += np.cross(rotor_pos_world, force_world)

            yaw_dir = -1 if i in [0, 2] else 1
            total_torque += body_z * yaw_dir * thrust * self.config["control"]["thrust_to_torque"]

        # RigidPrim (批量视图) 施力
        self._rigid_view.apply_forces_and_torques_at_pos(
            forces=total_force.reshape(1, 3),
            torques=total_torque.reshape(1, 3),
            indices=[0],
            is_global=True,
        )

    def get_prim_path(self) -> str:
        return self._prim_path
