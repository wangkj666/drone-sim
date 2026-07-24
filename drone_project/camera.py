"""
无人机下视相机 — 使用isaacsim.sensors.camera + 手动旋转
"""
import numpy as np


class DroneCamera:

    def __init__(self, prim_path: str = "/World/Drone/Camera",
                 resolution: tuple = (640, 480),
                 frequency: int = 30):
        self.prim_path = prim_path
        self.resolution = resolution
        self.frequency = frequency
        self._camera = None
        self._annotators = {}

    def initialize(self):
        from isaacsim.sensors.camera import Camera
        import omni.replicator.core as rep

        self._camera = Camera(
            prim_path=self.prim_path,
            name="drone_cam",
            frequency=self.frequency,
            resolution=self.resolution,
        )

        # 先设局部位姿，再初始化（防止被initialize覆盖）
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath(self.prim_path)
        if cam_prim:
            cam_xf = UsdGeom.Xformable(cam_prim)
            cam_xf.ClearXformOpOrder()
            cam_xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.15))  # 无人机下方15cm

        self._camera.initialize()  # 在设置位姿之后初始化
        print("[Camera] pos: z=-0.15 below drone, no rotation (default -Z = down)")

        # 渲染产品 + 标注器
        rp = rep.create.render_product(self.prim_path, self.resolution)
        self._annotators["rgb"] = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annotators["depth"] = rep.AnnotatorRegistry.get_annotator(
            "distance_to_camera")
        for name, ann in self._annotators.items():
            if ann:
                ann.attach(rp)

        print(f"[Camera] ready: {self.prim_path} {self.resolution[0]}x{self.resolution[1]}")
        return self

    def get_rgb(self) -> np.ndarray:
        ann = self._annotators.get("rgb")
        if ann is None:
            return np.zeros((*self.resolution[::-1], 4), dtype=np.uint8)
        data = ann.get_data()
        return data if data is not None else np.zeros((*self.resolution[::-1], 4), dtype=np.uint8)

    def get_intrinsics(self) -> np.ndarray:
        """返回相机内参矩阵 3x3"""
        if self._camera is not None:
            return self._camera.get_intrinsics_matrix()
        w, h = self.resolution
        return np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)

    def get_depth(self) -> np.ndarray:
        ann = self._annotators.get("depth")
        if ann is None:
            return np.zeros(self.resolution[::-1], dtype=np.float32)
        data = ann.get_data()
        return data if data is not None else np.zeros(self.resolution[::-1], dtype=np.float32)
