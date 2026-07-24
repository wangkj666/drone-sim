"""
感知模块 — 颜色检测 + ArUco检测 + 3D位置估计
"""
import cv2
import numpy as np


class ColorDetector:
    """基于颜色检测目标方块，估算3D位置"""

    # HSV 颜色范围
    COLOR_RANGES = {
        "red":    [(0, 120, 80), (10, 255, 255)],
        "green":  [(45, 120, 80), (75, 255, 255)],
        "blue":   [(105, 150, 100), (125, 255, 255)],  # 高饱和度防误检
        "yellow": [(22, 120, 80), (33, 255, 255)],
    }

    def __init__(self, camera_height: float = 2.0,
                 camera_matrix: np.ndarray = None):
        self.camera_height = camera_height
        if camera_matrix is not None:
            self.fx = camera_matrix[0, 0]
            self.fy = camera_matrix[1, 1]
            self.cx = camera_matrix[0, 2]
            self.cy = camera_matrix[1, 2]
        else:
            self.fx = self.fy = 640
            self.cx, self.cy = 320, 240

    def detect(self, rgb_image: np.ndarray,
               drone_pos: np.ndarray = None) -> list:
        """
        检测图像中的彩色方块
        drone_pos: 无人机世界坐标 [x, y, z], 用于计算目标世界位置
        返回: [(颜色名, world_xyz, 图像中心), ...]
        """
        if rgb_image is None or rgb_image.size == 0:
            return []

        hsv = cv2.cvtColor(rgb_image[..., :3], cv2.COLOR_RGB2HSV)
        results = []

        # 相机离地高度
        z = self.camera_height
        if drone_pos is not None:
            z = drone_pos[2]  # 使用实际无人机高度

        for name, (lower, upper) in self.COLOR_RANGES.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

                # 图像坐标 → 相机相对坐标 (图像Y向下, 世界Y向前 → 取反)
                rel_x =  (cx - self.cx) * z / self.fx
                rel_y = -(cy - self.cy) * z / self.fy

                # → 世界坐标
                world_x = rel_x
                world_y = rel_y
                if drone_pos is not None:
                    world_x += drone_pos[0]
                    world_y += drone_pos[1]

                results.append((name, np.array([world_x, world_y, 0.0]),
                                (int(cx), int(cy))))

        return results

    def draw(self, rgb_image: np.ndarray, detections: list) -> np.ndarray:
        vis = rgb_image[..., :3].copy()
        for name, xyz, (cx, cy) in detections:
            cv2.circle(vis, (cx, cy), 10, (0, 255, 0), 2)
            cv2.putText(vis, f"{name} ({xyz[0]:.2f},{xyz[1]:.2f})",
                        (cx + 12, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        return vis


class ArUcoDetector:
    """ArUco 标记检测 (备用)"""

    def __init__(self, marker_size: float = 0.12):
        self.marker_size = marker_size
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

    def detect(self, rgb_image: np.ndarray) -> list:
        if rgb_image is None or rgb_image.size == 0:
            return []
        gray = cv2.cvtColor(rgb_image[..., :3], cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return []
        results = []
        for i, mid in enumerate(ids.flatten()):
            c = corners[i][0]
            cx, cy = c[:, 0].mean(), c[:, 1].mean()
            results.append((int(mid), (int(cx), int(cy))))
        return results

    def draw(self, rgb_image: np.ndarray, detections: list) -> np.ndarray:
        vis = rgb_image[..., :3].copy()
        for mid, (cx, cy) in detections:
            cv2.circle(vis, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(vis, f"ID:{mid}", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return vis
