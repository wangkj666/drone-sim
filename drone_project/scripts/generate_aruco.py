"""
生成 ArUco 标记图片，用于贴到仿真场景中的目标物体上
"""
import cv2
import numpy as np
import os

# 输出目录
out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "markers")
os.makedirs(out_dir, exist_ok=True)

# 使用 DICT_4X4_50 字典（小标记，适合远距离检测）
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

marker_size = 200  # 像素

for marker_id in range(4):
    img = np.zeros((marker_size, marker_size), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, img, 1)

    # 加白边
    bordered = np.ones((marker_size + 40, marker_size + 40), dtype=np.uint8) * 255
    bordered[20:20 + marker_size, 20:20 + marker_size] = img

    path = os.path.join(out_dir, f"aruco_{marker_id}.png")
    cv2.imwrite(path, bordered)
    print(f"  Saved: {path} (ID={marker_id})")

print(f"\nGenerated {4} ArUco markers in {out_dir}")
print("Use DICT_4X4_50, IDs 0-3")
