import numpy as np
import cv2
import os
from pathlib import Path
import math

# 경로 설정
image_folder = "path/to/images"  # 16장의 카메라 사진이 있는 폴더
save_folder = "path/to/save"
Path(save_folder).mkdir(parents=True, exist_ok=True)

# 카메라 내부 파라미터 설정
res = (1024, 1024)  # 사진 해상도
fx, fy = res[0] * 0.8, res[1] * 0.8  # focal length
cx, cy = res[0] / 2, res[1] / 2      # principal point
intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# 16방향 각도 설정
cam_nums = 16
degree_interval = 360 / cam_nums

# 각 이미지에 대해 extrinsic, intrinsic, mask 생성 및 저장
for i in range(cam_nums):
    angle = degree_interval * i  # 각도 설정
    img_path = os.path.join(image_folder, f"image_{i:03d}.jpg")  # 이미지 파일명 형식

    # 이미지 로드
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image {img_path} not found.")
        continue

    # Extrinsic 행렬 생성 (카메라 위치 및 방향)
    # 카메라가 (0, 0, 1)에서 바라보는 위치로 설정하고 Y축을 기준으로 회전
    distance = 2.0  # 카메라와 객체 사이의 거리
    cam_pos = np.array([distance * math.cos(math.radians(angle)), 0, distance * math.sin(math.radians(angle))])
    look_at = np.array([0, 0, 0])  # 원점 바라보는 경우
    up_vector = np.array([0, 1, 0])

    forward = (look_at - cam_pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(up_vector, forward)
    up = np.cross(forward, right)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.vstack([right, up, forward]).T
    extrinsic[:3, 3] = cam_pos

    # Mask 생성 (이진화)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    # 저장
    np.save(os.path.join(save_folder, f"intrinsic_{i:03d}.npy"), intrinsic)
    np.save(os.path.join(save_folder, f"extrinsic_{i:03d}.npy"), extrinsic[:3])  # 3x4 extrinsic 행렬
    cv2.imwrite(os.path.join(save_folder, f"mask_{i:03d}.png"), mask)

    print(f"Saved intrinsic, extrinsic, and mask for image {i}")
