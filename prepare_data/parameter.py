import numpy as np
import cv2
import os
from pathlib import Path
import math

def save(pid, data_id, vid, save_path, extr, intr, img, mask):
    img_save_path = os.path.join(save_path, 'img', data_id)
    mask_save_path = os.path.join(save_path, 'mask', data_id)
    parm_save_path = os.path.join(save_path, 'parm', data_id)
    
    Path(img_save_path).mkdir(exist_ok=True, parents=True)
    Path(mask_save_path).mkdir(exist_ok=True, parents=True)
    Path(parm_save_path).mkdir(exist_ok=True, parents=True)

    cv2.imwrite(os.path.join(img_save_path, '{}.jpg'.format(vid)), img)
    cv2.imwrite(os.path.join(mask_save_path, '{}.png'.format(vid)), mask)
    np.save(os.path.join(parm_save_path, '{}_intrinsic.npy'.format(vid)), intr)
    np.save(os.path.join(parm_save_path, '{}_extrinsic.npy'.format(vid)), extr)

def rotationX(angle):
    return np.array([
        [1, 0, 0],
        [0, math.cos(angle), -math.sin(angle)],
        [0, math.sin(angle), math.cos(angle)]
    ])

def rotationY(angle):
    return np.array([
        [math.cos(angle), 0, math.sin(angle)],
        [0, 1, 0],
        [-math.sin(angle), 0, math.cos(angle)]
    ])

def calculate_intrinsic(res, sensor_size=(7.4, 5.6), focal_length=4.3, input_res=(1440, 1440)):
    # 모델 입력 해상도에 맞춰 변환 비율 설정
    scale_x = res[0] / input_res[0]
    scale_y = res[1] / input_res[1]

    # 초점 거리(fx, fy) 계산
    fx = (focal_length / sensor_size[0]) * input_res[0] * scale_x
    fy = (focal_length / sensor_size[1]) * input_res[1] * scale_y

    # 이미지의 주점(cx, cy)를 이미지의 중심으로 설정
    cx = res[0] * 0.5
    cy = res[1] * 0.5

    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    return intrinsic

def calculate_extrinsic(angle, dis, look_at_center, height=1.5):
    # 거리와 높이를 반영하여 카메라 위치 설정
    ori_vec = np.array([0, height, dis])
    
    # 수평으로 각도 회전 적용
    # rotate = np.matmul(rotationY(math.radians(angle)))
    rotate = rotationY(math.radians(angle))
    cam_pos = look_at_center + np.matmul(rotate, ori_vec)
    target = look_at_center
    
    # 월드 좌표계의 기준축 설정
    zaxis = (cam_pos - target) / np.linalg.norm(cam_pos - target)
    xaxis = np.cross(np.array([0, 1, 0]), zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(zaxis, xaxis)
    
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.vstack([xaxis, yaxis, zaxis]).T
    extrinsic[:3, 3] = -np.matmul(extrinsic[:3, :3], cam_pos)
    return extrinsic[:3]

def generate_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 3, 1, cv2.THRESH_BINARY)
    mask = (np.clip(binary_mask, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    mask_3d = np.stack([mask] * 3, axis=-1)
    return mask_3d

def render_images(data_path, save_path, cam_nums=16, res=(1024, 1024), dis=2.0):
    data_folders = sorted(os.listdir(os.path.join(data_path, 'img')))
    intrinsic = calculate_intrinsic(res)
    look_at_center = np.array([0, 0, 0])

    # 각도 설정 (16방향으로 22.5도 간격)
    degree_interval = 360 / cam_nums
    angle_list = [degree_interval * i for i in range(cam_nums)]

    for data_id in data_folders:
        img_folder_path = os.path.join(data_path, 'img', data_id)
        images = sorted(os.listdir(img_folder_path))
        
        for pid, image_name in enumerate(images):
            img_path = os.path.join(img_folder_path, image_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            mask = generate_mask(img)

            # Save for each viewpoint (extrinsic, intrinsic, mask for each image)
            extr = calculate_extrinsic(angle_list[pid % cam_nums], dis, look_at_center, height=1.5)
            save(pid, data_id, pid, save_path, extr, intrinsic, img, mask)

if __name__ == '__main__':
    data_path = '../gps_dataset/test_data'
    save_path = '../gps_dataset/processed_data'
    render_images(data_path, save_path)
