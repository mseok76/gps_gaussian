import numpy as np
import cv2
import os
from pathlib import Path
import math
import random

def save(pid, data_id, vid, save_path, extr, intr, img, mask):
    # img_save_path = os.path.join(save_path, 'img', data_id, '%03d' % pid)
    # mask_save_path = os.path.join(save_path, 'mask', data_id, '%03d' % pid)
    # parm_save_path = os.path.join(save_path, 'parm', data_id, '%03d' % pid)
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

def calculate_intrinsic(res):
    # fx = res[0] * 0.8
    # fy = res[1] * 0.8
    # cx = res[0] * 0.5
    # cy = res[1] * 0.5
    fx = 835.14
    fy = 1105.71
    cx = 512
    cy = 512
    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    return intrinsic

# def calculate_extrinsic(angle, dis, look_at_center):
#     ori_vec = np.array([0, 0, dis])
#     rotate = np.matmul(rotationY(math.radians(angle)), rotationX(math.radians(-8)))
#     cam_pos = look_at_center + np.matmul(rotate, ori_vec)
#     target = look_at_center
#     zaxis = (cam_pos - target) / np.linalg.norm(cam_pos - target)
#     xaxis = np.cross(np.array([0, 1, 0]), zaxis)
#     xaxis /= np.linalg.norm(xaxis)
#     yaxis = np.cross(zaxis, xaxis)
#     extrinsic = np.eye(4)
#     extrinsic[:3, :3] = np.vstack([xaxis, yaxis, zaxis]).T
#     extrinsic[:3, 3] = -np.matmul(extrinsic[:3, :3], cam_pos)
#     return extrinsic[:3]

def calculate_extrinsic(angle, dis, look_at_center):
    # 객체와의 거리를 유지하면서 특정 각도(angle)로 회전한 카메라 위치 계산
    ori_vec = np.array([0, 0, dis])  # 카메라 초기 위치 벡터 (z 방향으로 거리 dis 만큼 떨어짐)
    rotate = np.matmul(rotationY(math.radians(angle)), rotationX(math.radians(0)))  # x축 회전 제거하고 y축으로만 회전
    cam_pos = look_at_center + np.matmul(rotate, ori_vec)  # 회전 후 카메라 위치 계산

    # 카메라 방향 설정 (객체의 중심을 바라봄)
    target = look_at_center
    zaxis = (cam_pos - target) / np.linalg.norm(cam_pos - target)  # 카메라가 바라보는 방향의 z축
    xaxis = np.cross(np.array([0, 1, 0]), zaxis)  # y축과 z축의 외적을 통해 x축 계산
    xaxis /= np.linalg.norm(xaxis)  # 단위 벡터로 정규화
    yaxis = np.cross(zaxis, xaxis)  # z축과 x축의 외적을 통해 y축 계산

    # Extrinsic 행렬 생성
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.vstack([xaxis, yaxis, zaxis]).T  # 회전 행렬 설정
    extrinsic[:3, 3] = -np.matmul(extrinsic[:3, :3], cam_pos)  # 평행 이동 벡터 설정
    return extrinsic[:3]  # 3x4 형태로 반환

def generate_mask(img):
    # 이미지를 흑백으로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 이진화 처리 (밝은 부분을 사람 영역으로 가정)
    _, binary_mask = cv2.threshold(gray, 3, 1, cv2.THRESH_BINARY)
    
    # 마스크 값을 0과 255로 조정
    mask = (np.clip(binary_mask, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    # print(mask.shape)
    # 3차원 numpy로 expand
    mask_3d = np.stack([mask] * 3, axis=-1)  
    # print(mask_3d.shape)

    return mask_3d

def render_images(data_path, save_path, cam_nums=16, res=(1024, 1024), dis=2.2):
    data_folders = sorted(os.listdir(os.path.join(data_path, 'img')))
    intrinsic = calculate_intrinsic(res)
    look_at_center = np.array([0, 0.85, 0])
    
    degree_interval = 360 / cam_nums
    # angle_list1 = list(range(360 - int(degree_interval // 2), 360))
    # angle_list2 = list(range(0, int(degree_interval // 2)))
    # angle_list = angle_list1 + angle_list2
    # angle_base = np.random.choice(angle_list, 1)[0]
    angle_base = 350

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
            extr = calculate_extrinsic(angle_base + pid * degree_interval, dis, look_at_center)
            save(pid, data_id, pid, save_path, extr, intrinsic, img, mask)

if __name__ == '__main__':
    random.seed(42)
    data_path = '../gps_dataset/test_data'
    save_path = '../gps_dataset/processed_data'
    render_images(data_path, save_path)
