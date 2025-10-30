import os
# import math
import json
import glob
import pickle
# import sys
import copy
import argparse

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.cm as cm

import torch
# import torch.nn as nn
# import torch.utils.data as data
from torchvision import transforms as T

# from setuptools import setup, find_packages
# from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange
from PIL import Image
import cv2
import open3d as o3d

from lidargsu.utils.common import *


def align_img(img: np.ndarray, img_size: int = 64) -> np.ndarray:
    """Aligns the image to the center.
    Args:
        img (np.ndarray): Image to align.
        img_size (int, optional): Image resizing size. Defaults to 64.
    Returns:
        np.ndarray: Aligned image.
    """    
    if img.sum() <= 10000:
        y_top = 0
        y_btm = img.shape[0]
    else:
        # Get the upper and lower points
        # img.sum
        y_sum = img.sum(axis=2).sum(axis=1)
        y_top = (y_sum != 0).argmax(axis=0)
        y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)

    img = img[y_top: y_btm, :,:]

    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    ratio = img.shape[1] / img.shape[0]
    img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)
    
    # Get the median of the x-axis and take it as the person's x-center.
    x_csum = img.sum(axis=2).sum(axis=0).cumsum()

    x_center = img.shape[1] // 2
    for idx, csum in enumerate(x_csum):
        if csum > img.sum() / 2:
            x_center = idx
            break
    
    half_width = img_size // 2
    img_copy = img.copy()
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            for c in range(img.shape[2]):
                img_copy[h][((w - x_center + img.shape[1]//2)) % img.shape[1]][c] = img[h][w][c]
    
    # if not x_center:
    #     logging.warning(f'{img_file} has no center.')
    #     continue
    
    # Get the left and right points
    # half_width = img_size // 2
    # left = x_center - half_width
    # right = x_center + half_width

    # if left <= 0 or right >= img.shape[1]:
    #     left += half_width 
    #     right += half_width
    #     # _ = np.zeros((img.shape[0], half_width,3))
    #     # img = np.concatenate([_, img, _], axis=1)

    left = img.shape[1]//2 - half_width
    right = img.shape[1]//2 + half_width

    # if left <= 0 or right >= img.shape[1]:
    #     img_copy = img.copy()
    #     for h in range(img.shape[0]):
    #         for c in range(img.shape[2]):
    #             for w in range(img.shape[1]):
    #                 img_copy[h][(w + 4 * half_width) // img.shape[1]][c] = img[h][w][c]
    #     left += (4 * half_width)
    #     right += (4 * half_width)
    
    img = img_copy[:, left:right, :].astype('uint8')
    return img


# For SUSTeck1K
def lidar_to_dep_and_pos_for_susteck1k(points_seq, 
                     v_res, 
                     h_res, 
                     v_fov, 
                     pkl_depth_file_path_clean, 
                     pkl_depth_file_path_v12p56f1010,
                     pkl_depth_file_path_v13p46f1010,  
                     pkl_depth_file_path_v14p36f1010, 
                     pkl_depth_file_path_v11p66f710, 
                     pkl_depth_file_path_v14p36f710, 
                     pkl_pos_file_path_clean, 
                     pkl_pos_file_path_v12p56f1010, 
                     pkl_pos_file_path_v13p46f1010, 
                     pkl_pos_file_path_v14p36f1010, 
                     pkl_pos_file_path_v11p66f710, 
                     pkl_pos_file_path_v14p36f710, 
                     proj_mode, 
                     depth_norm_mode,  
                     set_name, 
                     res_const, 
                     frame_const, 
                     y_fudge=0.0, 
                     val = 'depth', 
                     ):
    
    assert len(v_fov) == 2, "v_fov must be list/tuple of length 2"
    assert v_fov[0] <= 0, "first element in v_fov must be 0 or negative"
    assert val in ["depth", "height", "reflectance"], 'val must be one of {"depth", "height", "reflectance"}'

    if res_const == 0.030:
        img_z_add = 0.1
        img_y_add = 1.9
    elif res_const == 0.035:
        img_z_add = 0.2
        img_y_add = 2.2
    elif res_const == 0.040:
        img_z_add = 0.3
        img_y_add = 2.6

    pixel_v = 64
    pixel_h = 64
    pixel_res = res_const
    
    if frame_const == 'all':
        F = len(points_seq)
    else:
        points_seq = points_seq[:frame_const]
        F = len(points_seq)

    assert proj_mode in ['ortho', 'spher'], f'Makre sure your projection_mode: {proj_mode}'
    assert img_z_add in [0.1, 0.2, 0.3], f'Make sure your img_z_add: {img_z_add}'
    assert img_y_add in [1.9, 2.2, 2.6], f'Make sure your img_y_add: {img_y_add}'
    assert pixel_v in [64], f'Make sure your pixel_v: {pixel_v}'
    assert pixel_h in [64], f'Make sure your pixel_h: {pixel_h}'
    assert pixel_res in [0.030, 0.035, 0.040], f'Make sure your pixel_res: {pixel_res}'

    if proj_mode == 'ortho': 
        z_max = pixel_v * pixel_res
        x_max = pixel_h * pixel_res

        video_np_orig = np.zeros((F, pixel_v, pixel_h))
        xy_points_center_seq = list()

        for f, points in enumerate(points_seq):
            x_points = points[:, 0]
            y_points = points[:, 1]
            z_points = points[:, 2]

            rad_xy = np.arctan2(y_points, x_points)

            x_points_center = np.mean(x_points)
            y_points_center = np.mean(y_points)
            
            xy_points_center = np.array([x_points_center, y_points_center])
            xy_points_center_seq.append(xy_points_center)

            x_points_replaced = x_points - x_points_center
            y_points_replaced = y_points - y_points_center

            x_points_replaced_orig = copy.deepcopy(x_points_replaced)
            y_points_replaced_orig = copy.deepcopy(y_points_replaced)
            
            P = x_points.shape[0]

            # Sensor-view
            for p in range(P):
                xy_points_replaced_p = np.array([x_points_replaced[p], y_points_replaced[p]])

                rad_xy_p = rad_xy[p] + np.pi 
                rot_mat_p = rotation_matrix(rad_xy_p)
                xy_points_replaced_p = xy_points_replaced_p @ rot_mat_p

                x_points_replaced_orig[p] = xy_points_replaced_p[0]
                y_points_replaced_orig[p] = xy_points_replaced_p[1]

            # Normalize depths
            z_min = z_points.min()

            x_points_replaced_orig = x_points_replaced_orig + x_max/2
            y_points_replaced_orig = y_points_replaced_orig + x_max/2
            z_points_replaced_orig = z_points - z_min + img_z_add

            depth_norm_orig = x_points_replaced_orig / img_y_add

            # Map points onto video_np_orig using z-buffer manner
            for p in range(P):
                pixel_v_idx = int(z_points_replaced_orig[p] / pixel_res)
                pixel_h_idx = int(y_points_replaced_orig[p] / pixel_res)
                if (0 <= pixel_v_idx < pixel_v) and (0 <= pixel_h_idx < pixel_h):
                    if video_np_orig[f][pixel_v_idx][pixel_h_idx] <= depth_norm_orig[p]:
                        video_np_orig[f][pixel_v_idx][pixel_h_idx] = depth_norm_orig[p]
                    else:
                        pass
                else:
                    pass

        # Clean
        video_np_orig = rearrange(video_np_orig, 'f h w -> f 1 h w')
        video_np_orig = np.flip(video_np_orig, axis=2) 
        xy_points_center_seq = np.array(xy_points_center_seq)

        F, C, H, W = video_np_orig.shape


        # v12p56f1010
        v_const = 2
        p_const = 5/6
        f_const = 0/10

        video_np_v12p56f1010 = copy.deepcopy(video_np_orig)
        
        mask_np = np.zeros_like(video_np_orig)      
        mask_np[:, :, ::v_const, :] = 1
        video_np_v12p56f1010 = video_np_v12p56f1010 * mask_np
        
        mask_np = np.random.choice([0, 1], size=(F, C, H, W), p=[1. - p_const, p_const])
        video_np_v12p56f1010 = video_np_v12p56f1010 * mask_np


        # v13p46f1010
        v_const = 3
        p_const = 4/6
        f_const = 0/10

        video_np_v13p46f1010 = copy.deepcopy(video_np_orig)
        
        mask_np = np.zeros_like(video_np_orig)      
        mask_np[:, :, ::v_const, :] = 1
        video_np_v13p46f1010 = video_np_v13p46f1010 * mask_np
        
        mask_np = np.random.choice([0, 1], size=(F, C, H, W), p=[1. - p_const, p_const])
        video_np_v13p46f1010 = video_np_v13p46f1010 * mask_np


        # v14p36f1010
        v_const = 4
        p_const = 3/6
        f_const = 0/10

        video_np_v14p36f1010 = copy.deepcopy(video_np_orig)
        
        mask_np = np.zeros_like(video_np_orig)      
        mask_np[:, :, ::v_const, :] = 1
        video_np_v14p36f1010 = video_np_v14p36f1010 * mask_np
        
        mask_np = np.random.choice([0, 1], size=(F, C, H, W), p=[1. - p_const, p_const])
        video_np_v14p36f1010 = video_np_v14p36f1010 * mask_np


        # v11p66f710
        v_const = 1
        p_const = 6/6
        f_const = 7/10

        video_np_v11p66f710 = copy.deepcopy(video_np_orig)

        mask_np = np.random.choice(F, int(F * f_const), replace=False)
        video_np_v11p66f710[mask_np, :, :, :] = 0


        # v14p36f710
        v_const = 4
        p_const = 3/6
        f_const = 3/10

        video_np_v14p36f710 = copy.deepcopy(video_np_orig)
        
        mask_np = np.zeros_like(video_np_orig)      
        mask_np[:, :, ::v_const, :] = 1
        video_np_v14p36f710 = video_np_v14p36f710 * mask_np
        
        mask_np = np.random.choice([0, 1], size=(F, C, H, W), p=[1. - p_const, p_const])
        video_np_v14p36f710 = video_np_v14p36f710 * mask_np

        mask_np = np.random.choice(F, int(F * f_const), replace=False)
        video_np_v14p36f710[mask_np, :, :, :] = 0


        # Save pkl files
        with open(pkl_depth_file_path_clean, 'wb') as f:
            pickle.dump(video_np_orig, f)

        with open(pkl_depth_file_path_v12p56f1010, 'wb') as f:
            pickle.dump(video_np_v12p56f1010, f)

        with open(pkl_depth_file_path_v13p46f1010, 'wb') as f:
            pickle.dump(video_np_v13p46f1010, f)

        with open(pkl_depth_file_path_v14p36f1010, 'wb') as f:
            pickle.dump(video_np_v14p36f1010, f)

        with open(pkl_depth_file_path_v11p66f710, 'wb') as f:
            pickle.dump(video_np_v11p66f710, f)

        with open(pkl_depth_file_path_v14p36f710, 'wb') as f:
            pickle.dump(video_np_v14p36f710, f)

        with open(pkl_pos_file_path_clean, 'wb') as f:
            pickle.dump(xy_points_center_seq, f)

        with open(pkl_pos_file_path_v12p56f1010, 'wb') as f:
            pickle.dump(xy_points_center_seq, f)

        with open(pkl_pos_file_path_v13p46f1010, 'wb') as f:
            pickle.dump(xy_points_center_seq, f)

        with open(pkl_pos_file_path_v14p36f1010, 'wb') as f:
            pickle.dump(xy_points_center_seq, f)

        with open(pkl_pos_file_path_v11p66f710, 'wb') as f:
            pickle.dump(xy_points_center_seq, f)

        with open(pkl_pos_file_path_v14p36f710, 'wb') as f:
            pickle.dump(xy_points_center_seq, f)
            

    if proj_mode == 'spher': 
        
        for f_idx, points_fil in enumerate(points_seq):
            x_lidar = - points_fil[:, 0]
            y_lidar = - points_fil[:, 1]
            z_lidar = points_fil[:, 2]
            
            d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)                     # distance relative to origin when looked from top
            # # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)   # absolute distance relative to origin
            
            v_fov_total = -v_fov[0] + v_fov[1]
            
            # if res_mode == 'resize-ratio1':
            #     const_dep = 0
            # elif res_mode == 'resize-ratio2':
            #     const_dep = pixel_res / ( np.tan(v_res * np.pi/180) )
            #     d_lidar = d_lidar + const_dep
            # elif res_mode == 'resize-ratio3':
            #     const_dep = pixel_res / ( np.tan(v_res * np.pi/180) )
            #     d_lidar = d_lidar + (const_dep * 2)

            # for i in range(x_lidar.shape[0]):
            #     rad_xy = math.atan2(y_lidar[i], x_lidar[i])
            #     x_lidar[i] = x_lidar[i] + const_dep * math.cos(rad_xy)
            #     y_lidar[i] = y_lidar[i] + const_dep * math.sin(rad_xy)
                
            v_res_rad = v_res * (np.pi/180)
            h_res_rad = h_res * (np.pi/180)
            
            x_img = np.arctan2(-y_lidar, x_lidar)/ h_res_rad
            y_img = np.arctan2(z_lidar, d_lidar)/ v_res_rad
            
            x_min = -360.0 / h_res / 2
            x_img -= x_min
            x_max = 360.0 / h_res
            
            y_min = v_fov[0] / v_res
            y_img -= y_min
            y_max = v_fov_total / v_res
            
            y_max += y_fudge
            
            if val == "reflectance":
                pass
            elif val == "height":
                pixel_values = z_lidar
            else:
                pixel_values = -d_lidar


"""
implementaion
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--implementation_mode', type=str, default='preprocessing')
    parser.add_argument('--sensor_name', type=str, default='vls-128')
    parser.add_argument('--projection_mode', type=str, default='ortho')
    parser.add_argument('--set_name', type=str, default='train')
    parser.add_argument('--res_const', type=float, default=0.040)
    parser.add_argument('--frame_const', default='all')
    parser.add_argument('--dataset_name', default='kugait30')
    # parser.add_argument('--input_folder', type=str, default='./datasets/kugait30/pcd_frame-20/detection/pvrcnn/pedestrian')
    # parser.add_argument('--output_folder', type=str, default='./datasets/kugait30/proj_frame-20_jeongho/pvrcnn')

    args = parser.parse_args()

    implementation_mode = args.implementation_mode
    sensor_name = args.sensor_name
    projection_mode = args.projection_mode
    set_name = args.set_name
    res_const = args.res_const
    frame_const = args.frame_const
    dataset_name = args.dataset_name

    assert implementation_mode in ['preprocessing', 'visualization'], f'Make sure your implemetaiton_mode: {implementation_mode}'
    assert sensor_name in ['vls-128', 'vlp-32'], f'Make sure your sensor_name: {sensor_name}'
    assert projection_mode in ['ortho', 'spher'], f'Make sure your projection_mode: {projection_mode}'
    assert set_name in ['train', 'test'], f'Make sure your set_name: {set_name}'
    assert res_const in [0.030, 0.035, 0.040], f'Make sure your res: {res_const}'
    assert frame_const in ['all', 10, 15, 20], f'Make sure your frame: {frame_const}'
    assert dataset_name in ['susteck1k', 'kugait30'], f'Make sure your dataset_name: {dataset_name}'

    if implementation_mode == 'preprocessing':

        if sensor_name == 'vls-128':
            HRES = 0.19188        # h res of VLS-128
            VRES = 0.2            # v res of VLS-128
            VFOV = (-25.0, 15.0)  # v fov of VLS-128
            Y_FUDGE = 0           # 0 if VLS-128
        elif sensor_name == 'vlp-32':
            HRES = 0.19188        # h res of VLP-32C
            VRES = 0.2            # v res of VLP-32C
            VFOV = (-25.0, 15.0)  # v fov of VLP-32C
            Y_FUDGE = 0.0         # 0 if VLP-32C
        else:
            raise ValueError(f'Unknown sensor name {sensor_name}')

        output_path = f'./datasets/{dataset_name}/projection/proj-{projection_mode}_depth-pos_drop-vpf_set-{set_name}_frame-{frame_const}_res-{res_const}_pkl'  # check here!
        source_path = './dataset/susteck/SUSTech1K-Released-2023'  # check here!
        json_path = './configs/susteck1k/SUSTech1K.json'  # check here !

        with open(json_path, 'r', encoding = 'utf-8') as f:
            json_file = json.load(f)
        
        if set_name == 'train':
            pid_list = json_file['TRAIN_SET']
        elif set_name == 'test':
            pid_list = json_file['TEST_SET']
        else:
            raise ValueError(f'Unknown set name {set_name}')
        
        pid_list = sorted(pid_list)
        print(f'len(pid_list): {len(pid_list)}')

        if dataset_name == 'susteck1k':
            for pid in pid_list:
                print(f'pid: {str(pid)}')

                pcd_folder_paths = sorted(glob.glob(os.path.join(source_path, pid, '*/*/PCDs/')))
                for pcd_folder_path in pcd_folder_paths:
                    pcd_file_paths = glob.glob(os.path.join(pcd_folder_path, '*.pcd'))

                    pid = pcd_folder_path.split('/')[-5]
                    var = pcd_folder_path.split('/')[-4]
                    ang = pcd_folder_path.split('/')[-3]

                    pkl_folder_path_clean = os.path.join(output_path, pid, f'{var}_clean', ang)
                    pkl_folder_path_v12p56f1010 = os.path.join(output_path, pid, f'{var}_v12p56f1010_orig', ang)
                    pkl_folder_path_v13p46f1010 = os.path.join(output_path, pid, f'{var}_v13p46f1010_orig', ang)
                    pkl_folder_path_v14p36f1010 = os.path.join(output_path, pid, f'{var}_v14p36f1010_orig', ang)
                    pkl_folder_path_v11p66f710 = os.path.join(output_path, pid, f'{var}_v11p66f710_orig', ang)
                    pkl_folder_path_v14p36f710 = os.path.join(output_path, pid, f'{var}_v14p36f710_orig', ang)

                    pkl_depth_file_path_clean = os.path.join(pkl_folder_path_clean, '01-000-LiDAR-PCDs_depths.pkl')
                    pkl_depth_file_path_v12p56f1010 = os.path.join(pkl_folder_path_v12p56f1010, '01-000-LiDAR-PCDs_depths.pkl')
                    pkl_depth_file_path_v13p46f1010 = os.path.join(pkl_folder_path_v13p46f1010, '01-000-LiDAR-PCDs_depths.pkl')
                    pkl_depth_file_path_v14p36f1010 = os.path.join(pkl_folder_path_v14p36f1010, '01-000-LiDAR-PCDs_depths.pkl')
                    pkl_depth_file_path_v11p66f710 = os.path.join(pkl_folder_path_v11p66f710, '01-000-LiDAR-PCDs_depths.pkl')
                    pkl_depth_file_path_v14p36f710 = os.path.join(pkl_folder_path_v14p36f710, '01-000-LiDAR-PCDs_depths.pkl')

                    pkl_pos_file_path_clean = os.path.join(pkl_folder_path_clean, '50-000-LiDAR-PCDs_pos.pkl')
                    pkl_pos_file_path_v12p56f1010 = os.path.join(pkl_folder_path_v12p56f1010, '50-000-LiDAR-PCDs_pos.pkl')
                    pkl_pos_file_path_v13p46f1010 = os.path.join(pkl_folder_path_v13p46f1010, '50-000-LiDAR-PCDs_pos.pkl')
                    pkl_pos_file_path_v14p36f1010 = os.path.join(pkl_folder_path_v14p36f1010, '50-000-LiDAR-PCDs_pos.pkl')
                    pkl_pos_file_path_v11p66f710 = os.path.join(pkl_folder_path_v11p66f710, '50-000-LiDAR-PCDs_pos.pkl')
                    pkl_pos_file_path_v14p36f710 = os.path.join(pkl_folder_path_v14p36f710, '50-000-LiDAR-PCDs_pos.pkl')

                    os.makedirs(pkl_folder_path_clean, exist_ok=True)
                    os.makedirs(pkl_folder_path_v12p56f1010, exist_ok=True)
                    os.makedirs(pkl_folder_path_v13p46f1010, exist_ok=True)
                    os.makedirs(pkl_folder_path_v14p36f1010, exist_ok=True)
                    os.makedirs(pkl_folder_path_v11p66f710, exist_ok=True)
                    os.makedirs(pkl_folder_path_v14p36f710, exist_ok=True)

                    points_seq = []
                    for pcd_file_path in sorted(pcd_file_paths):
                        pcd_data = o3d.io.read_point_cloud(pcd_file_path)
                        points = np.asarray(pcd_data.points)
                        points_seq.append(points)

                    lidar_to_dep_and_pos_for_susteck1k(points_seq = points_seq, 
                                                v_res = VRES, 
                                                h_res = HRES, 
                                                v_fov = VFOV, 
                                                val = 'depth', 
                                                y_fudge = Y_FUDGE,
                                                pkl_depth_file_path_clean = pkl_depth_file_path_clean, 
                                                pkl_depth_file_path_v12p56f1010 = pkl_depth_file_path_v12p56f1010, 
                                                pkl_depth_file_path_v13p46f1010 = pkl_depth_file_path_v13p46f1010, 
                                                pkl_depth_file_path_v14p36f1010 = pkl_depth_file_path_v14p36f1010, 
                                                pkl_depth_file_path_v11p66f710 = pkl_depth_file_path_v11p66f710, 
                                                pkl_depth_file_path_v14p36f710 = pkl_depth_file_path_v14p36f710, 
                                                pkl_pos_file_path_clean = pkl_pos_file_path_clean, 
                                                pkl_pos_file_path_v12p56f1010 = pkl_pos_file_path_v12p56f1010, 
                                                pkl_pos_file_path_v13p46f1010 = pkl_pos_file_path_v13p46f1010, 
                                                pkl_pos_file_path_v14p36f1010 = pkl_pos_file_path_v14p36f1010, 
                                                pkl_pos_file_path_v11p66f710 = pkl_pos_file_path_v11p66f710, 
                                                pkl_pos_file_path_v14p36f710 = pkl_pos_file_path_v14p36f710, 
                                                proj_mode = projection_mode, 
                                                depth_norm_mode = True, 
                                                set_name = set_name, 
                                                res_const = res_const, 
                                                frame_const = frame_const, 
                                                )
        # elif dataset_name == 'kugait30':
        #     for dist in ['10m', '20m', '30m']:
                
        #         for pid in pid_list:
        #             print(f'dist_pid: {dist}_{pid}')

        #             npy_folder_paths = os.path.join(source_path, dist, pid, '*/*')
        #             npy_folder_paths = glob.glob(npy_folder_paths)

        #             for npy_folder_path in sorted(npy_folder_paths):
        #                 npy_file_paths = sorted(glob.glob(os.path.join(npy_folder_path, '*.npy')))

        #                 dist = npy_folder_path.split('/')[-4]
        #                 pid = npy_folder_path.split('/')[-3]
        #                 var = npy_folder_path.split('/')[-2]
        #                 ang = npy_folder_path.split('/')[-1]

        #                 pkl_folder_path = os.path.join(output_path, pid, str(f'{var}-{dist}'), ang) 
        #                 pkl_depth_file_path = os.path.join(pkl_folder_path, '01-000-LiDAR-PCDs_depths.pkl')
        #                 pkl_pos_file_path = os.path.join(pkl_folder_path, '50-000-LiDAR-PCDs_pos.pkl')
        #                 os.makedirs(pkl_folder_path, exist_ok=True)

        #                 points_seq = []
        #                 for npy_file_path in sorted(npy_file_paths):
        #                     points = np.load(npy_file_path)
        #                     points = points[:, :3]
        #                     points_seq.append(points)

        #                 lidar_to_dep_and_pos_for_kugait30(points_seq=points_seq, 
        #                                             v_res=VRES, 
        #                                             h_res=HRES, 
        #                                             v_fov=VFOV, 
        #                                             val='depth', 
        #                                             y_fudge=Y_FUDGE,
        #                                             pkl_depth_file_path=pkl_depth_file_path, 
        #                                             pkl_pos_file_path=pkl_pos_file_path, 
        #                                             proj_mode=projection_mode, 
        #                                             depth_norm_mode=True, 
        #                                             set_name=set_name, 
        #                                             res_const = res_const, 
        #                                             frame_const = frame_const, 
        #                                             )
        else:
            raise ValueError(f'Unknown dataset name {dataset_name}')
    elif implementation_mode == 'visualization':
        pid_check = '0002'
        var_check = '00-nm'
        ang_check = '225'
        
        pkl_to_gif(
            pkl_path = f"./datasets/susteck1k/projection/proj-{projection_mode}_depth-pos_drop-vpf_set-{set_name}_frame-{frame_const}_res-{res_const}_pkl/{pid_check}/{var_check}_clean/{ang_check}/01-000-LiDAR-PCDs_depths.pkl", 
            gif_path = f'./checking_res-{res_const}_{pid_check}_{var_check}_{ang_check}_clean.gif'
            )

        pkl_to_gif(
            pkl_path = f"./datasets/susteck1k/projection/proj-{projection_mode}_depth-pos_drop-vpf_set-{set_name}_frame-{frame_const}_res-{res_const}_pkl/{pid_check}/{var_check}_v12p56f1010_orig/{ang_check}/01-000-LiDAR-PCDs_depths.pkl", 
            gif_path = f'./checking_res-{res_const}_{pid_check}_{var_check}_{ang_check}_v12p56f1010_orig.gif'
            )
        
        pkl_to_gif(
            pkl_path = f"./datasets/susteck1k/projection/proj-{projection_mode}_depth-pos_drop-vpf_set-{set_name}_frame-{frame_const}_res-{res_const}_pkl/{pid_check}/{var_check}_v13p46f1010_orig/{ang_check}/01-000-LiDAR-PCDs_depths.pkl", 
            gif_path = f'./checking_res-{res_const}_{pid_check}_{var_check}_{ang_check}_v13p46f1010_orig.gif'
            )
        
        pkl_to_gif(
            pkl_path = f"./datasets/susteck1k/projection/proj-{projection_mode}_depth-pos_drop-vpf_set-{set_name}_frame-{frame_const}_res-{res_const}_pkl/{pid_check}/{var_check}_v14p36f1010_orig/{ang_check}/01-000-LiDAR-PCDs_depths.pkl", 
            gif_path = f'./checking_res-{res_const}_{pid_check}_{var_check}_{ang_check}_v14p36f1010_orig.gif'
            )

        pkl_to_gif(
            pkl_path = f"./datasets/susteck1k/projection/proj-{projection_mode}_depth-pos_drop-vpf_set-{set_name}_frame-{frame_const}_res-{res_const}_pkl/{pid_check}/{var_check}_v11p66f710_orig/{ang_check}/01-000-LiDAR-PCDs_depths.pkl", 
            gif_path = f'./checking_res-{res_const}_{pid_check}_{var_check}_{ang_check}_v11p66f710_orig.gif'
            )

        pkl_to_gif(
            pkl_path = f"./datasets/susteck1k/projection/proj-{projection_mode}_depth-pos_drop-vpf_set-{set_name}_frame-{frame_const}_res-{res_const}_pkl/{pid_check}/{var_check}_v14p36f710_orig/{ang_check}/01-000-LiDAR-PCDs_depths.pkl", 
            gif_path = f'./checking_res-{res_const}_{pid_check}_{var_check}_{ang_check}_v14p36f710_orig.gif'
            )
    else:
        raise ValueError(f'Unknown implementation mode {implementation_mode}')