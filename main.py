import argparse
import sys
import os
import warnings
import json
import pickle
import numpy as np
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import cv2

import scipy.signal
import scipy.ndimage
from scipy.ndimage import zoom
from scipy.interpolate import griddata

import torch
import torch.multiprocessing as mp  # debug
from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP  # debug
from torchvision import transforms as T, utils

import einops
from einops import rearrange

from lidargsu.diffusion_pytorch import ContinuousTimeGaussianDiffusion, ContinuousTimeSampler, Trainer  # debug
from lidargsu.models.nn import *
from lidargsu.models.unet import UNet
from lidargsu.models.unet3d import UNet3D
from lidargsu.utils.common import *


# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='./datasets/susteck1k/projection/ortho_depth-pos_drop-point_pkl')
parser.add_argument('--pretrained_path', type=str, default='./model_iter200000.pt')
parser.add_argument('--json_path', type=str, default='configs/susteck1k/susteck1k.json')
parser.add_argument('--dataset', type=str, default='kugait30')  # susteck1k | kugait30
parser.add_argument('--mode', type=str, default='test')  # train | test
parser.add_argument('--rank', type=str, default='cuda:0')
parser.add_argument('--num_sample_steps', type=int, default=32)
opt = parser.parse_args()

train_path = opt.train_path
pretrained_path = opt.pretrained_path
json_path = opt.json_path
rank = opt.rank
num_sample_steps = opt.num_sample_steps
dataset = opt.dataset
mode = opt.mode

assert opt.rank in ['cuda:0', 'cuda:1'], f'check the opt.rank {opt.rank}'
assert opt.num_sample_steps in [1, 2, 4, 8, 16, 32, 64, 128], f'check the opt.num_sample_steps {opt.num_sample_steps}'
assert opt.dataset in ['susteck1k', 'kugait30'], f'check the opt.dataset {opt.dataset}'
assert opt.mode in ['train', 'test'], f'check the opt.mode {opt.mode}'


# Set a denoiser for LidarGSU
denoiser = UNet3D(
    dim = 64, 
    cond_dim = None, 
    out_dim = 1, 
    dim_mults = (1, 2, 4, 8), 
    channels = 2, 
    attn_heads = 8, 
    attn_dim_head = 32, 
    use_bert_text_cond = False, 
    init_dim = None, 
    init_kernel_size = 7, 
    use_sparse_linear_attn = True, 
    block_type = 'resnet'
)


# Set a diffusion model with discrete time
diffusion_model = ContinuousTimeGaussianDiffusion(
    denoise_fn = denoiser, 
    image_size = 64,                   # default: 64
    channels = 2,                      # default: 2(depth) | \in [2, 6]
    frames = 10,                       # default: 10
    loss_type = 'l2',                  # default: l2 | \in [l1, l2]
    use_dynamic_thres = False,         # default: False
    dynamic_thres_percentile = 0.9,    # default: 0.9
    rank = rank,
    )


if mode == 'train':
    trainer = Trainer(
        diffusion_model = diffusion_model, 
        train_folder_path = train_path, 
        modelzoo_file_path = None, 
        train_batch_size = 8,              # default: 8 
        train_lr = 1e-4,                   # default: 1e-4
        sampler = 'ddpm',                  # \in {'ddpm', 'ddim'}
        num_sample_steps = 128,              # default: 2, 4, 8, 16, 32, 64, 128
        save_and_sample_every = 5000,      # default: 5000 
        train_num_steps = 500000,          # default: 400000 | \in [400000, 1000000]
        gradient_accumulate_every = 2,     # default: 2
        ema_decay = 0.995,                 # default: 0.995
        amp = True,                        # default: True
        step_start_ema = 2000,             # default: 2000
        update_ema_every = 10,             # default: 10
        results_folder = './results',
        num_sample_rows = 4,               # default: 4
        max_grad_norm = 1.,                # default: None
        rank = rank
        )
elif mode == 'test':
    continuous_time_sampler = ContinuousTimeSampler(
        diffusion_model = diffusion_model, 
        modelzoo_file_path = pretrained_path, 
        rank = rank, 
        sampler_type = 'ddpm', 
        num_sample_steps = num_sample_steps, 
        )


if dataset == 'susteck1k' and mode == 'test':
    class DatasetSusteck1kTest(data.Dataset):
        def __init__(
            self, 
            folder_cond=f'./outputs/susteck1k/lidargsu_masking-yes_sampler-ddpm{num_sample_steps}',  # check
            json_path=json_path, 
            set_name='TEST_SET',
            num_frames=10,          # check
            pid_start_const=0,      # check
            pid_end_const=800,      # check
        ):
            super().__init__()
            self.folder_cond = folder_cond
            self.num_frames = num_frames

            self.cond_np_p50_paths = list() 
            self.cond_np_p33_paths = list() 
            self.cond_np_p25_paths = list() 
            self.output_np_p50_pkl_paths = list()        
            self.output_np_p33_pkl_paths = list()   
            self.output_np_p25_pkl_paths = list()          

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert set_name in ['TRAIN_SET', 'TEST_SET'], "Check your set_name!"
            
            pid_list = sorted(data[f'{set_name}'])
            pid_list = pid_list[pid_start_const:pid_end_const]
            print(pid_list)
            print(f"len(pid_list): {len(pid_list)}")


            for pid in pid_list:
                print(f'pid: {pid}')

                var_list = sorted(os.listdir(os.path.join(self.folder_cond, pid)))  # debug
                var_list_filtered = [var for var in var_list if not any(word in var for word in ['00-nm', 'clean', 'restoration'])]  # debug
                for var in var_list_filtered:   

                    ang_list = sorted(os.listdir(os.path.join(self.folder_cond, pid, var)))
                    for ang in ang_list:
                        if 'p50' in var:
                            cond_np_p50_pkl = os.path.join(self.folder_cond, pid, var, ang, '01-000-LiDAR-PCDs_depths.pkl')
                            output_np_p50_pkl = os.path.join(self.folder_cond, pid, var.replace('_orig', '_restoration'), ang, '01-000-LiDAR-PCDs_depths.pkl')
                            os.makedirs(os.path.join(self.folder_cond, pid, var.replace('_orig', '_restoration'), ang), exist_ok=True)
                            with open(cond_np_p50_pkl, 'rb') as f:
                                cond_np_p50 = pickle.load(f)
                            self.cond_np_p50_paths.append(cond_np_p50)
                            self.output_np_p50_pkl_paths.append(output_np_p50_pkl)
                        elif 'p33' in var:
                            cond_np_p33_pkl = os.path.join(self.folder_cond, pid, var, ang, '01-000-LiDAR-PCDs_depths.pkl')
                            output_np_p33_pkl = os.path.join(self.folder_cond, pid, var.replace('_orig', '_restoration'), ang, '01-000-LiDAR-PCDs_depths.pkl')
                            os.makedirs(os.path.join(self.folder_cond, pid, var.replace('_orig', '_restoration'), ang), exist_ok=True)
                            with open(cond_np_p33_pkl, 'rb') as f:
                                cond_np_p33 = pickle.load(f)
                            self.cond_np_p33_paths.append(cond_np_p33)
                            self.output_np_p33_pkl_paths.append(output_np_p33_pkl)
                        elif 'p25' in var:
                            cond_np_p25_pkl = os.path.join(self.folder_cond, pid, var, ang, '01-000-LiDAR-PCDs_depths.pkl')
                            output_np_p25_pkl = os.path.join(self.folder_cond, pid, var.replace('_orig', '_restoration'), ang, '01-000-LiDAR-PCDs_depths.pkl')
                            os.makedirs(os.path.join(self.folder_cond, pid, var.replace('_orig', '_restoration'), ang), exist_ok=True)
                            with open(cond_np_p25_pkl, 'rb') as f:
                                cond_np_p25 = pickle.load(f)
                            self.cond_np_p25_paths.append(cond_np_p25)
                            self.output_np_p25_pkl_paths.append(output_np_p25_pkl)
                        else:
                            raise ValueError(f"Check your SUSTeck1K dataset.")

        def __len__(self):
            return len(self.cond_np_p50_paths)

        def __getitem__(self, item):
            cond_np_p50 = self.cond_np_p50_paths[item]
            cond_np_p33 = self.cond_np_p33_paths[item]
            cond_np_p25 = self.cond_np_p25_paths[item]

            output_np_p50_pkl = self.output_np_p50_pkl_paths[item]
            output_np_p33_pkl = self.output_np_p33_pkl_paths[item]
            output_np_p25_pkl = self.output_np_p25_pkl_paths[item]

            cond_tensor_p50 = torch.from_numpy(cond_np_p50).clone()
            cond_tensor_p33 = torch.from_numpy(cond_np_p33).clone()
            cond_tensor_p25 = torch.from_numpy(cond_np_p25).clone()
            
            return cond_tensor_p50, cond_tensor_p33, cond_tensor_p25, output_np_p50_pkl, output_np_p33_pkl, output_np_p25_pkl
elif dataset == 'kugait30' and mode == 'test':
    class DatasetKUGait30Test(data.Dataset):
        def __init__(
            self, 
            folder_cond = f'./outputs/kugait30/lidargsu_masking-yes_sampler-ddpm{num_sample_steps}',  # check
            num_frames=10,          # check
            pid_start_const=0,      # check
            pid_end_const=30,       # check
        ):
            super().__init__()
            self.folder_cond = folder_cond
            self.num_frames = num_frames

            self.cond_np_10m_paths = list()
            self.cond_np_20m_paths = list()

            self.output_np_10m_pkl_paths = list()
            self.output_np_20m_pkl_paths = list()

            pid_list = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020', '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029']
            pid_list = pid_list[pid_start_const:pid_end_const]
            print(pid_list)
            print(f"len(pid_list): {len(pid_list)}")

            for pid in pid_list:
                print(f'pid: {pid}')

                var_list = sorted(os.listdir(os.path.join(self.folder_cond, pid)))  # debug
                var_list_filtered = [var for var in var_list if not any(word in var for word in ['_restoration'])]  # debug
                for var in var_list_filtered:   

                    ang_list = sorted(os.listdir(os.path.join(self.folder_cond, pid, var)))
                    for ang in ang_list:

                        if '10m' in var:
                            cond_np_10m_pkl = os.path.join(self.folder_cond, pid, var, ang, '01-000-LiDAR-PCDs_depths.pkl')
                            output_np_10m_pkl = os.path.join(self.folder_cond, pid, var.replace('_orig', '_restoration'), ang, '01-000-LiDAR-PCDs_depths.pkl')
                            os.makedirs(os.path.join(self.folder_cond, pid, var.replace('_orig', '_restoration'), ang), exist_ok=True)
                            with open(cond_np_10m_pkl, 'rb') as f:
                                cond_np_10m = pickle.load(f)
                            self.cond_np_10m_paths.append(cond_np_10m)
                            self.output_np_10m_pkl_paths.append(output_np_10m_pkl)
                        elif '20m' in var:
                            cond_np_20m_pkl = os.path.join(self.folder_cond, pid, var, ang, '01-000-LiDAR-PCDs_depths.pkl')
                            output_np_20m_pkl = os.path.join(self.folder_cond, pid, var.replace('_orig', '_restoration'), ang, '01-000-LiDAR-PCDs_depths.pkl')
                            os.makedirs(os.path.join(self.folder_cond, pid, var.replace('_orig', '_restoration'), ang), exist_ok=True)
                            with open(cond_np_20m_pkl, 'rb') as f:
                                cond_np_20m = pickle.load(f)
                            self.cond_np_20m_paths.append(cond_np_20m)
                            self.output_np_20m_pkl_paths.append(output_np_20m_pkl)
                        else:
                            raise ValueError(f"Check your KUGait30 dataset.")

        def __len__(self):
            return len(self.cond_np_10m_paths)

        def __getitem__(self, item):
            cond_np_10m = self.cond_np_10m_paths[item]
            cond_np_20m = self.cond_np_20m_paths[item]

            output_np_10m_pkl = self.output_np_10m_pkl_paths[item]
            output_np_20m_pkl = self.output_np_20m_pkl_paths[item]

            cond_tensor_10m = torch.from_numpy(cond_np_10m).clone()
            cond_tensor_20m = torch.from_numpy(cond_np_20m).clone()
            
            return cond_tensor_10m, cond_tensor_20m, output_np_10m_pkl, output_np_20m_pkl


# implementation
if __name__ == '__main__':
    if mode == 'train':
        trainer.load()
        trainer.train()
    elif mode == 'test':
        
        batch_size = 1
        continuous_time_sampler.load()

        if dataset == 'susteck1k':
            ds = DatasetSusteck1kTest() 
        elif dataset == 'kugait30':
            ds = DatasetKUGait30Test()

        dl = iter(data.DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True))

        step = 0
        total_steps = len(ds)

        while step < total_steps: 
            print(f'step: {step}/{total_steps}, num_sample_steps: {num_sample_steps}')

            if opt.dataset == 'susteck1k':
                cond_tensor_p050, cond_tensor_p033, cond_tensor_p025, output_np_p050_pkl, output_np_p033_pkl, output_np_p025_pkl = next(dl)

                continuous_time_sampler.sample(input_depth=cond_tensor_p050, output_pkl=output_np_p050_pkl)
                continuous_time_sampler.sample(input_depth=cond_tensor_p033, output_pkl=output_np_p033_pkl)
                continuous_time_sampler.sample(input_depth=cond_tensor_p025, output_pkl=output_np_p025_pkl)
            elif opt.dataset == 'kugait30':
                cond_tensor_10m, cond_tensor_20m, output_np_10m_pkl, output_np_20m_pkl = next(dl)

                continuous_time_sampler.sample(input_depth=cond_tensor_10m, output_pkl=output_np_10m_pkl)
                continuous_time_sampler.sample(input_depth=cond_tensor_20m, output_pkl=output_np_20m_pkl)

            step += 1