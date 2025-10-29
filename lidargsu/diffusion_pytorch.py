import math
import copy
from functools import partial
from pathlib import Path
import json
import pickle
import os
from glob import glob
import numpy as np
from collections import namedtuple
from inspect import isfunction

import scipy
import cv2
import matplotlib.cm as cm

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils import data
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.special import expm1        # debug
from torch.cuda.amp import autocast    # debug
from torch import sqrt                 # debug

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import einops    # debug
from einops.layers.torch import Rearrange    # debug
from einops import rearrange, reduce, repeat
from einops_exts import check_shape, rearrange_many
# from accelerate import Accelerator
import deepinv

from utils.common import *


# Alpha-cosine schedule for continuous time
def alpha_cosine_log_snr(t, s = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5)


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


# Cosine beta schedule for discrete time
def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


# EMA
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# Gaussian diffusion for continuous time
class ContinuousTimeGaussianDiffusion(nn.Module):
    def __init__(
        self, 
        denoise_fn, 
        rank, 
        *, 
        image_size, 
        channels = 2,                     # default: 2 when using a depth channel
        frames = 10, 
        noise_schedule = 'cosine', 
        loss_type = 'l2', 
        use_dynamic_thres = False,        # taken from the Imagen paper
        dynamic_thres_percentile = 0.9, 
        clip_sample_denoised = True, 
        min_snr_loss_weight = False, 
        min_snr_gamma = 5, 
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.rank = rank

        self.image_size = image_size
        self.channels = channels
        self.frames = frames
        self.loss_type = loss_type

        self.clip_sample_denoised = clip_sample_denoised

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.rank)

        if noise_schedule == 'cosine':
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'unknown noise schedule {noise_schedule}')


    @torch.no_grad()
    def p_sample(self, x, x_cond, time, time_next, sampler):
        # batch, *_, device = *x.shape, x.device
        batch, *_ = x.shape     # debug

        log_snr_t = self.log_snr(time)
        log_snr_s = self.log_snr(time_next)

        squared_alpha_t, squared_alpha_s = log_snr_t.sigmoid(), log_snr_s.sigmoid()
        squared_sigma_t, squared_sigma_s = (-log_snr_t).sigmoid(), (-log_snr_s).sigmoid()

        alpha_t, sigma_t, alpha_s, sigma_s = map(sqrt, (squared_alpha_t, squared_sigma_t, squared_alpha_s, squared_sigma_s))    # debug

        batch_log_snr = repeat(log_snr_t, ' -> b', b = x.shape[0])
        pred_noise = self.denoise_fn(torch.cat([x_cond, x], dim=1), batch_log_snr)
        x_0 = (x - sigma_t * pred_noise) / alpha_t

        if self.clip_sample_denoised:
            x_0.clamp_(-1., 1.)    # in Imagen, it will be changed to dynamic thresholding

        if sampler == 'ddpm': 
            c = -expm1(log_snr_t - log_snr_s)
            mean = alpha_s * (x * (1 - c) / alpha_t + c * x_0)
            var = squared_sigma_s * c
            var_noise = torch.randn_like(x)
            if time_next == 0:
                return mean
            x_s = mean + var.sqrt() * var_noise
        elif sampler == 'ddim':
            noise = (x - alpha_t * x_0) / sigma_t.clamp(min=1e-8)
            x_s = alpha_s * x_0 + sigma_s * noise
        else:
            raise ValueError(f'invalid sampler {sampler}')

        return x_s


    @torch.no_grad()
    def p_sample_loop(self, x_cond, num_sample_steps, sampler):
        batch, *_ = x_cond.shape
        mask = x_cond == 0.    # debug
        x_cond = normalize(x_cond)

        x = torch.randn_like(x_cond, device = self.rank)
        x = x * mask.float() + x_cond * (1. - mask.float())   # debug 

        x_list = list()    # debug

        steps = torch.linspace(1., 0., num_sample_steps + 1, device = self.rank)

        for i in tqdm(range(num_sample_steps), desc = 'sampling loop time step', total = num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            x = self.p_sample(x = x, x_cond = x_cond, time = times, time_next = times_next, sampler = sampler)
            x = x * mask.float() + x_cond * (1. - mask.float())   # debug 
            x_list.append(x)

        # x.clamp_(-1., 1.)
        x = unnormalize(x)
        x_list.append(x)
        
        return x, x_list


    @torch.no_grad()
    def sample(self, x_cond, num_sample_steps, sampler):
        return self.p_sample_loop(x_cond = x_cond, num_sample_steps = num_sample_steps, sampler = sampler)    # debug


    # Training related functions - noise prediction
    # @autocast(enabled = False)    # debug
    def q_sample(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr


    def random_times(self, batch_size):
        # batch_size, *_ = x_cond.shape
        # times are now uniform from 0 to 1
        return torch.zeros((batch_size,), device = self.rank).float().uniform_(0, 1)

    
    def p_losses(self, x_start, x_cond, times, mask, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)
        x = x * mask.float() + x_cond * ( 1. - mask.float() )                    # debug | initialize with known_area(x_cond)

        noise_hat = self.denoise_fn(torch.cat([x_cond, x], dim=1), log_snr)

        if self.loss_type == 'l1':
            losses = F.l1_loss(noise_hat, noise, reduction = 'none')
        elif self.loss_type == 'l2':
            losses = F.mse_loss(noise_hat, noise, reduction = 'none')
        else:
            raise NotImplementedError()

        losses_masked = einops.reduce(losses * mask, 'b ... -> b ()', 'sum')    # debug
        mask = einops.reduce(mask, 'b ... -> b ()', 'sum')                      # debug
        losses_masked = losses_masked / mask.add(1e-8)                          # debug

        if self.min_snr_loss_weight:
            snr = log_snr.exp()
            loss_weight = snr.clamp(min = self.min_snr_gamma) / snr             # debug | when objective is eps(noise)
            losses_masked = ( losses_masked * loss_weight ).mean()

        return losses_masked


    def forward(self, x_0, x_cond, *args, **kwargs):
        # b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        b, *_ = x_0.shape    ## debug

        times = self.random_times(b)

        mask = x_cond == 0.    # debug
        x_0 = normalize(x_0)
        x_cond = normalize(x_cond)

        return self.p_losses(x_start = x_0, x_cond = x_cond, times = times, mask = mask, *args, **kwargs)



""" 
Dataset for self-occlusion restoration
"""
class DatasetSusteck1kOcclusion(data.Dataset):
    def __init__(
        self, 
        dataset, 
        image_size = 64, 
        channels = 1, 
        frames = 10, 
        cfg_path='../configs/susteck1k/susteck1k.json', 
    ):
        super().__init__()
        self.dataset = Path(dataset)
        self.image_size = image_size
        self.channels = channels
        self.frames = frames
        self.cfg_path = cfg_path
        
        self.video_np_list = list()
        
        # self.transform = T.Compose([
        #     T.Resize(self.image_size), 
        #     T.RandomHorizontalFlip() if self.horizontal_flip else torch.nn.Identity(),   # debug
        #     T.CenterCrop(self.image_size), 
        #     T.ToTensor(), 
        #     T.Lambda(lambda x: ((x * 2) - 1)) if min1to1 else torch.nn.Identity()        # debug
        # ])
        
        with open(self.cfg_path, 'r', encoding='utf-8') as f:
            data_cfg = json.load(f)
            data_set = set(data_cfg['TRAIN_SET'])
            data_set = sorted(list(data_set))
            
            print(f'len(data_set): {len(data_set)}')

            for pid in data_set:
                print(f'pid: {pid}')
                
                var_list = sorted(os.listdir(os.path.join(self.dataset, pid)))
                for var in var_list:

                    ang_list = sorted(os.listdir(os.path.join(self.dataset, pid, var)))
                    # ang_list = ['90', '180']
                    for ang in ang_list:

                        video_np_path_orig = os.path.join(self.dataset, pid, var, ang, '01-000-LiDAR-PCDs_depths_orig.pkl')
                        video_np_path_p050 = os.path.join(self.dataset, pid, var, ang, '41-000-LiDAR-PCDs_depths_p050.pkl')
                        video_np_path_p033 = os.path.join(self.dataset, pid, var, ang, '42-000-LiDAR-PCDs_depths_p033.pkl')
                        video_np_path_p025 = os.path.join(self.dataset, pid, var, ang, '43-000-LiDAR-PCDs_depths_p025.pkl')

                        if not os.path.isfile(video_np_path_orig):
                            continue

                        with open(video_np_path_orig, 'rb') as f:
                            video_np_orig = pickle.load(f)    ## (F, C, H, W)

                        with open(video_np_path_p050, 'rb') as f:
                            video_np_p050 = pickle.load(f)

                        with open(video_np_path_p033, 'rb') as f:
                            video_np_p033 = pickle.load(f)

                        with open(video_np_path_p025, 'rb') as f:
                            video_np_p025 = pickle.load(f)

                        if video_np_orig.shape[0] < 1:
                            continue

                        video_np_orig = rearrange(video_np_orig, 'f c h w -> c f h w')
                        video_np_p050 = rearrange(video_np_p050, 'f c h w -> c f h w')
                        video_np_p033 = rearrange(video_np_p033, 'f c h w -> c f h w')
                        video_np_p025 = rearrange(video_np_p025, 'f c h w -> c f h w')

                        # option 3
                        i_const = 5    # debug
                        for i in range((video_np_orig.shape[1] - self.frames + 1) // i_const):
                            f = i * i_const
                            video_np_orig_temp = video_np_orig[:, f:(f+self.frames), :, :]
                            video_np_p050_temp = video_np_p050[:, f:(f+self.frames), :, :]
                            video_np_p033_temp = video_np_p033[:, f:(f+self.frames), :, :]
                            video_np_p025_temp = video_np_p025[:, f:(f+self.frames), :, :]

                            pair_p050 = np.concatenate((video_np_p050_temp, video_np_orig_temp), axis=0)
                            pair_p033 = np.concatenate((video_np_p033_temp, video_np_orig_temp), axis=0)
                            pair_p025 = np.concatenate((video_np_p025_temp, video_np_orig_temp), axis=0)

                            self.video_np_list.append(pair_p050)
                            self.video_np_list.append(pair_p033)
                            self.video_np_list.append(pair_p025)
    
    def __len__(self):
        return len(self.video_np_list)
    
    def __getitem__(self, item):
        video_np = self.video_np_list[item]
        video_tensor = (torch.from_numpy(video_np)).to(torch.float32)
        return video_tensor


""" 
Trainer class 
"""
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        train_folder_path,
        modelzoo_file_path, 
        rank, 
        *,
        ema_decay=0.995,
        train_batch_size=8,
        train_lr=1e-4,
        sampler = 'ddpm', 
        num_sample_steps = 128,
        train_num_steps=400000,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=5000,
        gradient_accumulate_every=2, 
        results_folder='./results',
        num_sample_rows=4,
        num_samples=9,
        max_grad_norm=1.
    ):
        super().__init__()
        self.rank = rank                                           # debug
        self.diffusion_model = diffusion_model.to(self.rank)       # debug
        self.train_folder_path = Path(train_folder_path)

        if modelzoo_file_path is not None:                         # debug
            self.modelzoo_file_path = Path(modelzoo_file_path)     # debug

        self.ema = EMA(ema_decay)
        self.ema_diffusion_model = copy.deepcopy(self.diffusion_model)
        self.update_ema_every = update_ema_every

        self.sampler = sampler
        self.num_sample_steps = num_sample_steps

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.train_batch_size = train_batch_size
        self.image_size = self.diffusion_model.image_size
        self.train_num_steps = train_num_steps

        self.image_size = self.diffusion_model.image_size
        self.channels = self.diffusion_model.channels
        self.frames = self.diffusion_model.frames

        # self.ds = DatasetSusteck1k(dataset=train_folder_path, image_size=image_size, channels=channels)
        self.ds = DatasetSusteck1kOcclusion(dataset=train_folder_path, image_size=self.image_size, channels=self.channels, frames=self.frames)    # debug

        print(f'found {len(self.ds)} images as png files at {train_folder_path}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'
        
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(self.diffusion_model.parameters(), lr = train_lr)    # debug

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulate_every = gradient_accumulate_every

        self.num_samples = num_samples
        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()


    def reset_parameters(self):
        self.ema_diffusion_model.load_state_dict(self.diffusion_model.state_dict())


    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_diffusion_model, self.diffusion_model)


    def save(self, x):
        data = {
            'step': self.step,
            'model': self.diffusion_model.state_dict(),
            'ema': self.ema_diffusion_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model_iter{x}.pt'))


    def load(self, **kwargs):
        data = torch.load(f'{self.modelzoo_file_path}')
        
        self.step = data['step']
        self.diffusion_model.load_state_dict(data['model'], **kwargs)
        self.ema_diffusion_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])


    def train(self):  
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                
                video_tensor = next(self.dl)
                with autocast(enabled = self.amp):

                    y_0 = video_tensor[:, 1, :, :, :].unsqueeze(dim=1)
                    y_cond = video_tensor[:, 0, :, :, :].unsqueeze(dim=1)

                    loss = self.diffusion_model(x_0 = y_0.to(self.rank), x_cond = y_cond.to(self.rank))
                    loss = loss.sum()
                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()    ## original

                print(f'{self.step}: {loss.item()}')

            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                """ option3 (super-res) """
                y0_orig_path = './datasets/susteck1k_projection/ortho_depth-pos_drop-point_pkl/0001/00-nm/090/01-000-LiDAR-PCDs_depths_orig.pkl'
                y0_cond_p050_path = './datasets/susteck1k_projection/ortho_depth-pos_drop-point_pkl/0001/00-nm/090/41-000-LiDAR-PCDs_depths_p050.pkl'
                y0_cond_p033_path = './datasets/susteck1k_projection/ortho_depth-pos_drop-point_pkl/0001/00-nm/090/42-000-LiDAR-PCDs_depths_p033.pkl'
                y0_cond_p025_path = './datasets/susteck1k_projection/ortho_depth-pos_drop-point_pkl/0001/00-nm/090/43-000-LiDAR-PCDs_depths_p025.pkl'

                with open(y0_orig_path, 'rb') as f:
                    y0_orig = pickle.load(f)
                
                with open(y0_cond_p050_path, 'rb') as f:
                    y0_cond_p050 = pickle.load(f)

                with open(y0_cond_p033_path, 'rb') as f:
                    y0_cond_p033 = pickle.load(f)

                with open(y0_cond_p025_path, 'rb') as f:
                    y0_cond_p025 = pickle.load(f)

                y0_cond_p050_tensor = torch.from_numpy(y0_cond_p050).to(torch.float32).to(self.rank)  
                y0_cond_p050_tensor = y0_cond_p050_tensor[:self.frames, :, :, :]
                y0_cond_p050_tensor = rearrange(y0_cond_p050_tensor, "f c h w -> c f h w")
                y0_cond_p050_tensor = y0_cond_p050_tensor.unsqueeze(0)

                y0_cond_p033_tensor = torch.from_numpy(y0_cond_p033).to(torch.float32).to(self.rank)  
                y0_cond_p033_tensor = y0_cond_p033_tensor[:self.frames, :, :, :]
                y0_cond_p033_tensor = rearrange(y0_cond_p033_tensor, "f c h w -> c f h w")
                y0_cond_p033_tensor = y0_cond_p033_tensor.unsqueeze(0)

                y0_cond_p025_tensor = torch.from_numpy(y0_cond_p025).to(torch.float32).to(self.rank)  
                y0_cond_p025_tensor = y0_cond_p025_tensor[:self.frames, :, :, :]
                y0_cond_p025_tensor = rearrange(y0_cond_p025_tensor, "f c h w -> c f h w")
                y0_cond_p025_tensor = y0_cond_p025_tensor.unsqueeze(0)
                
                y0_restored_p050_tensor = self.ema_diffusion_model.sample(x_cond=y0_cond_p050_tensor, num_sample_steps = self.num_sample_steps, sampler = self.sampler)
                y0_restored_p050 = y0_restored_p050_tensor.cpu().numpy()

                y0_restored_p033_tensor = self.ema_diffusion_model.sample(x_cond=y0_cond_p033_tensor, num_sample_steps = self.num_sample_steps, sampler = self.sampler)
                y0_restored_p033 = y0_restored_p033_tensor.cpu().numpy()

                y0_restored_p025_tensor = self.ema_diffusion_model.sample(x_cond=y0_cond_p025_tensor, num_sample_steps = self.num_sample_steps, sampler = self.sampler)
                y0_restored_p025 = y0_restored_p025_tensor.cpu().numpy()

                print(y0_restored_p050.max())
                print(y0_restored_p050.min())
                print(y0_restored_p033.max())
                print(y0_restored_p033.min())
                print(y0_restored_p025.max())
                print(y0_restored_p025.min())

                thres_max = 0.9
                thres_min = 0.1

                y0_restored_p050_refined = copy.deepcopy(y0_restored_p050)
                y0_restored_p050_refined[y0_restored_p050_refined <= thres_min] = 0.
                y0_restored_p050_refined[y0_restored_p050_refined >= thres_max] = 0.

                y0_restored_p033_refined = copy.deepcopy(y0_restored_p033)
                y0_restored_p033_refined[y0_restored_p033_refined <= thres_min] = 0.
                y0_restored_p033_refined[y0_restored_p033_refined >= thres_max] = 0.

                y0_restored_p025_refined = copy.deepcopy(y0_restored_p025)
                y0_restored_p025_refined[y0_restored_p025_refined <= thres_min] = 0.
                y0_restored_p025_refined[y0_restored_p025_refined >= thres_max] = 0.

                y0_orig_temp = (torch.from_numpy(y0_orig))
                y0_orig_temp = rearrange(y0_orig_temp, "f c h w -> 1 c f h w")
                
                y0_cond_p050_temp = (torch.from_numpy(y0_cond_p050))
                y0_cond_p050_temp = rearrange(y0_cond_p050_temp, "f c h w -> 1 c f h w")
                y0_restored_p050_temp = (torch.from_numpy(y0_restored_p050))
                y0_restored_p050_refined_temp = (torch.from_numpy(y0_restored_p050_refined))

                y0_cond_p033_temp = (torch.from_numpy(y0_cond_p033))
                y0_cond_p033_temp = rearrange(y0_cond_p033_temp, "f c h w -> 1 c f h w")
                y0_restored_p033_temp = (torch.from_numpy(y0_restored_p033))
                y0_restored_p033_refined_temp = (torch.from_numpy(y0_restored_p033_refined))

                y0_cond_p025_temp = (torch.from_numpy(y0_cond_p025))
                y0_cond_p025_temp = rearrange(y0_cond_p025_temp, "f c h w -> 1 c f h w")
                y0_restored_p025_temp = (torch.from_numpy(y0_restored_p025))
                y0_restored_p025_refined_temp = (torch.from_numpy(y0_restored_p025_refined))

                print(y0_orig_temp.shape)
                print(y0_cond_p050_temp.shape)
                print(y0_restored_p050_temp.shape)

                tensor_video = torch.cat((y0_orig_temp, y0_cond_p050_temp, y0_restored_p050_temp, y0_restored_p050_refined_temp, y0_cond_p033_temp, y0_restored_p033_temp, y0_restored_p033_refined_temp, y0_cond_p025_temp, y0_restored_p025_temp, y0_restored_p025_refined_temp), 0)
                tensor_video = rearrange(tensor_video, 'b c f h w  -> c f h (b w)')
                gif_path = str(self.results_folder / f'iter{self.step}.gif')
                video_tensor_to_gif(tensor_video, gif_path)

                self.save(self.step)

            self.step += 1

        print('Training completed.')
        
        

""" 
Sampler for continuous-time
"""
class ContinuousTimeSampler(nn.Module):
    def __init__(
        self, 
        diffusion_model, 
        modelzoo_file_path, 
        rank, 
        *, 
        sampler_type = 'ddpm',    # \in {'ddpm', 'ddim', 'dpm-solver', 'dpm-solver-v2'}
        num_sample_steps = 64,    # \in {1, 2, 4, 8, 16, 32, 64, 128, 256} 
    ):
        super().__init__()
        self.rank = rank

        self.diffusion_model = diffusion_model.to(self.rank)
        self.ema_diffusion_model = copy.deepcopy(self.diffusion_model)
        self.modelzoo_file_path = Path(modelzoo_file_path)
        
        self.sampler_type = sampler_type
        self.num_sample_steps = num_sample_steps

        self.image_size = self.diffusion_model.image_size 
        self.channels = self.diffusion_model.channels
        self.frames = self.diffusion_model.frames


    def load(self, **kwargs):
        data = torch.load(f'{self.modelzoo_file_path}')
        
        # self.step = data['step']
        self.diffusion_model.load_state_dict(data['model'], **kwargs)
        self.ema_diffusion_model.load_state_dict(data['ema'], **kwargs)
        # self.scaler.load_state_dict(data['scaler'])


    def sample(self, input_depth = None, output_pkl = None):
        # preparation
        input_depth = input_depth.to(torch.float32).to(self.rank)
        input_depth_temp = copy.deepcopy(input_depth)

        input_depth = einops.rearrange(input_depth, 'b f c h w -> b c f h w')
        input_depth_temp = einops.rearrange(input_depth_temp, 'b f c h w -> b c f h w')

        # sampling
        output_tensor, output_list = self.ema_diffusion_model.sample(x_cond = input_depth, num_sample_steps = self.num_sample_steps, sampler = self.sampler_type)

        # clip
        thres_max = 0.9
        thres_min = 0.1 
        output_tensor[output_tensor <= thres_min] = 0.
        output_tensor[output_tensor >= thres_max] = 0.

        # visualization
        video_tensor = torch.cat((input_depth_temp, output_tensor), dim = 0)
        video_tensor = einops.rearrange(video_tensor, 'b c f h w -> c f h (b w)')
        video_tensor_to_gif(video_tensor, './sample.gif')

        # tensor2numpy
        output_tensor = output_tensor.squeeze(dim=0)
        output_tensor = einops.rearrange(output_tensor, 'c f h w -> f c h w')
        output_np = output_tensor.cpu().numpy()

        # save np files
        pkl_file_path = output_pkl[0]
        pkl_dir_path = os.path.dirname(pkl_file_path)
        pkl_dir_path = Path(pkl_dir_path)
        pkl_dir_path.mkdir(exist_ok = True, parents = True)

        with open(pkl_file_path, 'wb') as f:
            pickle.dump(output_np, f)