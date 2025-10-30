import numpy as np

import pickle
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import transforms as T, utils

from PIL import Image
from einops import rearrange


"""
Tensor of shape (channels, frames, height, width) -> gif
"""
def video_tensor_to_gif(tensor, path, duration=100, loop=0, optimize=True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs, duration=duration, loop=loop, optimize=optimize)
    return images


def pkl_to_gif(pkl_path, gif_path):
    with open(pkl_path, 'rb') as f:
        np_video = pickle.load(f)

    """ (f c h w) """
    F, C, _, _ = np_video.shape
    assert C in [1, 3], 'check the np_video.shape!'

    print(f'np_video.shape: {np_video.shape}')
    print(f'np_video.max(): {np_video.max()}')
    print(f'np_video.min(): {np_video.min()}')
    print(f'np_video.dtype: {np_video.dtype}')
    
    tensor_video = (torch.from_numpy(np_video)).unsqueeze(0)
    tensor_video = rearrange(tensor_video, 'b f c h w  -> c f h (b w)')
    video_tensor_to_gif(tensor_video, gif_path)


"""
gif -> (channels, frame, height, width) tensor
"""
def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)


def rotation_matrix(rad_value):
    assert (-np.pi * 2) <= rad_value <= (np.pi * 2), f'Make sure your rad_value: {rad_value}, or degrees?'

    c, s = np.cos(rad_value), np.sin(rad_value)
    rot_mat = np.array(((c, -s), (s, c)))
    # rot_mat = np.array([np.cos(rad_value), -np.sin(rad_value)], [np.sin(rad_value), np.cos(rad_value)])
    return rot_mat


def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)


def divisible_by(numer, denom):
    return (numer % denom) == 0


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


def identity(t, *args, **kwargs):
    return t


def normalize(x):
    return x * 2 - 1    # \in [-1, 1]


def unnormalize(x):
    return (x + 1) * 0.5    # \in [0, 1]


def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))