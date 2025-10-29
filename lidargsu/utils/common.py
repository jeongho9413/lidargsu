import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import transforms as T, utils

from PIL import Image


# tensor of shape (channels, frames, height, width) -> gif
def video_tensor_to_gif(tensor, path, duration = 100, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images


# gif -> (channels, frame, height, width) tensor
def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)


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