"""
Utility functions: config I/O, logging, STN, and correlation maps.
"""
import datetime
import os
import json
from typing import Tuple

import torch
import torch.nn.functional as F


def reparameterize(mu, logvar):
    """z = mu + sigma * eps, eps ~ N(0, I)."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu)
    return mu + eps * std


def spatial_transform(image, z_pos, z_scale, out_dims, inverse=False, eps=1e-9):
    """
    Spatial transformer network with scaling and translation.
    image: [B * n_kp, ch, h, w]
    z_pos: [B * n_kp, 2] (translation)
    z_scale: [B * n_kp, 2] (scaling)
    out_dims: (B * n_kp, ch, h*, w*)
    """
    theta = torch.zeros(2, 3, device=image.device).repeat(image.shape[0], 1, 1)

    scale_x = z_scale[:, 1] if not inverse else 1 / (z_scale[:, 1] + eps)
    scale_y = z_scale[:, 0] if not inverse else 1 / (z_scale[:, 0] + eps)

    theta[:, 0, 0] = scale_x
    theta[:, 1, 1] = scale_y

    adjusted_tx = z_pos[:, 1]
    adjusted_ty = z_pos[:, 0]

    theta[:, 0, -1] = adjusted_tx if not inverse else -adjusted_tx / (z_scale[:, 1] + eps)
    theta[:, 1, -1] = adjusted_ty if not inverse else -adjusted_ty / (z_scale[:, 0] + eps)

    return affine_grid_sample(image, theta, out_dims, mode='bilinear')


@torch.no_grad()
def generate_correlation_maps(x, kp, patch_size, previous_objects=None, z_scale=None):
    """
    Correlation heatmaps between patches at `kp` and templates `previous_objects`.
    x: [B, ch, h, w] in [0, 1]
    kp: [B, n_kp, 2] in [-1, 1]
    previous_objects: [B * n_kp, 3, patch_size, patch_size] or None
    returns [B * n_kp, 4, patch_size, patch_size]
    """
    pad_size = patch_size
    batch_size = x.shape[0]
    img_size = x.shape[-1]
    n_kp = kp.shape[1]
    pad_func = torch.nn.ReplicationPad2d(pad_size)

    x_repeat = x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1).clamp(0, 1)
    x_repeat = x_repeat.view(-1, *x_repeat.shape[2:])
    if z_scale is None:
        z_scale = (patch_size / img_size) * torch.ones_like(kp)
    else:
        z_scale = torch.sigmoid(z_scale)
    z_pos = kp.reshape(-1, kp.shape[-1])
    z_scale = z_scale.view(-1, z_scale.shape[-1])
    out_dims = (batch_size * n_kp, x.shape[1], patch_size, patch_size)
    cropped_objects = spatial_transform(x_repeat, z_pos, z_scale, out_dims, inverse=False)

    if previous_objects is None:
        cropped_heatmaps = torch.zeros(cropped_objects.shape[0], 1, patch_size, patch_size,
                                       device=cropped_objects.device)
    else:
        in0 = cropped_objects.reshape(1, -1, *cropped_objects.shape[2:])
        in0 = pad_func(in0)
        in1 = previous_objects.reshape(-1, *previous_objects.shape[1:]).clamp(0, 1)
        output = correlate(in0, in1)
        output = output.view(batch_size * n_kp, -1, *output.shape[2:])
        output = output[:, :, pad_size // 2 + 1:-pad_size // 2, pad_size // 2 + 1:-pad_size // 2]
        cropped_heatmaps = output
        output_vals = output.reshape(output.shape[0], output.shape[1], -1)
        min_val = output_vals.min(-1)[0]
        max_val = output_vals.max(-1)[0]
        cropped_heatmaps = (cropped_heatmaps - min_val[:, :, None, None]) / (
                max_val[:, :, None, None] - min_val[:, :, None, None] + 1e-5)
    return torch.cat([cropped_objects, cropped_heatmaps], dim=1)


def calc_model_size(model):
    num_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return {'n_params': num_trainable_params, 'size_mb': size_all_mb}


def normalize_01(im):
    return (im - im.min()) / (im.max() - im.min() + 1e-12)


def to_uint16(im):
    return (normalize_01(im) * 65535).round().astype('uint16')


def prepare_logdir(runname, src_dir='./'):
    td_prefix = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
    path_to_dir = os.path.join(src_dir, f'{td_prefix}_{runname}')
    os.makedirs(os.path.join(path_to_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(path_to_dir, 'saves'), exist_ok=True)
    return path_to_dir


def save_config(src_dir, hparams):
    with open(os.path.join(src_dir, 'hparams.json'), "w") as outfile:
        json.dump(hparams, outfile, indent=2)


def flatten_config(config):
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict) and key != 'optics_dict':
            for sub_key, sub_value in value.items():
                assert sub_key not in flat, f'duplicate config key: {sub_key}'
                flat[sub_key] = sub_value
        else:
            assert key not in flat, f'duplicate config key: {key}'
            flat[key] = value
    return flat


def get_config(fpath):
    with open(fpath, 'r') as f:
        return flatten_config(json.load(f))


def log_line(src_dir, line):
    with open(os.path.join(src_dir, 'log.txt'), 'a') as fp:
        fp.writelines(line)


"""
JIT scripts
"""


@torch.jit.script
def correlate(x, kernel):
    groups = kernel.shape[0]
    output = F.conv2d(x, kernel, padding=0, groups=groups, stride=1, bias=None)
    norm = torch.sqrt(torch.sum(kernel ** 2) * F.conv2d(x ** 2, torch.ones_like(kernel), groups=groups,
                                                       bias=None, stride=1, padding=0) + 1e-10)
    return output / (norm + 1e-5)


@torch.jit.script
def affine_grid_sample(x, theta, out_dims: Tuple[int, int, int, int], mode: str):
    grid = F.affine_grid(theta, torch.Size(out_dims), align_corners=True)
    return F.grid_sample(x, grid, align_corners=True, mode=mode)
