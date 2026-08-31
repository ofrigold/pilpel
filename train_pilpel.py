"""
Train PILPEL.
Point CONFIG at a dataset config, set DEVICE for this machine, optionally tweak
the override block below, then run this file.
"""

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from utils.util_func import get_config
from trainer import train


DEVICE = 'cuda:0'  # GPU index for this machine; falls back to CPU if unavailable

CONFIG = 'configs/microtubules_3d.json'

def build_config() -> dict:
    config = get_config(CONFIG)
    config['device'] = DEVICE

    # per-run overrides
    config.update({
        # e.g.
        # 'num_epochs': 5,
        # 'beta_kl': 0.05
    })
    return config


if __name__ == '__main__':
    print('--- PILPEL ---')
    config = build_config()
    for key, value in config.items():
        print(f'{key}: {value}')

    train(config)
