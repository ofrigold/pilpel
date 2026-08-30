"""
Loads a trained PILPEL model from a checkpoint.
"""
import torch

from model import PILPEL
from utils.util_func import get_config


def load(pretrained_path, device, config_path):
    try:
        config = get_config(config_path)
    except FileNotFoundError:
        raise SystemExit("config file not found")

    ch = config['ch']
    image_size = config['image_size']
    kp_range = config['kp_range']
    kp_activation = config['kp_activation']
    enc_channels = config['enc_channels']
    prior_channels = config['prior_channels']
    pad_mode = config['pad_mode']
    n_kp = config['n_kp']
    n_kp_prior = config['n_kp_prior']
    n_kp_enc = config['n_kp_enc']
    patch_size = config['patch_size']
    anchor_s = config['anchor_s']
    learned_feature_dim = config['learned_feature_dim']
    bg_learned_feature_dim = config['bg_learned_feature_dim']
    dropout = config['dropout']
    use_resblock = config['use_resblock']
    use_correlation_heatmaps = config['use_correlation_heatmaps']
    filtering_heuristic = config['filtering_heuristic']

    read_noise = config.get('read_noise', False)
    zero_close_psfs_flag = config.get('zero_close_psfs_flag', False)

    sigma = config['sigma']
    scale_std = config['scale_std']
    offset_std = config['offset_std']
    obj_on_alpha = config['obj_on_alpha']
    obj_on_beta = config['obj_on_beta']

    optics_dict = config['optics_dict']

    model = PILPEL(optics_dict, cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                   image_size=image_size, n_kp=n_kp, learned_feature_dim=learned_feature_dim,
                   bg_learned_feature_dim=bg_learned_feature_dim,
                   pad_mode=pad_mode, sigma=sigma, read_noise=read_noise,
                   dropout=dropout, patch_size=patch_size, n_kp_enc=n_kp_enc,
                   n_kp_prior=n_kp_prior, kp_range=kp_range, kp_activation=kp_activation,
                   anchor_s=anchor_s, use_resblock=use_resblock,
                   scale_std=scale_std, offset_std=offset_std, obj_on_alpha=obj_on_alpha,
                   obj_on_beta=obj_on_beta,
                   use_correlation_heatmaps=use_correlation_heatmaps,
                   filtering_heuristic=filtering_heuristic,
                   zero_close_psfs_flag=zero_close_psfs_flag).to(device)
    print(model.info())

    try:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("loaded model from checkpoint")
    except RuntimeError:
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
            print("loaded model from checkpoint with partial matching")
        except Exception:
            raise RuntimeError("model checkpoint not found or could not be loaded")

    return model
