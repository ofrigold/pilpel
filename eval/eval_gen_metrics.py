"""
Evaluate image metrics (SSIM, PSNR, LPIPS) on the validation set.
"""
import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ExpDataset

try:
    from piqa import PSNR, LPIPS, SSIM
except ImportError:
    print("piqa library required to compute image metrics")
    raise SystemExit


class ImageMetrics(nn.Module):
    def __init__(self, metrics=('ssim', 'psnr', 'lpips'), num_channels=3):
        super().__init__()
        self.metrics = metrics
        self.ssim = SSIM(reduction='none', n_channels=num_channels) if 'ssim' in self.metrics else None
        self.psnr = PSNR(reduction='none') if 'psnr' in self.metrics else None
        self.lpips = LPIPS(network='vgg', reduction='none') if 'lpips' in self.metrics else None

    @torch.no_grad()
    def forward(self, x, y):
        results = {}
        if self.ssim is not None:
            results['ssim'] = self.ssim(x, y)
        if self.psnr is not None:
            results['psnr'] = self.psnr(x, y)
        if self.lpips is not None:
            results['lpips'] = self.lpips(x, y)
        return results


def eval_dlp_im_metric(model, device, config, val_mode='val', eval_dir='./',
                       metrics=('ssim', 'psnr', 'lpips'), batch_size=32):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    ch = config['ch']
    image_size = config['image_size']
    root = config['root']
    dataset = ExpDataset(root=root, mode=val_mode, image_size=image_size)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=False)

    evaluator = ImageMetrics(metrics=metrics, num_channels=ch).to(device)

    results = {}
    ssims = []
    psnrs = []
    lpipss = []
    for batch in tqdm(dataloader):
        x = batch[0].to(device)
        x_prior = x
        with torch.no_grad():
            output = model(x, x_prior=x_prior, deterministic=True)
            generated = output['rec'].clamp(0, 1)
            if len(x.shape) == 5:
                x = x.view(-1, *x.shape[2:])
            results = evaluator(x, generated)
        if 'ssim' in metrics:
            ssims.append(results['ssim'])
        if 'psnr' in metrics:
            psnrs.append(results['psnr'])
        if 'lpips' in metrics:
            lpipss.append(results['lpips'])

    if 'ssim' in metrics:
        ssims = torch.cat(ssims, dim=0)
        results['ssim'] = ssims.mean().data.cpu().item()
        results['ssim_std'] = ssims.std().data.cpu().item()
    if 'psnr' in metrics:
        psnrs = torch.cat(psnrs, dim=0)
        results['psnr'] = psnrs.mean().data.cpu().item()
        results['psnr_std'] = psnrs.std().data.cpu().item()
    if 'lpips' in metrics:
        lpipss = torch.cat(lpipss, dim=0)
        results['lpips'] = lpipss.mean().data.cpu().item()
        results['lpips_std'] = lpipss.std().data.cpu().item()

    path_to_conf = os.path.join(eval_dir, 'last_val_image_metrics.json')
    with open(path_to_conf, "w") as outfile:
        json.dump(results, outfile, indent=2)

    del evaluator
    return results
