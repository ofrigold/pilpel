"""
Evaluation of the ELBO on the validation set.
"""
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from dataset import ExpDataset
from utils.loss_functions import calc_reconstruction_loss, VGGDistance


def evaluate_validation_elbo(model, config, epoch, batch_size=100, recon_loss_type="vgg",
                             device=torch.device('cpu'), save_image=False, fig_dir='./',
                             recon_loss_func=None, beta_rec=1.0, beta_kl=1.0,
                             kl_balance=0.001):
    model.eval()
    image_size = config['image_size']
    root = config['root']
    dataset = ExpDataset(root=root, mode='valid', image_size=image_size)
    # shuffle=False: the mean over the full val set is order-independent anyway, and a
    # fixed order makes image_valid_<epoch>.jpg show the same crops across runs.
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=4, drop_last=False)
    if recon_loss_func is None:
        if recon_loss_type == "vgg":
            recon_loss_func = VGGDistance(device=device)
        else:
            recon_loss_func = calc_reconstruction_loss

    elbos = []
    fig_grid = None
    max_imgs = 4
    for batch in dataloader:
        x = batch[0].to(device)
        x_prior = x
        with torch.no_grad():
            model_output = model(x, x_prior=x_prior, deterministic=True, norm01=config['norm01_flag'])
            all_losses = model.calc_elbo(x, model_output, beta_kl=beta_kl,
                                         beta_rec=beta_rec, kl_balance=kl_balance,
                                         recon_loss_type=recon_loss_type,
                                         recon_loss_func=recon_loss_func)
        loss = all_losses['loss']

        if save_image and fig_grid is None:
            bg = model_output['bg']
            fig_grid = torch.cat([x[:max_imgs, -3:],
                                  model_output['rec'][:max_imgs, -3:],
                                  model_output['PSFs'][:max_imgs, -3:],
                                  model_output['dec_objects'][:max_imgs, -3:],
                                  bg[:max_imgs, -3:],
                                  model_output.get('noisy_bg', bg)[:max_imgs, -3:]],
                                 dim=0).data.cpu()

        elbos.append(loss.data.cpu().numpy())

    if fig_grid is not None:
        vutils.save_image(fig_grid, '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                          nrow=4, pad_value=1)

    return np.mean(elbos)
