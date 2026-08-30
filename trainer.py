"""
Single-GPU training of PILPEL.
"""
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.optim as optim

from model import PILPEL
from dataset import ExpDataset
from utils.loss_functions import calc_reconstruction_loss, VGGDistance
from utils.util_func import prepare_logdir, save_config, log_line, get_config
from eval.eval_model import evaluate_validation_elbo
from eval.eval_gen_metrics import eval_dlp_im_metric

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train(config):
    if isinstance(config, str):
        config = get_config(config)
    hparams = config

    # data and general
    ds = config['ds']
    ch = config['ch']
    image_size = config['image_size']
    root = config['root']
    batch_size = config['batch_size']
    lr = config['lr']
    num_epochs = config['num_epochs']
    eval_epoch_freq = config['eval_epoch_freq']
    weight_decay = config['weight_decay']
    run_prefix = config['run_prefix']
    load_model = config['load_model']
    pretrained_path = config['pretrained_path']
    adam_betas = config['adam_betas']
    adam_eps = config['adam_eps']
    scheduler_gamma = config['scheduler_gamma']
    eval_im_metrics = config['eval_im_metrics']
    device = config['device']
    if 'cuda' in device:
        device = torch.device(f'{device}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # model
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
    norm01_flag = config['norm01_flag']
    zero_close_psfs_flag = config['zero_close_psfs_flag']
    read_noise = config.get('read_noise', False)

    # optimization
    warmup_epoch = config['warmup_epoch']
    recon_loss_type = config['recon_loss_type']
    beta_kl = config['beta_kl']
    beta_rec = config['beta_rec']
    kl_balance = config['kl_balance']
    train_enc_prior = config['train_enc_prior']


    optics_dict = config['optics_dict']

    # priors
    sigma = config['sigma']
    scale_std = config['scale_std']
    offset_std = config['offset_std']
    obj_on_alpha = config['obj_on_alpha']
    obj_on_beta = config['obj_on_beta']

    dataset = ExpDataset(root=root, mode='train', image_size=image_size)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                            drop_last=True)

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

    run_name = ds + run_prefix
    log_dir = prepare_logdir(runname=run_name, src_dir='./runs')
    fig_dir = os.path.join(log_dir, 'figures')
    save_dir = os.path.join(log_dir, 'saves')
    save_config(log_dir, hparams)

    if recon_loss_type == "vgg":
        recon_loss_func = VGGDistance(device=device)
    else:
        recon_loss_func = calc_reconstruction_loss

    optimizer = optim.Adam(model.get_parameters(), lr=lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=scheduler_gamma, verbose=True)

    if load_model and pretrained_path is not None:
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
            print("loaded model from checkpoint")
        except Exception:
            print("model checkpoint not found")

    losses = []
    losses_rec = []
    losses_kl = []
    losses_kl_kp = []
    losses_kl_feat = []
    losses_kl_scale = []
    losses_kl_depth = []
    losses_kl_defocus = []
    losses_kl_obj_on = []

    valid_loss = best_valid_loss = 1e8
    valid_losses = []
    best_valid_epoch = 0
    psnrs = []

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        batch_losses_rec = []
        batch_losses_kl = []
        batch_losses_kl_kp = []
        batch_losses_kl_feat = []
        batch_losses_kl_scale = []
        batch_losses_kl_depth = []
        batch_losses_kl_defocus = []
        batch_losses_kl_obj_on = []
        batch_psnrs = []

        pbar = tqdm(iterable=dataloader)
        for batch in pbar:
            x = batch[0].to(device)
            x_prior = x
            noisy = False
            model_output = model(x, x_prior=x_prior, warmup=(epoch < warmup_epoch), noisy=noisy,
                                 train_enc_prior=train_enc_prior, norm01=norm01_flag,
                                 deterministic=True)
            all_losses = model.calc_elbo(x, model_output, warmup=(epoch < warmup_epoch), beta_kl=beta_kl,
                                         beta_rec=beta_rec, kl_balance=kl_balance,
                                         recon_loss_type=recon_loss_type,
                                         recon_loss_func=recon_loss_func, noisy=noisy)
            loss = all_losses['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            z_base = model_output['z_base']
            mu_offset = model_output['mu_offset']
            rec_x = model_output['rec']
            mu_scale = model_output['mu_scale']
            mu_depth = model_output['mu_depth']
            z_read_noise = model_output.get('z_read_noise', torch.zeros_like(mu_depth))

            defocus = model_output['z_defocus']
            obj_on = model_output['obj_on']
            psf_on = model_output['psf_on']
            PSFs = model_output['PSFs']

            psnr = all_losses['psnr']
            loss_kl = all_losses['kl']
            loss_rec = all_losses['loss_rec']
            loss_kl_kp = all_losses['loss_kl_kp']
            loss_kl_feat = all_losses['loss_kl_feat']
            loss_kl_scale = all_losses['loss_kl_scale']
            loss_kl_depth = all_losses['loss_kl_depth']
            loss_kl_defocus = all_losses['loss_kl_defocus']
            loss_kl_obj_on = all_losses['loss_kl_obj_on']

            mu_tot = z_base + mu_offset

            batch_psnrs.append(psnr.data.cpu().item())
            batch_losses.append(loss.data.cpu().item())
            batch_losses_rec.append(loss_rec.data.cpu().item())
            batch_losses_kl.append(loss_kl.data.cpu().item())
            batch_losses_kl_kp.append(loss_kl_kp.data.cpu().item())
            batch_losses_kl_feat.append(loss_kl_feat.data.cpu().item())
            batch_losses_kl_scale.append(loss_kl_scale.data.cpu().item())
            batch_losses_kl_depth.append(loss_kl_depth.data.cpu().item())
            batch_losses_kl_defocus.append(loss_kl_defocus.data.cpu().item())
            batch_losses_kl_obj_on.append(loss_kl_obj_on.data.cpu().item())

            if epoch < warmup_epoch:
                pbar.set_description_str(f'epoch #{epoch} (warmup)')
            else:
                pbar.set_description_str(f'epoch #{epoch}')
            pbar.set_postfix(loss=loss.data.cpu().item(), rec=loss_rec.data.cpu().item(),
                             kl=loss_kl.data.cpu().item())
        pbar.close()

        losses.append(np.mean(batch_losses))
        losses_rec.append(np.mean(batch_losses_rec))
        losses_kl.append(np.mean(batch_losses_kl))
        losses_kl_kp.append(np.mean(batch_losses_kl_kp))
        losses_kl_feat.append(np.mean(batch_losses_kl_feat))
        losses_kl_scale.append(np.mean(batch_losses_kl_scale))
        losses_kl_depth.append(np.mean(batch_losses_kl_depth))
        losses_kl_defocus.append(np.mean(batch_losses_kl_defocus))
        losses_kl_obj_on.append(np.mean(batch_losses_kl_obj_on))
        if len(batch_psnrs) > 0:
            psnrs.append(np.mean(batch_psnrs))

        scheduler.step()

        log_str = f'epoch {epoch} summary\n'
        log_str += (f'loss: {losses[-1]:.3f}, rec: {losses_rec[-1]:.3f}, kl: {losses_kl[-1]:.3f}, '
                    f'psnr: {psnrs[-1]:.3f}\n')
        log_str += (f'kl_balance: {kl_balance:.3f}, '
                    f'kl_kp: {losses_kl_kp[-1]:.3f}, kl_feat: {losses_kl_feat[-1]:.3f}\n')
        log_str += (f'kl_scale: {losses_kl_scale[-1]:.3f}, kl_depth: {losses_kl_depth[-1]:.3f}, '
                    f'kl_defocus: {losses_kl_defocus[-1]:.3f}, kl_obj_on: {losses_kl_obj_on[-1]:.3f}\n')
        log_str += f'mu max: {mu_tot.max()}, mu min: {mu_tot.min()}\n'
        log_str += f'mu offset max: {mu_offset.max()}, mu offset min: {mu_offset.min()}\n'
        log_str += (f'val loss (freq: {eval_epoch_freq}): {valid_loss:.3f}, '
                    f'best: {best_valid_loss:.3f} @ epoch: {best_valid_epoch}\n')
        if obj_on is not None:
            log_str += f'obj_on max: {obj_on.max()}, obj_on min: {obj_on.min()}\n'
            log_str += f'scale max: {mu_scale.sigmoid().max()}, scale min: {mu_scale.sigmoid().min()}\n'
            log_str += f'depth max: {mu_depth.max()}, depth min: {mu_depth.min()}\n'
            log_str += f'defocus max: {defocus.max()}, defocus min: {defocus.min()}\n'
            log_str += f'psf_on max: {psf_on.max()}, psf_on min: {psf_on.min()}\n'
            log_str += f'z_read_noise max: {z_read_noise.max()}, z_read_noise min: {z_read_noise.min()}\n'
        print(log_str)
        log_line(log_dir, log_str)

        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            max_imgs = 8
            with torch.no_grad():
                bb_scores = -1 * psf_on

            bb_str = (f'bb scores: max: {bb_scores.max():.2f}, min: {bb_scores.min():.2f}, '
                      f'mean: {bb_scores.mean():.2f}\n')
            print(bb_str)
            log_line(log_dir, bb_str)

            dec_objects = model_output['dec_objects']
            bg = model_output['bg']
            noisy_bg = model_output.get('noisy_bg', bg)

            vutils.save_image(torch.cat([x[:max_imgs, -3:],
                                         rec_x[:max_imgs, -3:], PSFs[:max_imgs, -3:],
                                         dec_objects[:max_imgs, -3:],
                                         bg[:max_imgs, -3:], noisy_bg[:max_imgs, -3:]],
                                        dim=0).data.cpu(), '{}/image_{}.jpg'.format(fig_dir, epoch),
                              nrow=4, pad_value=1)


            torch.save(model.state_dict(), os.path.join(save_dir, f'{run_name}.pth'))
            print("validation step...")
            valid_loss = evaluate_validation_elbo(model, config, epoch, batch_size=batch_size,
                                                  recon_loss_type=recon_loss_type, device=device,
                                                  save_image=True, fig_dir=fig_dir,
                                                  recon_loss_func=recon_loss_func, beta_rec=beta_rec,
                                                  beta_kl=beta_kl, kl_balance=kl_balance)
            log_str = f'validation loss: {valid_loss:.3f}\n'
            print(log_str)
            log_line(log_dir, log_str)
            if best_valid_loss > valid_loss:
                log_str = f'validation loss updated: {best_valid_loss:.3f} -> {valid_loss:.3f}\n'
                print(log_str)
                log_line(log_dir, log_str)
                best_valid_loss = valid_loss
                best_valid_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(save_dir, f'{run_name}_best.pth'))
            torch.cuda.empty_cache()
            if eval_im_metrics and epoch > 0:
                valid_imm_results = eval_dlp_im_metric(model, device, config,
                                                       metrics=('ssim', 'psnr'),
                                                       val_mode='val',
                                                       eval_dir=log_dir,
                                                       batch_size=batch_size)
                im_str = (f'val psnr: {valid_imm_results["psnr"]:.3f}, '
                          f'val ssim: {valid_imm_results["ssim"]:.3f}\n')
                print(im_str)
                log_line(log_dir, im_str)
                torch.cuda.empty_cache()
        valid_losses.append(valid_loss)

        if epoch > 0:
            num_plots = 4
            fig = plt.figure()
            ax = fig.add_subplot(num_plots, 1, 1)
            ax.plot(np.arange(len(losses[1:])), losses[1:], label="loss")
            ax.set_title(run_name)
            ax.legend()

            ax = fig.add_subplot(num_plots, 1, 2)
            ax.plot(np.arange(len(losses_kl[1:])), losses_kl[1:], label="kl", color='red')
            if learned_feature_dim > 0:
                ax.plot(np.arange(len(losses_kl_kp[1:])), losses_kl_kp[1:], label="kl_kp", color='cyan')
                ax.plot(np.arange(len(losses_kl_feat[1:])), losses_kl_feat[1:], label="kl_feat", color='green')
            ax.legend()

            ax = fig.add_subplot(num_plots, 1, 3)
            ax.plot(np.arange(len(losses_rec[1:])), losses_rec[1:], label="rec", color='green')
            ax.legend()

            ax = fig.add_subplot(num_plots, 1, 4)
            ax.plot(np.arange(len(valid_losses[1:])), valid_losses[1:], label="valid_loss", color='magenta')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'{fig_dir}/{run_name}_graph.jpg')
            plt.close('all')
    return model
