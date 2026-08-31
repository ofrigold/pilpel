# imports
import numpy as np
# torch
import torch
import torch.nn.functional as F
import torch.nn as nn
# modules
from modules import KeyPointCNNOriginal, CNNDecoder, ObjectDecoderCNN, FCToCNN, VariationalKeyPointPatchEncoderPSF
from modules import ParticleAttributeEncoder, ParticleFeaturesEncoder
# util functions
from utils.util_func import reparameterize, spatial_transform, calc_model_size
from utils.loss_functions import ChamferLossKL, calc_kl, calc_reconstruction_loss, VGGDistance, calc_kl_beta_dist
from physics import PhysicalLayer_mask_rec, PhysicalLayer_mask_rec2D, Normalize01, Croplayer
import torchvision

class FgDLP(nn.Module):
    def __init__(self, optics_dict, cdim=3, enc_channels=(16, 16, 32), prior_channels=(16, 16, 32), image_size=64, n_kp=1,
                 pad_mode='replicate', sigma=1.0, dropout=0.0,
                 patch_size=16, n_kp_enc=20, n_kp_prior=20, learned_feature_dim=16,
                 kp_range=(-1, 1), kp_activation="tanh", anchor_s=0.25,
                 use_resblock=False, use_correlation_heatmaps=True,
                 filtering_heuristic='variance', zero_close_psfs_flag=True):
        """
        DLP Foreground Module -- extract objects from an image
        cdim: channels of the input image (3...)
        enc_channels: channels for the posterior CNN (takes in the whole image)
        prior_channels: channels for prior CNN (takes in patches)
        n_kp: number of kp to extract from each (!) patch
        n_kp_prior: number of kp to filter from the set of prior kp (of size n_kp x num_patches)
        n_kp_enc: number of posterior kp to be learned (this is the actual number of kp that will be learnt)
        pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
        sigma: the prior std of the KP
        dropout: dropout for the CNNs. We don't use it though...
        patch_size: patch size for the prior KP proposals network (not to be confused with the glimpse size)
        kp_range: the range of keypoints, can be [-1, 1] (default) or [0,1]
        learned_feature_dim: the latent visual features dimensions extracted from glimpses.
        kp_activation: the type of activation to apply on the keypoints: "tanh" for kp_range [-1, 1], "sigmoid" for [0, 1]
        anchor_s: defines the glimpse size as a ratio of image_size (e.g., 0.25 for image_size=128 -> glimpse_size=32)
        use_correlation_heatmaps: use correlation heatmaps as input to model particle properties (e.g., xy offset)
        filtering heuristic: filtering heuristic to filter prior keypoints, ['distance', 'max_corr']
        """
        super(FgDLP, self).__init__()
        self.image_size = image_size
        self.sigma = sigma
        self.dropout = dropout
        self.kp_range = kp_range
        self.psf_model = optics_dict.get('psf_model', '3d')
        assert self.psf_model in ['2d', '3d'], f'unknown psf model: {self.psf_model}'
        assert self.psf_model == '2d' or 'phase_mask_root' in optics_dict, \
            "a '3d' psf model needs optics_dict['phase_mask_root']; set 'psf_model': '2d' for a mask-free run"
        self.z_range = optics_dict.get('z_range', None)
        self.n_kp = n_kp
        self.n_kp_total = n_kp_prior
        self.n_kp_prior = n_kp_prior
        self.n_kp_enc = n_kp_enc
        self.kp_activation = kp_activation
        self.patch_size = patch_size
        self.anchor_patch_s = patch_size / image_size
        self.features_dim = int(image_size // (2 ** (len(enc_channels) - 1)))
        self.learned_feature_dim = learned_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.anchor_s = anchor_s
        if anchor_s<1:
            self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
        else:
            self.obj_patch_size  = image_size
        self.exclusive_patches = False
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.use_correlation_heatmaps = use_correlation_heatmaps
        self.zero_close_psfs_flag = zero_close_psfs_flag
        assert filtering_heuristic in ['distance',
                                       'max_corr'], f'unknown filtering heuristic: {filtering_heuristic}'
        self.filtering_heuristic = filtering_heuristic
        self.optics_dict = optics_dict

        # prior
        self.prior = VariationalKeyPointPatchEncoderPSF(self.optics_dict, cdim=cdim, channels=prior_channels, image_size=image_size,
                                                     n_kp=n_kp, kp_range=self.kp_range,
                                                     patch_size=patch_size,
                                                     pad_mode=pad_mode, sigma=sigma, dropout=dropout,
                                                     learnable_logvar=False, learned_feature_dim=0,
                                                     use_resblock=self.use_resblock)
        # attribute encoder - anchor (z_a), offset (z_o), scale (z_s), transparency (z_t) and depth (z_d)
        self.particle_attribute_enc = ParticleAttributeEncoder(anchor_size=anchor_s, image_size=image_size,
                                                               margin=0, ch=cdim,
                                                               kp_activation=kp_activation,
                                                               use_resblock=self.use_resblock,
                                                               max_offset=1.0, cnn_channels=prior_channels,
                                                               use_correlation_heatmaps=use_correlation_heatmaps)
        # appearance encoder - visual features encoder (z_f)
        self.particle_features_enc = ParticleFeaturesEncoder(anchor_s, learned_feature_dim,
                                                             image_size,
                                                             cnn_channels=prior_channels,
                                                             margin=0, ch=cdim)

        # object decoder
        self.object_dec = ObjectDecoderCNN(patch_size=(self.obj_patch_size, self.obj_patch_size), num_chans=2,
                                           bottleneck_size=learned_feature_dim, use_resblock=self.use_resblock)
        if self.psf_model == '2d':
            self.physical_layer = PhysicalLayer_mask_rec2D(self.optics_dict, self.image_size)
        else:
            self.physical_layer = PhysicalLayer_mask_rec(self.optics_dict, self.image_size)
        self.init_weights()

    def get_parameters(self, prior=True, encoder=True, decoder=True):
        parameters = []
        if prior:
            parameters.extend(list(self.prior.parameters()))
        if encoder:
            parameters.extend(list(self.particle_attribute_enc.parameters()))
            parameters.extend(list(self.particle_features_enc.parameters()))
        if decoder:
            parameters.extend(list(self.object_dec.parameters()))
        return parameters


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # use pytorch's default
                pass

    def encode_all(self, x, deterministic=False, noisy=False, warmup=False, kp_init=None, cropped_objects_prev=None,
                   scale_prev=None, refinement_iter=False, with_offset=True, defocus_init=None, max_corr=None):
        """
        2-stage encoding:
        0. if kp_init is None: create evenly spaced anchors. kp_init is z_base.
        1. attributes encoding: obj_on (z_t), depth (z_d), offset (z_o) and scale (z_s) encoding:
            produces [obj_on_a, obj_on_b] / [mu, logvar].
        2. features (z_f) encoding: [mu, logvar]
        """
        # kp_init: [batch_size, n_kp, 2] in [-1, 1]
        batch_size, ch, h, w = x.shape
        # 0. create or filter anchors
        if kp_init is None:
            # randomly sample n_kp_enc kp
            mu = torch.rand(batch_size, self.n_kp_enc, 2, device=x.device) * 2 - 1  # in [-1, 1]
        elif kp_init.shape[1] > self.n_kp_enc:
            mu = kp_init[:, :self.n_kp_enc]
        else:
            mu = kp_init
        logvar = torch.zeros_like(mu)
        z_base = mu + 0.0 * logvar  # deterministic value for chamfer-kl
        defocus_base = defocus_init.view(mu.shape[0], mu.shape[1], 1)

        # 1. posterior offsets and scale, it is okay of scale_prev is None
        scale_in = None if (noisy or warmup) else scale_prev
        cropped_objects_prev = None if (noisy or warmup) else cropped_objects_prev
        particle_stats_dict = self.particle_attribute_enc(x, z_base.detach(),
                                                          previous_objects=cropped_objects_prev,
                                                          z_scale=scale_in)
        # first iteration encodes the refined anchor (z_a), then a second one to lock on target better (z_o)
        if refinement_iter:
            mu_offset = particle_stats_dict['mu']
            mu = z_base + mu_offset
            z_base = mu + 0.0 * logvar
            mu_defocus_offset = particle_stats_dict['mu_defocus_offset']
            defocus_base = defocus_base + mu_defocus_offset
            if scale_prev is None:
                scale_prev = None if (noisy or warmup) else particle_stats_dict['mu_scale'].detach()
            cropped_objects_prev = None if (noisy or warmup) else cropped_objects_prev
            particle_stats_dict = self.particle_attribute_enc(x, z_base.detach(),
                                                              previous_objects=cropped_objects_prev,
                                                              z_scale=scale_prev)

        if with_offset:
            # Scale down network offset predictions by 10 for finer sub-pixel localization
            mu_offset = particle_stats_dict['mu'] / 10
            logvar_offset = particle_stats_dict['logvar']
        else:
            mu_offset = torch.zeros_like(particle_stats_dict['mu'])
            logvar_offset = torch.zeros_like(particle_stats_dict['logvar'])
        mu_scale = particle_stats_dict['mu_scale']
        logvar_scale = particle_stats_dict['logvar_scale']
        lobj_on_a = particle_stats_dict['lobj_on_a']
        lobj_on_b = particle_stats_dict['lobj_on_b']
        lpsf_on_a = particle_stats_dict['lpsf_on_a']
        lpsf_on_b = particle_stats_dict['lpsf_on_b']
        mu_depth = particle_stats_dict['mu_depth']
        logvar_depth = particle_stats_dict['logvar_depth']
        mu_defocus_offset = particle_stats_dict['mu_defocus_offset']
        logvar_defocus_offset = particle_stats_dict['logvar_defocus_offset']

        # final position
        mu_tot = z_base + mu_offset
        if self.psf_model == '2d':
            mu_defocus_tot = torch.zeros_like(defocus_base)
        else:
            mu_defocus_tot = defocus_base + mu_defocus_offset


        obj_on_a = lobj_on_a.exp().clamp_min(1e-5)
        obj_on_b = lobj_on_b.exp().clamp_min(1e-5)
        psf_on_a = lpsf_on_a.exp().clamp_min(1e-5)
        psf_on_b = lpsf_on_b.exp().clamp_min(1e-5)

        obj_on_beta_dist = torch.distributions.Beta(obj_on_a, obj_on_b)
        psf_on_beta_dist = torch.distributions.Beta(psf_on_a, psf_on_b)

        # reparameterize
        if deterministic:
            z_scale = mu_scale
            z_depth = mu_depth
            # always determinstic:
            z = mu_tot
            z_defocus = mu_defocus_tot
            z_obj_on = obj_on_beta_dist.mean
            z_psf_on = psf_on_beta_dist.mean * (max_corr - max_corr.min(dim=1, keepdim=True).values) /\
                       (max_corr.max(dim=1, keepdim=True).values - max_corr.min(dim=1, keepdim=True).values + 1e-6)
        else:
            z_scale = reparameterize(mu_scale, logvar_scale)
            z_depth = reparameterize(mu_depth, logvar_depth)
            # always determinstic:
            z = mu_tot
            z_defocus = mu_defocus_tot
            z_obj_on = obj_on_beta_dist.mean
            z_psf_on = psf_on_beta_dist.mean * (max_corr - max_corr.min(dim=1, keepdim=True).values) / \
                       (max_corr.max(dim=1, keepdim=True).values - max_corr.min(dim=1, keepdim=True).values + 1e-6)


        # during warm-up and noisy stages we use small values around the patch size for the scale
        if z_scale is not None and noisy:
            anchor_size = self.anchor_s
            z_scale = 0.0 * z_scale + (np.log(anchor_size / (1 - anchor_size + 1e-5)) + 0.1 * torch.randn_like(z_scale))
        # to avoid null cases where obj_on -> 0, we noise its values during the noisy stage

        if warmup:
            z_base = z_base.detach()
            z = z.detach()
            z_scale = z_scale.detach()
            z_defocus = z_defocus.detach()
            z_psf_on = z_psf_on.detach()

        # 2. posterior attributes: obj_on, depth and visual features
        obj_enc_out = self.particle_features_enc(x, z, z_scale=z_scale.detach())

        mu_features = obj_enc_out['mu_features']
        logvar_features = obj_enc_out['logvar_features']
        cropped_objects = obj_enc_out['cropped_objects']

        # reparameterize
        if deterministic:
            z_features = mu_features
        else:
            z_features = reparameterize(mu_features, logvar_features)

        encode_dict = {'mu': mu, 'logvar': logvar, 'z_base': z_base, 'z': z,
                       'mu_features': mu_features, 'logvar_features': logvar_features, 'z_features': z_features,
                       'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'obj_on': z_obj_on,
                       'psf_on_a': psf_on_a, 'psf_on_b': psf_on_b, 'psf_on': z_psf_on,
                       'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
                       'mu_defocus_offset': mu_defocus_offset, 'logvar_defocus_offset': logvar_defocus_offset,'defocus_base':defocus_base, 'z_defocus': z_defocus,
                       'cropped_objects': cropped_objects,
                       'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
                       'mu_offset': mu_offset, 'logvar_offset': logvar_offset}

        return encode_dict

    def encode_prior(self, x, x_prior=None, filtering_heuristic='variance', k=None):
        # encodes prior keypoints by patchifying the image and applying spatial-softmax
        if k is None:
            k = self.n_kp_prior
        if x_prior is None:
            x_prior = x
        kp_p, var_kp_p, defocus_p, max_corr_p, psfs = self.prior(x_prior, global_kp=True)
        kp_p = kp_p.view(x_prior.shape[0], -1, 2)  # [batch_size, n_kp_total, 2]
        var_kp_p = var_kp_p.view(x_prior.shape[0], kp_p.shape[1], -1)  # [batch_size, n_kp_total, 3]
        max_corr_p = max_corr_p.view(x_prior.shape[0], kp_p.shape[1])  # [batch_size, n_kp_total, 3]
        defocus_p = defocus_p.view(x_prior.shape[0], kp_p.shape[1])  # [batch_size, n_kp_total, 3]
        if filtering_heuristic == 'distance':
            # filter proposals by distance to the patches' center
            dist_from_center = self.prior.get_distance_from_patch_centers(kp_p, global_kp=True)
            _, indices = torch.topk(dist_from_center, k=k, dim=-1, largest=False)
            batch_indices = torch.arange(kp_p.shape[0]).view(-1, 1).to(kp_p.device)
            kp_p = kp_p[batch_indices, indices]
            max_corr_p = max_corr_p[batch_indices, indices]
            defocus_p = defocus_p[batch_indices, indices]
        else:
            # 'max_corr': keep the proposals correlating best with the PSF templates
            _, indices = torch.topk(max_corr_p, k=k, dim=-1, largest=True)
            batch_indices = torch.arange(kp_p.shape[0]).view(-1, 1).to(kp_p.device)
            kp_p = kp_p[batch_indices, indices]
            max_corr_p = max_corr_p[batch_indices, indices]
            defocus_p = defocus_p[batch_indices, indices]
        return kp_p, defocus_p, max_corr_p

    def translate_patches(self, kp_batch, patches_batch, scale=None, scale_normalized=False):
        """
        translate patches to be centered around given keypoints
        kp_batch: [bs, n_kp, 2] in [-1, 1]
        patches: [bs, n_kp, ch_patches, patch_size, patch_size]
        scale: None or [bs, n_kp, 2] or [bs, n_kp, 1]
        scale_normalized: False if scale is not in [0, 1]
        :return: translated_padded_patches [bs, n_kp, ch, img_size, img_size]
        """
        batch_size, n_kp, ch_patch, patch_size, _ = patches_batch.shape
        img_size = self.image_size
        if scale is None:
            z_scale = (patch_size / img_size) * torch.ones_like(kp_batch)
        else:
            # normalize to [0, 1]
            if scale_normalized:
                z_scale = scale
            else:
                z_scale = torch.sigmoid(scale)  # -> [0, 1]
        z_pos = kp_batch.reshape(-1, kp_batch.shape[-1])  # [bs * n_kp, 2]
        z_scale = z_scale.view(-1, z_scale.shape[-1])  # [bs * n_kp, 2]
        patches_batch = patches_batch.reshape(-1, *patches_batch.shape[2:])
        out_dims = (batch_size * n_kp, ch_patch, img_size, img_size)
        trans_patches_batch = spatial_transform(patches_batch, z_pos, z_scale, out_dims, inverse=True)
        trans_padded_patches_batch = trans_patches_batch.view(batch_size, n_kp, *trans_patches_batch.shape[1:])
        # [bs, n_kp, ch, img_size, img_size]
        return trans_padded_patches_batch


    def zero_close_coordinates_corr(self, coordinates, max_corr, distance_threshold=2/128*16):
        """
        Assign zero to max_corr values of close coordinates, keeping the ones with highest max_corr.

        Args:
        coordinates (torch.Tensor): Tensor of shape [batch_size, num_keypoints, 1, 2]
        max_corr (torch.Tensor): Tensor of shape [batch_size, num_keypoints]
        distance_threshold (float): Minimum distance between keypoints

        Returns:
        torch.Tensor: Updated max_corr tensor with zeroed values for close coordinates
        """
        batch_size, num_keypoints, _ = coordinates.shape

        # Reshape coordinates for easier calculations
        coords = coordinates.squeeze(2)  # Shape: [batch_size, num_keypoints, 2]

        # Create a copy of max_corr to modify
        updated_max_corr = max_corr.clone()

        for b in range(batch_size):
            for i in range(num_keypoints):
                # Calculate distances to all other points
                distances = torch.norm(coords[b] - coords[b, i], dim=1)

                # Find close points
                close_points = (distances < distance_threshold) & (distances >= 0)

                if close_points.any():
                    # Among close points, find the one with highest max_corr
                    close_corr = updated_max_corr[b, close_points]
                    best_idx = torch.argmax(close_corr)
                    best_corr = close_corr[best_idx]

                    # If current point doesn't have the highest max_corr, zero it out
                    if updated_max_corr[b, i] < best_corr:
                        updated_max_corr[b, i] = 0
                    else:
                        # If current point has highest max_corr, zero out all close points
                        updated_max_corr[b, close_points] = 0
                        updated_max_corr[b, i] = max_corr[b, i]  # Preserve original value

        return updated_max_corr

    def get_objects_alpha_intensity(self, z_kp, z_features, z_defocus, psf_on, z_scale=None, noisy=False):
        # decode the latent particles into [alpha, intensity] glimpses and place them on the canvas
        dec_objects = self.object_dec(z_features)  # [bs * n_kp, 2, patch_size, patch_size]
        dec_objects = dec_objects.view(-1, self.n_kp_enc,
                                       *dec_objects.shape[1:])  # [bs, n_kp, 2, patch_size, patch_size]
        # generate PSF
        z_kp_um = z_kp[:, :, [1, 0]] * (self.image_size//2 * self.optics_dict['pixel_size_CCD']/self.optics_dict['M'])
        if self.psf_model == '2d':
            PSFs = self.physical_layer(z_kp_um)
        else:
            PSFs = self.physical_layer(z_kp_um, z_defocus)

        if self.zero_close_psfs_flag:
            psf_on_new = self.zero_close_coordinates_corr(z_kp, psf_on, distance_threshold=self.zero_close_psfs_flag*2/self.image_size)
        else:
            psf_on_new = psf_on

        # translate patches - place the decoded glimpses on the canvas
        PSFs_trans = PSFs * psf_on_new[:, :, None, None]

        dec_objects_trans = self.translate_patches(z_kp, dec_objects, scale=z_scale)
        dec_objects_trans = dec_objects_trans.clamp(0, 1)  # STN can change values to be < 0


        # dec_objects_trans: [bs, n_kp, 2, im_size, im_size]
        a_ob, obj = torch.split(dec_objects_trans, [1, 1], dim=2)

        intensity_obj = obj ; a_obj = a_ob

        intensity_obj = intensity_obj.clamp(0, 1)

        if noisy:
            a_obj = a_obj + 0.1 * torch.randn_like(a_obj)
            a_obj = a_obj.clamp(0, 1)

        return dec_objects, a_obj, intensity_obj, PSFs_trans.sum(dim=1, keepdim=True), PSFs_trans

    def get_objects_alpha_intensity_with_depth(self, a_obj, intensity_obj, obj_on, z_depth, eps=1e-5):
        # stitching the glimpses by factoring the alpha maps and the particle's inferred depth
        # obj_on: [bs, n_kp, 1]
        # z_depth: [bs, n_kp, 1]
        # turn off inactive particles
        a_obj = obj_on[:, :, None, None, None] * a_obj  # [bs, n_kp, 1, im_size, im_size]
        alpha_intensity_obj = a_obj * intensity_obj
        # importance map
        importance_map = a_obj * torch.sigmoid(-z_depth[:, :, :, None, None])
        # normalize
        importance_map = importance_map / (torch.sum(importance_map, dim=1, keepdim=True) + eps)
        # this imitates softmax to move objects on the depth axis
        dec_objects_trans = alpha_intensity_obj * importance_map
        alpha_mask = 1.0 - (importance_map * a_obj).sum(dim=1)
        a_obj = importance_map * a_obj
        return a_obj, alpha_mask, dec_objects_trans

    def decode_objects(self, z_kp, z_features, obj_on, z_defocus, psf_on, z_scale=None, noisy=False, z_depth=None):
        # stitching the decoded latent particles -> image, factoring the alpha maps and depths
        dec_objects, a_obj, intensity_obj, PSFs, PSFs_trans = self.get_objects_alpha_intensity(z_kp, z_features, z_defocus, psf_on, z_scale=z_scale,
                                                                 noisy=noisy)
        alpha_masks, bg_mask, dec_objects_trans = self.get_objects_alpha_intensity_with_depth(a_obj, intensity_obj, obj_on=obj_on,
                                                                                        z_depth=z_depth)
        return dec_objects, dec_objects_trans, alpha_masks, bg_mask, PSFs

    def decode_all(self, z, z_features, obj_on, z_defocus, psf_on, z_depth=None, noisy=False, z_scale=None):
        # a wrapper function to decode latent particles into an image (no bg)
        object_dec_out = self.decode_objects(z, z_features, obj_on, z_defocus, psf_on, noisy=noisy, z_depth=z_depth, z_scale=z_scale)
        dec_objects, dec_objects_trans, alpha_masks, bg_mask, PSFs = object_dec_out

        decoder_out = {'dec_objects': dec_objects, 'dec_objects_trans': dec_objects_trans,
                       'bg_mask': bg_mask, 'alpha_masks': alpha_masks, 'PSFs':PSFs}

        return decoder_out

    def generate_decoded_image(self, z_kp, z_psf, z_features, obj_on, z_defocus, psf_on, z_depth=None, noisy=False, z_scale=None,
                            eps=1e-5):
        # Decode the latent particles into [alpha, intensity] glimpses
        dec_objects = self.object_dec(z_features)
        dec_objects = dec_objects.view(-1, self.n_kp_enc, *dec_objects.shape[1:])

        # Generate PSF
        z_psf_um = z_psf[:, :, [1, 0]] * (self.image_size//2 * self.optics_dict['pixel_size_CCD']/self.optics_dict['M'])
        if self.psf_model == '2d':
            PSFs = self.physical_layer(z_psf_um)
        else:
            PSFs = self.physical_layer(z_psf_um, z_defocus)

        if self.zero_close_psfs_flag:
            psf_on_new = self.zero_close_coordinates_corr(z_psf, psf_on,
                                                          distance_threshold=self.zero_close_psfs_flag * 2 / self.image_size)
        else:
            psf_on_new = psf_on

        PSFs_trans = PSFs * psf_on_new[:, :, None, None]

        dec_objects_trans = self.translate_patches(z_kp, dec_objects, scale=z_scale)
        dec_objects_trans = dec_objects_trans.clamp(0, 1)

        # Split objects into alpha and intensity
        a_ob, obj = torch.split(dec_objects_trans, [1, 1], dim=2)
        intensity_obj = obj.clamp(0, 1)
        a_obj = a_ob

        # Add noise if requested
        if noisy:
            a_obj = (a_obj + 0.1 * torch.randn_like(a_obj)).clamp(0, 1)

        # Process alpha and intensity with depth
        # Turn off inactive particles
        a_obj = obj_on[:, :, None, None, None] * a_obj
        alpha_intensity_obj = a_obj * intensity_obj

        # Create importance map using depth
        importance_map = a_obj * torch.sigmoid(-z_depth[:, :, :, None, None])
        importance_map = importance_map / (torch.sum(importance_map, dim=1, keepdim=True) + eps)

        # Final composition
        dec_objects_trans_final = alpha_intensity_obj * importance_map
        alpha_masks = importance_map * a_obj
        bg_mask = 1.0 - (importance_map * a_obj).sum(dim=1)

        return {
            'dec_objects': dec_objects,
            'dec_objects_trans': dec_objects_trans_final,
            'bg_mask': bg_mask,
            'alpha_masks': alpha_masks,
            'PSFs': PSFs_trans.sum(dim=1, keepdim=True),
            'psf_on': psf_on_new
        }

    def forward(self, x, deterministic=False, x_prior=None, warmup=False, noisy=False,
                cropped_objects_prev=None, mu_scale_prev=None, train_prior=True, refinement_iter=False):
        # refinement_iter: do another encoding step to get a better lock on the object's position (z_a + z_o)
        # first, extract prior KP proposals
        # prior proposals
        kp_p, defocus_p, max_corr_p = self.encode_prior(x, x_prior=x_prior, filtering_heuristic=self.filtering_heuristic)
        kp_init = kp_p if train_prior else (0.0 * kp_p + kp_p.detach())  # 0.0 * kp_p is because of distributed training
        encoder_out = self.encode_all(x, deterministic=deterministic, noisy=noisy, warmup=warmup, kp_init=kp_init, with_offset=True,
                                      cropped_objects_prev=cropped_objects_prev, scale_prev=mu_scale_prev, defocus_init=defocus_p, max_corr=max_corr_p,
                                      refinement_iter=refinement_iter)
        # detach for the kl-divergence
        kp_p = kp_p.detach()
        mu = encoder_out['mu']
        logvar = encoder_out['logvar']
        z_base = encoder_out['z_base']
        z = encoder_out['z']
        mu_offset = encoder_out['mu_offset']
        logvar_offset = encoder_out['logvar_offset']
        mu_features = encoder_out['mu_features']
        logvar_features = encoder_out['logvar_features']
        z_features = encoder_out['z_features']
        obj_on = encoder_out['obj_on']
        obj_on_a = encoder_out['obj_on_a']
        obj_on_b = encoder_out['obj_on_b']
        psf_on = encoder_out['psf_on']
        psf_on_a = encoder_out['psf_on_a']
        psf_on_b = encoder_out['psf_on_b']
        mu_depth = encoder_out['mu_depth']
        logvar_depth = encoder_out['logvar_depth']
        z_depth = encoder_out['z_depth']
        mu_defocus = encoder_out['mu_defocus_offset']
        logvar_defocus = encoder_out['logvar_defocus_offset']
        z_defocus = encoder_out['z_defocus']
        cropped_objects = encoder_out['cropped_objects']
        mu_scale = encoder_out['mu_scale']
        logvar_scale = encoder_out['logvar_scale']
        z_scale = encoder_out['z_scale']

        obj_on_sample = obj_on

        decoder_out = self.decode_all(z, z_features, obj_on_sample, z_defocus, psf_on, z_depth, noisy=noisy, z_scale=z_scale)
        dec_objects = decoder_out['dec_objects']
        dec_objects_trans = decoder_out['dec_objects_trans']
        bg_mask = decoder_out['bg_mask']
        alpha_masks = decoder_out['alpha_masks']
        PSFs = decoder_out['PSFs']


        output_dict = {}
        output_dict['kp_p'] = kp_p
        output_dict['mu'] = mu
        output_dict['logvar'] = logvar
        output_dict['z_base'] = z_base
        output_dict['z'] = z
        output_dict['mu_offset'] = mu_offset
        output_dict['logvar_offset'] = logvar_offset
        output_dict['mu_features'] = mu_features
        output_dict['logvar_features'] = logvar_features
        output_dict['z_features'] = z_features
        output_dict['bg_mask'] = bg_mask
        output_dict['cropped_objects_original'] = cropped_objects
        output_dict['obj_on_a'] = obj_on_a
        output_dict['obj_on_b'] = obj_on_b
        output_dict['obj_on'] = obj_on
        output_dict['psf_on_a'] = psf_on_a
        output_dict['psf_on_b'] = psf_on_b
        output_dict['psf_on'] = psf_on
        output_dict['dec_objects_original'] = dec_objects
        output_dict['dec_objects'] = dec_objects_trans
        output_dict['mu_depth'] = mu_depth
        output_dict['logvar_depth'] = logvar_depth
        output_dict['z_depth'] = z_depth
        output_dict['mu_defocus'] = mu_defocus
        output_dict['logvar_defocus'] = logvar_defocus
        output_dict['z_defocus'] = z_defocus
        output_dict['mu_scale'] = mu_scale
        output_dict['logvar_scale'] = logvar_scale
        output_dict['z_scale'] = z_scale
        output_dict['alpha_masks'] = alpha_masks
        output_dict['PSFs'] = PSFs

        return output_dict


class BgDLP(nn.Module):
    def __init__(self, cdim=3, enc_channels=(16, 16, 32), image_size=64, pad_mode='replicate', dropout=0.0,
                 learned_feature_dim=16, n_kp_enc=10, use_resblock=False):
        """
        DLP Background Module -- encode a latent for the (masked) background, z_bg
        Basically, just a convolutional-based encoder used in standard VAEs
        cdim: channels of the input image (3...)
        enc_channels: channels for the posterior CNN (takes in the whole image)
        pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
        learned_feature_dim: the latent visual features dimensions extracted from glimpses.
        """
        super(BgDLP, self).__init__()
        self.image_size = image_size
        self.dropout = dropout
        self.features_dim = int(np.ceil(image_size / (2 ** (len(enc_channels) - 1))))
        self.learned_feature_dim = learned_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.cdim = cdim
        self.n_kp_enc = 32
        self.use_resblock = use_resblock

        # encoder
        self.bg_cnn_enc = KeyPointCNNOriginal(cdim=cdim, channels=enc_channels, image_size=image_size,
                                              n_kp=self.n_kp_enc,
                                              pad_mode=pad_mode, use_resblock=self.use_resblock)
        bg_enc_output_dim = self.learned_feature_dim * 2  # [mu_features, sigma_features]
        self.bg_enc = nn.Sequential(nn.Linear(self.n_kp_enc * self.features_dim ** 2, 256),
                                    nn.ReLU(True),
                                    nn.Linear(256, 256),
                                    nn.ReLU(True),
                                    nn.Linear(256, bg_enc_output_dim))

        # decoder
        decoder_n_kp = self.n_kp_enc
        self.latent_to_feat_map = FCToCNN(target_hw=self.features_dim, n_ch=decoder_n_kp,
                                          features_dim=self.learned_feature_dim, pad_mode=pad_mode,
                                          use_resblock=self.use_resblock)
        self.dec = CNNDecoder(cdim=cdim, channels=enc_channels, image_size=image_size, in_ch=decoder_n_kp,
                              pad_mode=pad_mode, use_resblock=self.use_resblock)
        self.init_weights()
        self.crop_layer = Croplayer(W=self.image_size)

    def get_parameters(self, prior=True, encoder=True, decoder=True):
        parameters = []
        if encoder:
            parameters.extend(list(self.bg_cnn_enc.parameters()))
            parameters.extend(list(self.bg_enc.parameters()))
        if decoder:
            parameters.extend(list(self.dec.parameters()))
            parameters.extend(list(self.latent_to_feat_map.parameters()))
        return parameters


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # use pytorch's default
                pass

    def encode_bg_features(self, x, masks=None):
        # x: [bs, ch, image_size, image_size]
        # masks: [bs, 1, image_size, image_size]
        batch_size, _, features_dim, _ = x.shape
        # bg features
        if masks is not None:
            x_in = x * masks
        else:
            x_in = x
        _, cnn_features = self.bg_cnn_enc(x_in)
        cnn_features = cnn_features.view(batch_size, -1)  # flatten
        bg_enc_out = self.bg_enc(cnn_features)  # [bs,, 2 * learned_features_dim]
        mu_bg, logvar_bg = bg_enc_out.chunk(2, dim=-1)

        return mu_bg, logvar_bg

    def encode_all(self, x, masks=None, deterministic=True):
        # encode background
        mu_bg, logvar_bg = self.encode_bg_features(x, masks)
        if deterministic:
            z_bg = mu_bg
        else:
            z_bg = reparameterize(mu_bg, logvar_bg)

        z_kp = torch.zeros(mu_bg.shape[0], 1, 2, device=x.device, dtype=torch.float)
        encode_dict = {'mu_bg': mu_bg, 'logvar_bg': logvar_bg, 'z_bg': z_bg, 'z_kp': z_kp}
        return encode_dict

    def decode_all(self, z_features):
        feature_maps = self.latent_to_feat_map(z_features)
        bg_rec = self.dec(feature_maps)
        if bg_rec.shape[-1] > self.image_size:
            bg_rec_out = self.crop_layer(bg_rec)
        else:
            bg_rec_out = bg_rec
        return bg_rec_out


    def forward(self, x, masks=None, deterministic=False):
        encoder_out = self.encode_all(x, masks, deterministic)
        mu_bg = encoder_out['mu_bg']
        logvar_bg = encoder_out['logvar_bg']
        z_bg = encoder_out['z_bg']
        z_kp = encoder_out['z_kp']
        bg_rec = self.decode_all(z_bg)
        output_dict = {'mu_bg': mu_bg, 'logvar_bg': logvar_bg, 'z_bg': z_bg, 'z_kp': z_kp, 'bg_rec': bg_rec}
        return output_dict


class ReadNoiseDLP(nn.Module):
    def __init__(self, cdim=3, enc_channels=(16, 16, 32), image_size=64, pad_mode='replicate', dropout=0.0,
                 learned_feature_dim=16, n_kp_enc=10, use_resblock=False):
        super(ReadNoiseDLP, self).__init__()
        self.image_size = image_size
        self.dropout = dropout
        self.features_dim = int(np.ceil(image_size / (2 ** (len(enc_channels) - 1))))
        self.learned_feature_dim = learned_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.cdim = cdim
        self.n_kp_enc = 32
        self.use_resblock = use_resblock

        # encoder
        self.read_noise_cnn_enc = KeyPointCNNOriginal(cdim=cdim, channels=enc_channels, image_size=image_size,
                                              n_kp=self.n_kp_enc,
                                              pad_mode=pad_mode, use_resblock=self.use_resblock)
        read_noise_enc_output_dim = self.learned_feature_dim * 2  # [mu_features, sigma_features]
        self.read_noise_enc = nn.Sequential(nn.Linear(self.n_kp_enc * self.features_dim ** 2, 256),
                                    nn.ReLU(True),
                                    nn.Linear(256, 256),
                                    nn.ReLU(True),
                                    nn.Linear(256, read_noise_enc_output_dim))

        self.init_weights()
        self.crop_layer = Croplayer(W=self.image_size)

    def get_parameters(self, prior=True, encoder=True):
        parameters = []
        if encoder:
            parameters.extend(list(self.read_noise_cnn_enc.parameters()))
            parameters.extend(list(self.read_noise_enc.parameters()))

        return parameters


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # use pytorch's default
                pass

    def encode_read_noise(self, x):
        # x: [bs, ch, image_size, image_size]
        # masks: [bs, 1, image_size, image_size]
        batch_size, _, features_dim, _ = x.shape
        # bg features
        _, cnn_features = self.read_noise_cnn_enc(x)
        cnn_features = cnn_features.view(batch_size, -1)  # flatten
        read_noise_enc_out = self.read_noise_enc(cnn_features)  # [bs,, 2 * learned_features_dim]
        mu_read_noise, logvar_read_noise = read_noise_enc_out.chunk(2, dim=-1)

        return mu_read_noise, logvar_read_noise

    def encode_all(self, x, deterministic=False):
        mu_read_noise, logvar_read_noise = self.encode_read_noise(x)
        if deterministic:
            z_read_noise = mu_read_noise
        else:
            z_read_noise = reparameterize(mu_read_noise, logvar_read_noise)/10
        encode_dict = {'mu_read_noise': mu_read_noise, 'logvar_read_noise': logvar_read_noise, 'z_read_noise': z_read_noise}
        return encode_dict

    def forward(self, x, deterministic=False):
        encoder_out = self.encode_all(x, deterministic)
        mu_read_noise = encoder_out['mu_read_noise']
        logvar_read_noise = encoder_out['logvar_read_noise']
        z_read_noise= encoder_out['z_read_noise']
        output_dict = {'mu_read_noise': mu_read_noise, 'logvar_read_noise': logvar_read_noise, 'z_read_noise': z_read_noise}
        return output_dict

# single-image deep latent particles
class PILPEL(nn.Module):
    def __init__(self, optics_dict, cdim=3, enc_channels=(16, 16, 32), prior_channels=(16, 16, 32), image_size=64, n_kp=1,
                 pad_mode='replicate', sigma=1.0, dropout=0.0,
                 patch_size=16, n_kp_enc=20, n_kp_prior=20, learned_feature_dim=16,
                 bg_learned_feature_dim=None,
                 kp_range=(-1, 1), kp_activation="tanh", anchor_s=0.25, read_noise=False,
                 use_resblock=False, scale_std=0.3, offset_std=0.2, obj_on_alpha=0.1, obj_on_beta=0.1,
                 use_correlation_heatmaps=False, filtering_heuristic='variance', zero_close_psfs_flag=True):
        """
        cdim: channels of the input image (3...)
        enc_channels: channels for the posterior CNN (takes in the whole image)
        prior_channels: channels for prior CNN (takes in patches)
        n_kp: number of kp to extract from each (!) patch
        n_kp_prior: number of kp to filter from the set of prior kp (of size n_kp x num_patches)
        n_kp_enc: number of posterior kp to be learned (this is the actual number of kp that will be learnt)
        pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
        sigma: the prior std of the KP
        dropout: dropout for the CNNs. We don't use it though...
        patch_size: patch size for the prior KP proposals network (not to be confused with the glimpse size)
        kp_range: the range of keypoints, can be [-1, 1] (default) or [0,1]
        learned_feature_dim: the latent visual features dimensions extracted from glimpses.
        bg_learned_feature_dim: the latent visual features dimensions extracted from masked bg. None-> =learned_feature_dim
        kp_activation: the type of activation to apply on the keypoints: "tanh" for kp_range [-1, 1], "sigmoid" for [0, 1]
        anchor_s: defines the glimpse size as a ratio of image_size (e.g., 0.25 for image_size=128 -> glimpse_size=32)
        scale_std: prior std for the scale
        offset_std: prior std for the offset
        obj_on_alpha: prior alpha (Beta distribution) for obj_on
        obj_on_beta: prior beta (Beta distribution) for obj_on
        use_correlation_heatmaps: use correlation heatmaps in the particle encoder
        filtering heuristic: filtering heuristic to filter prior keypoints, ['distance', 'max_corr']
        """
        super(PILPEL, self).__init__()
        self.image_size = image_size
        self.sigma = sigma
        self.dropout = dropout
        self.kp_range = kp_range
        self.n_kp = n_kp
        self.n_kp_total = n_kp_prior
        self.n_kp_prior = n_kp_prior
        self.n_kp_enc = n_kp_enc
        self.kp_activation = kp_activation
        self.patch_size = patch_size
        self.features_dim = int(image_size // (2 ** (len(enc_channels) - 1)))
        self.learned_feature_dim = learned_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.bg_learned_feature_dim = learned_feature_dim if bg_learned_feature_dim is None else bg_learned_feature_dim
        assert self.bg_learned_feature_dim > 0, "bg_learned_feature_dim must be greater than 0"
        self.anchor_s = anchor_s
        self.obj_patch_size = np.round(anchor_s * image_size).astype(int)
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.use_correlation_heatmaps = use_correlation_heatmaps
        self.read_noise = read_noise
        self.zero_close_psfs_flag = zero_close_psfs_flag
        assert filtering_heuristic in ['distance',
                                       'max_corr'], f'unknown filtering heuristic: {filtering_heuristic}'
        self.filtering_heuristic = filtering_heuristic
        self.optics_dict = optics_dict

        # priors
        self.register_buffer('logvar_kp', torch.log(torch.tensor(sigma ** 2)))
        self.register_buffer('mu_scale_prior',
                             torch.tensor(np.log(0.75 * self.anchor_s / (1 - 0.75 * self.anchor_s + 1e-5))))
        self.register_buffer('logvar_scale_p', torch.log(torch.tensor(scale_std ** 2)))
        self.register_buffer('logvar_offset_p', torch.log(torch.tensor(offset_std ** 2)))
        self.register_buffer('obj_on_a_p', torch.tensor(obj_on_alpha))
        self.register_buffer('obj_on_b_p', torch.tensor(obj_on_beta))

        # foreground module
        self.fg_module = FgDLP(self.optics_dict, cdim=cdim, enc_channels=enc_channels, prior_channels=prior_channels,
                               image_size=image_size, n_kp=n_kp, pad_mode=pad_mode,
                               sigma=sigma, dropout=dropout, patch_size=patch_size, n_kp_enc=n_kp_enc,
                               n_kp_prior=n_kp_prior, learned_feature_dim=learned_feature_dim, kp_range=kp_range,
                               kp_activation=kp_activation, anchor_s=anchor_s,
                               use_resblock=self.use_resblock,
                               use_correlation_heatmaps=use_correlation_heatmaps,
                               filtering_heuristic=filtering_heuristic, zero_close_psfs_flag=self.zero_close_psfs_flag)

        # background module
        self.bg_module = BgDLP(cdim=cdim, enc_channels=enc_channels, image_size=image_size, pad_mode=pad_mode,
                               dropout=dropout, learned_feature_dim=self.bg_learned_feature_dim, n_kp_enc=n_kp_enc,
                               use_resblock=self.use_resblock)
        if read_noise:
            self.read_noise_module = ReadNoiseDLP(cdim=cdim, enc_channels=enc_channels, image_size=image_size, pad_mode=pad_mode,
                                   dropout=dropout, learned_feature_dim=1, n_kp_enc=n_kp_enc,
                                   use_resblock=self.use_resblock)
        self.init_weights()
        self.norm01 = Normalize01()
        self.blur = torchvision.transforms.GaussianBlur(15, sigma=5)

    def get_parameters(self, prior=True, encoder=True, decoder=True):
        parameters = []
        parameters.extend(self.fg_module.get_parameters(prior, encoder, decoder))
        parameters.extend(self.bg_module.get_parameters(prior, encoder, decoder))
        return parameters


    def init_weights(self):
        self.fg_module.init_weights()
        self.bg_module.init_weights()

    def info(self):
        log_str = f'prior (patch) kp filtering: {self.n_kp_total} -> prior kp: {self.n_kp_prior}\n'
        log_str += f'prior (patch) kp filtering method: {self.filtering_heuristic}\n'
        log_str += f'prior patch size: {self.patch_size}\n'
        log_str += f'# posterior particles: {self.n_kp_enc}\n'
        log_str += f'particle visual feature dim: {self.learned_feature_dim}\n'
        log_str += f'bg visual feature dim: {self.bg_learned_feature_dim}\n'
        log_str += f'posterior object patch size: {self.obj_patch_size}\n'
        # cnns output sizes
        # prior
        prior_cnn_size = self.fg_module.prior.enc.conv_output_size
        # posterior - objects
        fg_cnn_size = self.fg_module.particle_attribute_enc.cnn.conv_output_size
        # posterior - bg
        bg_cnn_size = self.bg_module.bg_cnn_enc.conv_output_size
        log_str += f'cnn pre-pool out sizes: prior-{prior_cnn_size}, obj enc-{fg_cnn_size}, bg enc-{bg_cnn_size}\n'
        # object decoder
        dec_upsample_l = int(np.log(self.fg_module.object_dec.patch_size[0]) // np.log(2)) - 3
        log_str += f'object decoder # upsampling layers: {dec_upsample_l}\n'
        log_str += f'correlation maps enabled: {self.use_correlation_heatmaps}\n'
        # num parameters and model size
        size_dict = calc_model_size(self)
        size_mb = size_dict['size_mb']
        n_params = size_dict['n_params']
        log_str += f'trainable parameters: {n_params} ({n_params / (10 ** 6):.4f}M)\n'
        log_str += f'estimated size on disk: {size_mb:.3f}MB\n'
        return log_str


    def decode_all_generate_image(self, z, z_psf, z_features, z_bg, obj_on, z_defocus, psf_on, z_read_noise, z_depth=None, noisy=False, z_scale=None, blur_bg=False, norm01=True):
        # a wrapper function to convert latent particles to an image

        # foreground
        fg_dict = self.fg_module.generate_decoded_image(z, z_psf, z_features, obj_on, z_defocus, psf_on, z_depth=z_depth, noisy=noisy, z_scale=z_scale)
        dec_objects = fg_dict['dec_objects']
        dec_objects_trans = fg_dict['dec_objects_trans']
        alpha_masks = fg_dict['alpha_masks']
        PSFs = fg_dict['PSFs']
        bg_mask = fg_dict['bg_mask']

        # background
        bg_mask = ((1. - PSFs).clamp(0, 1))
        bg = self.bg_module.decode_all(z_bg)
        if blur_bg:
            bg = self.blur(bg)

        # stitching
        rec = bg_mask * (bg + self.blur(dec_objects_trans.sum(dim=1))) + PSFs

        if self.read_noise:
            noise = torch.abs(z_read_noise).view(-1, 1, 1, 1) * torch.randn_like(rec)
            rec = torch.max(rec + noise, torch.FloatTensor([0.0]).to(rec.device))
        else:
            rec = rec

        if norm01:
            rec = self.norm01(rec)
        else:
            rec = rec

        total_bg = (bg + self.blur(dec_objects_trans.sum(dim=1)))

        decoder_out = {'rec': rec, 'dec_objects': dec_objects, 'dec_objects_trans': dec_objects_trans,
                       'bg': bg, 'bg_mask': bg_mask, 'alpha_masks': alpha_masks, 'PSFs': PSFs,'total_bg':total_bg,
                       'psf_on': fg_dict['psf_on']}

        return decoder_out


    def forward(self, x, norm01=False, deterministic=False, x_prior=None, warmup=False, noisy=False,
                train_enc_prior=True):
        fg_dict = self.fg_module(x, deterministic=deterministic, x_prior=x_prior, warmup=warmup, noisy=noisy,
                                 train_prior=train_enc_prior, refinement_iter=False)
        # encoder
        kp_p = fg_dict['kp_p']
        mu = fg_dict['mu']
        logvar = fg_dict['logvar']
        z_base = fg_dict['z_base']
        z = fg_dict['z']
        mu_offset = fg_dict['mu_offset']
        logvar_offset = fg_dict['logvar_offset']
        mu_features = fg_dict['mu_features']
        z_features = fg_dict['z_features']
        logvar_features = fg_dict['logvar_features']
        cropped_objects = fg_dict['cropped_objects_original']
        obj_on_a = fg_dict['obj_on_a']
        obj_on_b = fg_dict['obj_on_b']
        z_obj_on = fg_dict['obj_on']
        psf_on_a = fg_dict['psf_on_a']
        psf_on_b = fg_dict['psf_on_b']
        z_psf_on = fg_dict['psf_on']
        mu_depth = fg_dict['mu_depth']
        logvar_depth = fg_dict['logvar_depth']
        z_depth = fg_dict['z_depth']
        mu_defocus = fg_dict['mu_defocus']
        logvar_defocus = fg_dict['logvar_defocus']
        z_defocus = fg_dict['z_defocus']
        mu_scale = fg_dict['mu_scale']
        logvar_scale = fg_dict['logvar_scale']
        z_scale = fg_dict['z_scale']

        # decoder
        bg_mask = fg_dict['bg_mask']
        dec_objects = fg_dict['dec_objects_original']
        dec_objects_trans = fg_dict['dec_objects']
        alpha_masks = fg_dict['alpha_masks']
        PSFs = fg_dict['PSFs']

        bg_mask = ((1. - PSFs).clamp(0, 1))
        bg_enc_mask = bg_mask
        bg_dict = self.bg_module(x, bg_enc_mask, deterministic=False)
        mu_bg = bg_dict['mu_bg']
        logvar_bg = bg_dict['logvar_bg']
        z_bg = bg_dict['z_bg']
        z_kp_bg = bg_dict['z_kp']
        bg = bg_dict['bg_rec']

        # stitch
        rec = bg_mask * (bg + self.blur(dec_objects_trans.sum(dim=1))) + PSFs

        if self.read_noise and not warmup:
            read_noise_dict = self.read_noise_module(x, deterministic=False) # currently fixed encoder, future version will include learned noise
            mu_read_noise = read_noise_dict['mu_read_noise']
            logvar_read_noise = read_noise_dict['logvar_read_noise']
            z_read_noise = read_noise_dict['z_read_noise']

            noise = torch.abs(z_read_noise).view(-1, 1, 1, 1) * torch.randn_like(rec)
            rec_ = torch.max(rec + noise, torch.FloatTensor([0.0]).to(rec.device))

            noisy_bg_ = torch.max(bg_mask * (bg + self.blur(dec_objects_trans.sum(dim=1))) + noise, torch.FloatTensor([0.0]).to(rec.device))
            Nbatch = noisy_bg_.size(0)
            Nparticles = noisy_bg_.size(1)
            noisy_bg = torch.zeros_like(noisy_bg_)
            for i in range(Nbatch):
                for j in range(Nparticles):
                    min_val = (rec_[i, j, :, :]).min()
                    max_val = (rec_[i, j, :, :]).max()
                    noisy_bg[i, j, :, :] = (noisy_bg_[i, j, :, :] - min_val) / (max_val - min_val + 1e-6)

        else:
            rec_ = rec

        if norm01:
            rec01 = self.norm01(rec_)
        else:
            rec01 = rec_

        output_dict = {}
        output_dict['kp_p'] = kp_p
        output_dict['rec'] = rec01
        output_dict['rec_clean'] = rec
        output_dict['mu'] = mu
        output_dict['logvar'] = logvar
        output_dict['z'] = z
        output_dict['z_base'] = z_base
        output_dict['z_kp_bg'] = z_kp_bg
        output_dict['mu_offset'] = mu_offset
        output_dict['logvar_offset'] = logvar_offset
        output_dict['mu_features'] = mu_features
        output_dict['logvar_features'] = logvar_features
        output_dict['z_features'] = z_features
        output_dict['bg'] = bg
        output_dict['bg_mask'] = bg_mask
        output_dict['mu_bg'] = mu_bg
        output_dict['logvar_bg'] = logvar_bg
        output_dict['z_bg'] = z_bg
        # object stuff
        output_dict['cropped_objects_original'] = cropped_objects
        output_dict['obj_on_a'] = obj_on_a
        output_dict['obj_on_b'] = obj_on_b
        output_dict['obj_on'] = z_obj_on
        output_dict['psf_on_a'] = psf_on_a
        output_dict['psf_on_b'] = psf_on_b
        output_dict['psf_on'] = z_psf_on
        output_dict['dec_objects_original'] = dec_objects
        output_dict['dec_objects_warmup'] = dec_objects_trans
        output_dict['dec_objects'] = dec_objects_trans.sum(dim=1)
        output_dict['blurred_dec_objects'] = self.blur(dec_objects_trans.sum(dim=1))
        output_dict['mu_depth'] = mu_depth
        output_dict['logvar_depth'] = logvar_depth
        output_dict['z_depth'] = z_depth
        output_dict['mu_defocus'] = mu_defocus
        output_dict['logvar_defocus'] = logvar_defocus
        output_dict['z_defocus'] = z_defocus
        output_dict['mu_scale'] = mu_scale
        output_dict['logvar_scale'] = logvar_scale
        output_dict['z_scale'] = z_scale
        output_dict['alpha_masks'] = alpha_masks
        output_dict['PSFs'] = PSFs

        if self.read_noise and not warmup:
            output_dict['mu_read_noise'] = mu_read_noise
            output_dict['logvar_read_noise'] = logvar_read_noise
            output_dict['z_read_noise'] = z_read_noise
            output_dict['noisy_bg'] = noisy_bg

        return output_dict


    def calc_elbo(self, x, model_output, warmup=False, beta_kl=0.05, beta_rec=1.0, kl_balance=0.001,
                  recon_loss_type="mse", recon_loss_func=None, noisy=False):
        # x: [batch_size, ch, h, w]
        # define losses
        kl_loss_func = ChamferLossKL(use_reverse_kl=False)
        if recon_loss_type == "vgg":
            if recon_loss_func is None:
                recon_loss_func = VGGDistance(device=x.device)
        else:
            recon_loss_func = calc_reconstruction_loss

        # unpack output
        mu_p = model_output['kp_p']
        mu = model_output['mu']
        logvar = model_output['logvar']
        z = model_output['z']
        mu_offset = model_output['mu_offset']
        logvar_offset = model_output['logvar_offset']
        rec_x = model_output['rec']
        mu_features = model_output['mu_features']
        logvar_features = model_output['logvar_features']
        mu_bg = model_output['mu_bg']
        logvar_bg = model_output['logvar_bg']
        mu_scale = model_output['mu_scale']
        logvar_scale = model_output['logvar_scale']
        z_scale = model_output['z_scale']
        mu_depth = model_output['mu_depth']
        logvar_depth = model_output['logvar_depth']
        mu_defocus = model_output['mu_defocus']
        logvar_defocus = model_output['logvar_defocus']
        # object stuff
        dec_objects_original = model_output['dec_objects_original']
        obj_on_a = model_output['obj_on_a']  # [batch_size, n_kp]
        obj_on_b = model_output['obj_on_b']  # [batch_size, n_kp]
        psf_on = model_output['psf_on']  # [batch_size, n_kp]
        rec = model_output['rec']

        batch_size = x.shape[0]
        if len(x.shape) == 5:
            x = x.view(-1, *x.shape[2:])

        # --- reconstruction error --- #
        if dec_objects_original is not None and warmup:
            z_pos = z.reshape(-1, z.shape[-1])
            z_scale = z_scale.view(-1, z_scale.shape[-1]) * 0 + self.anchor_s
            out_dims = (batch_size * z.shape[1], x.shape[1], self.fg_module.patch_size, self.fg_module.patch_size)
            x_repeated = x.unsqueeze(1).repeat(1, z.shape[1], 1, 1, 1)  # [batch_size, n_kp, ch, image_size, image_size]
            idx_on = psf_on.view(-1)>=0
            x_repeated = x_repeated * psf_on[:, :, None, None, None]
            x_repeated = x_repeated.view(-1, *x.shape[1:])  # [batch_size * n_kp, ch, image_size, image_size]
            cropped_objects = spatial_transform(x_repeated, z_pos, z_scale, out_dims, inverse=False)  # [batch_size * n_kp, ch, patch_size, patch_size]
            cropped_objects = cropped_objects[idx_on]

            rec_repeated = rec.unsqueeze(1).repeat(1, z.shape[1], 1, 1, 1)  # [batch_size, n_kp, ch, image_size, image_size]
            rec_repeated = rec_repeated * psf_on[:, :, None, None, None]
            rec_repeated = rec_repeated.view(-1, *x.shape[1:])  # [batch_size * n_kp, ch, image_size, image_size]
            cropped_rec = spatial_transform(rec_repeated, z_pos, z_scale, out_dims, inverse=False)  # [batch_size * n_kp, ch, patch_size, patch_size]
            cropped_rec = cropped_rec[idx_on]

            if recon_loss_type == "vgg":
                loss_rec_obj = recon_loss_func(cropped_objects, cropped_rec, reduction="mean")
            else:
                loss_rec_obj = calc_reconstruction_loss(cropped_objects, cropped_rec, loss_type='mse', reduction='mean')
            loss_rec = loss_rec_obj + (0 * rec_x).mean()  # + (0 * rec_x).mean() for distributed training
            psnr = torch.tensor(0.0, dtype=torch.float, device=x.device)
        else:
            if recon_loss_type == "vgg":
                loss_rec = recon_loss_func(x, rec_x, reduction="mean")
            else:
                loss_rec = calc_reconstruction_loss(x, rec_x, loss_type='mse', reduction='mean')

            with torch.no_grad():
                psnr = -10 * torch.log10(F.mse_loss(rec_x, x))
        # --- end reconstruction error --- #

        # --- define priors --- #
        warmup_logvar = torch.log(torch.tensor(0.1 ** 2))
        logvar_kp = self.logvar_kp.expand_as(mu_p)
        logvar_offset_p = warmup_logvar if (warmup or noisy) else self.logvar_offset_p
        logvar_scale_p = warmup_logvar if (warmup or noisy) else self.logvar_scale_p
        # encourage objects to be 'on' during warmup
        obj_on_a_prior = torch.tensor(0.2, device=obj_on_a.device) if (warmup or noisy) else self.obj_on_a_p
        obj_on_b_prior = torch.tensor(0.1, device=obj_on_b.device) if (warmup or noisy) else self.obj_on_b_p
        # as the scale is sigmoid-activated, we want the mean to be the inverse of the sigmoid of the glimpse size
        mu_scale_prior = self.mu_scale_prior

        # --- end priors --- #

        # kl-divergence and priors
        mu_prior = mu_p
        logvar_prior = logvar_kp
        mu_post = mu
        logvar_post = torch.zeros_like(logvar)
        loss_kl_kp_base = kl_loss_func(mu_preds=mu_post, logvar_preds=logvar_post, mu_gts=mu_prior,
                                       logvar_gts=logvar_prior)
        loss_kl_kp_base = loss_kl_kp_base.mean()
        loss_kl_kp_offset = calc_kl(logvar_offset.view(-1, logvar_offset.shape[-1]),
                                    mu_offset.view(-1, mu_offset.shape[-1]), logvar_o=logvar_offset_p,
                                    reduce='none')
        loss_kl_kp_offset = (loss_kl_kp_offset.view(-1, self.n_kp_enc)).sum(-1).mean()
        loss_kl_kp = 0.5 * kl_balance * loss_kl_kp_base + loss_kl_kp_offset

        # depth
        loss_kl_depth = calc_kl(logvar_depth.view(-1, logvar_depth.shape[-1]),
                                mu_depth.view(-1, mu_depth.shape[-1]), reduce='none')
        loss_kl_depth = (loss_kl_depth.view(-1, self.n_kp_enc)).sum(-1).mean()

        # defocus
        loss_kl_defocus = calc_kl(logvar_defocus.view(-1, logvar_defocus.shape[-1]),
                                mu_defocus.view(-1, mu_defocus.shape[-1]), logvar_o=warmup_logvar, reduce='none')
        loss_kl_defocus = (loss_kl_defocus.view(-1, self.n_kp_enc)).sum(-1).mean()

        # scale
        # assume sigmoid activation on z_scale
        loss_kl_scale = calc_kl(logvar_scale.view(-1, logvar_scale.shape[-1]),
                                mu_scale.view(-1, mu_scale.shape[-1]), mu_o=mu_scale_prior, logvar_o=logvar_scale_p,
                                reduce='none')

        loss_kl_scale = (loss_kl_scale.view(-1, self.n_kp_enc)).sum(-1).mean()

        # obj_on
        loss_kl_obj_on = calc_kl_beta_dist(obj_on_a, obj_on_b,
                                           obj_on_a_prior,
                                           obj_on_b_prior).sum(-1)
        loss_kl_obj_on = loss_kl_obj_on.mean()

        # features
        loss_kl_feat = calc_kl(logvar_features.view(-1, logvar_features.shape[-1]),
                               mu_features.view(-1, mu_features.shape[-1]), reduce='none')
        loss_kl_feat_obj = loss_kl_feat.view(-1, self.n_kp_enc)
        loss_kl_feat_obj = loss_kl_feat_obj.sum(-1).mean()

        loss_kl_feat_bg = calc_kl(logvar_bg.view(-1, logvar_bg.shape[-1]),
                                  mu_bg.view(-1, mu_bg.shape[-1]), reduce='none')
        loss_kl_feat_bg = loss_kl_feat_bg.mean()
        loss_kl_feat = loss_kl_feat_obj + loss_kl_feat_bg

        # We only apply variational regularization to the scale, depth, and appearance features.
        # The physical coordinates (offsets and defocus) are kept purely deterministic and driven solely by the reconstruction loss.
        # loss_kl = loss_kl_scale + kl_balance * (loss_kl_feat + loss_kl_depth)
        loss_kl = loss_kl_feat_bg

        loss = beta_rec * loss_rec + beta_kl * loss_kl

        if recon_loss_type == "vgg":
            loss = 1e-3 * loss
        loss_dict = {'loss': loss, 'psnr': psnr.detach(), 'kl': loss_kl, 'loss_rec': loss_rec,
                     'loss_kl_kp': loss_kl_kp, 'loss_kl_feat': loss_kl_feat,'loss_kl_defocus': loss_kl_defocus,
                     'loss_kl_obj_on': loss_kl_obj_on, 'loss_kl_scale': loss_kl_scale, 'loss_kl_depth': loss_kl_depth}
        return loss_dict
