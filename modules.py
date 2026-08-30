"""
Basic modules and layers.
"""

# imports
import numpy as np
# torch
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.util_func import spatial_transform, generate_correlation_maps
from physics import PhysicalLayer_mask_rec, Normalize01


# ResBlock from: https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding='zeros'):
    """3x3 convolution with padding"""
    if padding == 'zeros':
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)
    else:
        return nn.Sequential(nn.ReplicationPad2d(1),
                             nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride,
                                       padding=0, groups=groups, bias=False, dilation=dilation))


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, padding='replicate', norm_type='gn'):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d if norm_type == 'bn' else nn.GroupNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        n_groups = 4 if (planes % 4 == 0) else 5
        self.conv1 = conv3x3(inplanes, planes, stride, padding=padding)
        self.bn1 = norm_layer(planes) if norm_type == 'bn' else norm_layer(n_groups, planes, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=padding)
        self.bn2 = norm_layer(planes) if norm_type == 'bn' else norm_layer(n_groups, planes, eps=1e-4)
        if downsample is not None:
            self.downsample = downsample
        elif stride > 1 or inplanes != planes:
            self.downsample = conv1x1(inplanes, planes, stride)
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, pad=0, pool=False, upsample=False, bias=False,
                 activation=True, batchnorm=True, relu_type='relu', pad_mode='replicate', use_resblock=False):
        super(ConvBlock, self).__init__()
        self.main = nn.Sequential()
        if use_resblock:
            self.main.add_module(f'conv_{c_in}_to_{c_out}',
                                 BasicBlock(c_in, c_out, stride=stride, padding=pad_mode))
        else:
            if pad_mode != 'zeros':
                self.main.add_module('replicate_pad', nn.ReplicationPad2d(pad))
                pad = 0
            self.main.add_module(f'conv_{c_in}_to_{c_out}', nn.Conv2d(c_in, c_out, kernel_size,
                                                                      stride=stride, padding=pad, bias=bias))
        if batchnorm and not use_resblock:
            n_groups = 4 if (c_out % 4 == 0) else 5
            self.main.add_module(f'grouupnorm_{c_out}', nn.GroupNorm(n_groups, c_out, eps=1e-4))
        if activation and not use_resblock:
            if relu_type == 'leaky':
                self.main.add_module('relu', nn.LeakyReLU(0.01))
            else:
                self.main.add_module('relu', nn.ReLU())
        if pool:
            self.main.add_module('max_pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        if upsample:
            self.main.add_module('upsample_bilinear_2',
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

    def forward(self, x):
        y = self.main(x)
        return y


class KeyPointCNNOriginal(nn.Module):

    def __init__(self, cdim=3, channels=(32, 64, 128, 256), image_size=64, n_kp=8, pad_mode='replicate',
                 use_resblock=False, first_conv_kernel_size=7):
        super(KeyPointCNNOriginal, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.n_kp = n_kp
        cc = channels[0]
        ch = cc
        first_conv_pad = first_conv_kernel_size // 2
        self.main = nn.Sequential()
        self.main.add_module('in_block_1',
                             ConvBlock(cdim, cc, kernel_size=first_conv_kernel_size, stride=1,
                                       pad=first_conv_pad, pool=False, pad_mode=pad_mode,
                                       use_resblock=False, relu_type='relu'))
        self.main.add_module('in_block_2',
                             ConvBlock(cc, cc, kernel_size=3, stride=1, pad=1, pool=False, pad_mode=pad_mode,
                                       use_resblock=use_resblock, relu_type='relu'))

        sz = image_size
        for ch in channels[1:]:
            self.main.add_module('conv_in_{}_0'.format(sz), ConvBlock(cc, ch, kernel_size=3, stride=2, pad=1,
                                                                      pool=False, pad_mode=pad_mode,
                                                                      use_resblock=use_resblock, relu_type='relu'))
            self.main.add_module('conv_in_{}_1'.format(ch), ConvBlock(ch, ch, kernel_size=3, stride=1, pad=1,
                                                                      pool=False, pad_mode=pad_mode,
                                                                      use_resblock=use_resblock, relu_type='relu'))
            cc, sz = ch, sz // 2

        self.keymap = nn.Conv2d(channels[-1], n_kp, kernel_size=1)
        self.conv_output_size = self.calc_conv_output_size()

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x):
        y = self.main(x)
        # heatmap
        hm = self.keymap(y)
        return y, hm


def normalized_cross_correlation(patches, psfs):
    # Normalize patches
    patches = (patches - patches.mean(dim=(-2, -1), keepdim=True)) / (patches.std(dim=(-2, -1), keepdim=True) + 1e-6)

    # Normalize PSFs
    psfs = (psfs - psfs.mean(dim=(-2, -1), keepdim=True)) / psfs.std(dim=(-2, -1), keepdim=True)

    # Compute full correlation maps
    corr_maps = F.conv2d(patches, psfs, padding='same')

    return corr_maps


class AlternativeSpatialSoftmaxKP3D(torch.nn.Module):

    def __init__(self, optics_dict, kp_range=(-1, 1), image_size=None):
        super().__init__()
        self.dz = 21
        self.physical_layer = PhysicalLayer_mask_rec(optics_dict, W=image_size)
        self.psf_crop = self.physical_layer.psf_crop
        self.defocus_vect = torch.linspace(optics_dict['z_range'][0], optics_dict['z_range'][1], self.dz).view(-1,1,1)
        self.psf_dict = self.physical_layer(torch.zeros((self.dz,1,2)), self.defocus_vect)
        self.kp_range = kp_range
        self.z_range = optics_dict['z_range']
        self.norm01 = Normalize01()

    def central_crop(self, patch, crop_size):
        _,_,h, w = patch.shape
        start_x = int((w - crop_size) // 2)
        start_y = int((h - crop_size) // 2)

        return patch[:,:,start_y:start_y + crop_size, start_x:start_x + crop_size].contiguous()

    def forward(self, patches, locations=None, probs=False, variance=False):
        device = patches.device
        psfs = self.psf_dict.to(patches.device)
        N, _cdim, height, width = patches.shape
        patches01 = self.norm01(patches)


        corr_maps = normalized_cross_correlation(patches01, psfs)

        clear_zone = 8
        crop_size = int(corr_maps.shape[-1] - clear_zone*2)
        corr_maps = self.central_crop(corr_maps, crop_size)


        maxcorr, flat_indices = corr_maps.view(N, -1).max(dim=1)

        H_prime, W_prime = corr_maps.shape[2], corr_maps.shape[3]  # Correlation map dimensions

        template_indices = flat_indices // (H_prime * W_prime)
        y_indices = torch.clamp((flat_indices % (H_prime * W_prime)) // W_prime + clear_zone, 0, height-1)
        x_indices = torch.clamp(flat_indices % W_prime + clear_zone, 0, width-1)
        z_indices = template_indices

        z_axis = torch.linspace(self.z_range[0], self.z_range[1], self.dz, device=device)
        y_axis = torch.linspace(self.kp_range[0], self.kp_range[1], height, device=device)
        x_axis = torch.linspace(self.kp_range[0], self.kp_range[1], width, device=device)

        kp_d = z_axis[z_indices]
        kp_h = y_axis[y_indices]
        kp_w = x_axis[x_indices]


        best_psfs = psfs[template_indices]

        # stack keypoints
        kp = torch.stack([kp_h, kp_w], dim=-1)  # [N, 2], xy of each kp; z is returned separately as kp_d

        return kp, kp, kp_d, maxcorr, best_psfs


class CNNDecoder(nn.Module):
    def __init__(self, cdim=3, channels=(64, 128, 256, 512, 512, 512), image_size=64, in_ch=16, n_kp=8,
                 pad_mode='zeros', use_resblock=False):
        super(CNNDecoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        cc = channels[-1]
        self.in_ch = in_ch
        self.n_kp = n_kp

        sz = 4

        self.main = nn.Sequential()
        self.main.add_module('depth_up',
                             ConvBlock(self.in_ch, cc, kernel_size=3, pad=1, upsample=True, pad_mode=pad_mode,
                                       use_resblock=use_resblock, batchnorm=False))
        for ch in reversed(channels[1:-1]):
            self.main.add_module('conv_to_{}'.format(sz * 2), ConvBlock(cc, ch, kernel_size=3, pad=1, upsample=True,
                                                                        pad_mode=pad_mode, use_resblock=use_resblock))
            cc, sz = ch, sz * 2

        self.main.add_module('conv_to_{}'.format(sz * 2),
                             ConvBlock(cc, channels[0], kernel_size=3, pad=1,
                                       upsample=False,
                                       pad_mode=pad_mode, use_resblock=False))
        self.final_conv = ConvBlock(channels[0], cdim, kernel_size=1, bias=True,
                                    activation=False, batchnorm=False, use_resblock=False, pad_mode=pad_mode, pad=1)

    def forward(self, z, masks=None):
        y = self.main(z)
        if masks is not None:
            # masks: [bs, n_kp, feat_dim, feat_dim]
            bs, n_kp, fs, _ = masks.shape
            # y: [bs, n_kp * ch[0], feat_dim, feat_dim]
            y = y.view(bs, n_kp, -1, fs, fs)
            y = masks.unsqueeze(2) * y
            y = y.view(bs, -1, fs, fs)
        y = self.final_conv(y)
        y = torch.sigmoid(y)
        return y


class ImagePatcher1(nn.Module):
    def __init__(self,image_size: int, patch_size: int, stride: int):

        super(ImagePatcher1, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, x):

        B, cdim, H, W = x.shape

        # Unfold the image into patches
        patches = F.unfold(
            x,
            kernel_size=self.patch_size,
            stride=self.stride
        )  # Shape: (B, cdim * patch_size * patch_size, num_patches)

        # Reshape to split patch dimensions
        patches = patches.view(
            B, cdim, self.patch_size, self.patch_size, -1
        ).permute(0, 1, 4, 2, 3)  # Shape: (B, cdim, num_patches, patch_size, patch_size)

        # Compute global locations of the patches
        num_patches_per_row = (W - self.patch_size) // self.stride + 1
        num_patches_per_col = (H - self.patch_size) // self.stride + 1

        locations = []
        for row in range(num_patches_per_col):
            for col in range(num_patches_per_row):
                locations.append((row * self.stride, col * self.stride))

        locations = torch.tensor(locations, dtype=torch.long, device=x.device)  # Shape: (num_patches, 2)

        return patches, locations

    def get_patch_location_idx(self):

        num_patches_per_row = (self.image_size - self.patch_size) // self.stride + 1
        num_patches_per_col = (self.image_size - self.patch_size) // self.stride + 1

        locations = []
        for row in range(num_patches_per_col):
            for col in range(num_patches_per_row):
                locations.append((row * self.stride, col * self.stride))

        return torch.tensor(locations, dtype=torch.float32)


class VariationalKeyPointPatchEncoderPSF(nn.Module):

    def __init__(self, optics_dict, cdim=3, channels=(16, 16, 32), image_size=64, n_kp=4, patch_size=16, kp_range=(0, 1),
                 use_logsoftmax=False, pad_mode='replicate', sigma=0.1, dropout=0.0, learnable_logvar=False,
                 learned_feature_dim=0, use_resblock=False):
        super(VariationalKeyPointPatchEncoderPSF, self).__init__()
        self.use_logsoftmax = use_logsoftmax
        self.image_size = image_size
        self.dropout = dropout
        self.kp_range = kp_range
        self.use_resblock = use_resblock
        self.n_kp = n_kp  # kp per patch
        self.patcher = ImagePatcher1(image_size=image_size, patch_size=patch_size, stride=int(0.5*patch_size))

        self.features_dim = int(patch_size // (2 ** (len(channels) - 1)))
        self.enc = KeyPointCNNOriginal(cdim=cdim, channels=channels, image_size=patch_size, n_kp=n_kp,
                                       pad_mode=pad_mode, use_resblock=self.use_resblock, first_conv_kernel_size=3)
        self.ssm3d = AlternativeSpatialSoftmaxKP3D(optics_dict, kp_range=kp_range, image_size=self.image_size)
        self.sigma = sigma
        self.learnable_logvar = learnable_logvar
        self.learned_feature_dim = learned_feature_dim

        if self.learnable_logvar:
            self.to_logvar = nn.Sequential(nn.Linear(self.n_kp * (self.features_dim ** 2), 512),
                                           nn.ReLU(True),
                                           nn.Linear(512, 256),
                                           nn.ReLU(True),
                                           nn.Linear(256, self.n_kp * 2))  # logvar_x, logvar_y
        if self.learned_feature_dim > 0:
            self.to_features = nn.Sequential(nn.Linear(self.n_kp * (self.features_dim ** 2), 512),
                                             nn.ReLU(True),
                                             nn.Linear(512, 256),
                                             nn.ReLU(True),
                                             nn.Linear(256, self.n_kp * self.learned_feature_dim))  # logvar_x, logvar_y


    def get_global_kp(self, local_kp):
        # local_kp: [batch_size, num_patches, n_kp, 2]
        # returns the global coordinates of a KP within the original image.
        batch_size, num_patches, n_kp, _ = local_kp.shape
        global_coor = self.patcher.get_patch_location_idx().to(local_kp.device)  # [num_patches, 2]
        global_coor = global_coor[:, None, :].repeat(1, n_kp, 1)
        global_coor = (((local_kp - self.kp_range[0]) / (self.kp_range[1] - self.kp_range[0])) * (
                self.patcher.patch_size - 1) + global_coor) / (self.image_size - 1)
        global_coor = global_coor * (self.kp_range[1] - self.kp_range[0]) + self.kp_range[0]
        return global_coor

    def get_distance_from_patch_centers(self, kp, global_kp=False):
        # calculates the distance of a KP from the center of its parent patch. This is useful to understand (and filter)
        # if SSM detected something, otherwise, the KP will probably land in the center of the patch
        # (e.g., a solid-color patch will have the same activation in all pixels).
        if not global_kp:
            global_coor = self.get_global_kp(kp).view(kp.shape[0], -1, 2)
        else:
            global_coor = kp
        centers = 0.5 * (self.kp_range[1] + self.kp_range[0]) * torch.ones_like(kp).to(kp.device)
        global_centers = self.get_global_kp(centers.view(kp.shape[0], -1, self.n_kp, 2)).view(kp.shape[0], -1, 2)
        return ((global_coor - global_centers) ** 2).sum(-1)


    def encode(self, x, global_kp=False):
        # x: [batch_size, cdim, image_size, image_size]
        # global_kp: set True to get the global coordinates within the image (instead of local KP inside the patch)
        batch_size, cdim, image_size, image_size = x.shape
        x_patches = self.patcher(x)[0]  # [batch_size, cdim, num_patches, patch_size, patch_size]
        locations = self.patcher(x)[1]
        x_patches = x_patches.permute(0, 2, 1, 3, 4)  # [batch_size, num_patches, cdim, patch_size, patch_size]
        x_patches = x_patches.contiguous().view(-1, cdim, self.patcher.patch_size, self.patcher.patch_size)
        z = x_patches
        mu_kp, var_kp, defocus_p, max_corr_p, psfs = self.ssm3d(z, locations, probs=False, variance=True)  # [batch_size * num_patches, n_kp, 2]

        mu_kp = mu_kp.view(batch_size, -1, self.n_kp, 2)  # [batch_size, num_patches, n_kp, 2]
        var_kp = var_kp.view(batch_size, mu_kp.shape[1], self.n_kp, -1)  # [batch_size, num_patches, n_kp, 3]
        max_corr_p = max_corr_p.view(batch_size, mu_kp.shape[1], self.n_kp, -1)  # [batch_size, num_patches, n_kp, 1]
        defocus_p = defocus_p.view(batch_size, mu_kp.shape[1], self.n_kp, -1)  # [batch_size, num_patches, n_kp, 1]
        psfs = psfs.view(batch_size, mu_kp.shape[1], self.n_kp, self.ssm3d.psf_crop, self.ssm3d.psf_crop)  # [batch_size, num_patches, n_kp, 1]

        if global_kp:
            mu_kp_global = self.get_global_kp(mu_kp)

            mu_kp_out = mu_kp_global
            _, n_kp, _, _ = mu_kp_global.shape
            updated_max_corr = max_corr_p.squeeze()


        if self.learned_feature_dim > 0:
            mu_features = self.to_features(z.view(z.shape[0], -1))
            mu_features = mu_features.view(batch_size, -1, self.n_kp, self.learned_feature_dim)
            # [batch_size, num_patches, n_kp, learned_feature_dim]
        if self.learnable_logvar:
            logvar_kp = self.to_logvar(z.view(z.shape[0], -1))
            logvar_kp = logvar_kp.view(batch_size, -1, self.n_kp, 2)  # [batch_size, num_patches, n_kp, 2]
            if self.learned_feature_dim > 0:
                return mu_kp_out, logvar_kp, mu_features, defocus_p, updated_max_corr, psfs
            else:
                return mu_kp_out, logvar_kp, defocus_p, updated_max_corr, psfs
        elif self.learned_feature_dim > 0:
            return mu_kp_out, mu_features, defocus_p, updated_max_corr, psfs
        else:
            return mu_kp_out, var_kp, defocus_p, updated_max_corr, psfs

    def forward(self, x, global_kp=False):
        return self.encode(x, global_kp)


class ParticleAttributeEncoder(nn.Module):

    def __init__(self, anchor_size, image_size, cnn_channels=(16, 16, 32), margin=0, ch=3, max_offset=1.0,
                 kp_activation='tanh', use_resblock=False, use_correlation_heatmaps=False,
                 hidden_dims=(256, 256)):
        super().__init__()
        # use_correlation_heatmaps: use correlation heatmaps as input to model particle properties (e.g., xy offset)
        self.anchor_size = anchor_size
        self.channels = cnn_channels
        self.image_size = image_size
        if anchor_size<1:
            self.patch_size = np.round(anchor_size * (image_size - 1)).astype(int)
        else:
            self.patch_size = image_size
        self.margin = margin
        self.crop_size = self.patch_size + 2 * margin
        self.ch = ch
        self.use_resblock = use_resblock
        self.use_correlation_heatmaps = use_correlation_heatmaps
        self.kp_activation = kp_activation
        self.max_offset = max_offset  # max offset of x-y, [-max_offset, +max_offset]
        self.hidden_dims = hidden_dims
        hidden_dim_1 = hidden_dims[0]
        hidden_dim_2 = hidden_dims[1]

        in_ch = (ch + 1) if self.use_correlation_heatmaps else ch
        self.cnn = KeyPointCNNOriginal(cdim=in_ch, channels=cnn_channels, image_size=self.crop_size, n_kp=32,
                                       pad_mode='replicate', use_resblock=self.use_resblock,
                                       first_conv_kernel_size=3)

        feature_map_size = (self.crop_size // 2 ** (len(cnn_channels) - 1)) ** 2
        fc_in_dim = 32 * feature_map_size
        self.projection = nn.Linear(fc_in_dim, hidden_dim_2)

        self.backbone = nn.Sequential(nn.Linear(hidden_dim_2, hidden_dim_1),
                                      nn.ReLU(True),
                                      nn.Linear(hidden_dim_1, hidden_dim_2),
                                      nn.ReLU(True))

        self.x_head = nn.Linear(hidden_dim_2, 2)  # mu_x, logvar_x
        self.y_head = nn.Linear(hidden_dim_2, 2)  # mu_y, logvar_y
        self.scale_xy_head = nn.Linear(hidden_dim_2, 4)  # mu_sx, logvar_sx, mu_sy, logvar_sy
        self.obj_on_head = nn.Linear(hidden_dim_2, 2)  # [log_obj_on_a, log_obj_on_b]
        self.psf_on_head = nn.Linear(hidden_dim_2, 2)  # [log_psf_on_a, log_psf_on_b]
        self.depth_head = nn.Linear(hidden_dim_2, 2)  # mu_depth, logvar_depth

        self.defocus_offset_head = nn.Linear(hidden_dim_2, 2)  # mu_defocus, logvar_defocus

    def forward(self, x, kp, z_scale=None, previous_objects=None):
        # x: [bs, ch, image_size, image_size]
        # kp: [bs, n_kp, 2] in [-1, 1]
        # previous_objects: [bs * n_kp, ch, patch_size, patch_size] or None, used as correlation templates
        batch_size, _, _, img_size = x.shape
        _, n_kp, _ = kp.shape
        if self.use_correlation_heatmaps:
            cropped_objects = generate_correlation_maps(x, kp, self.patch_size, previous_objects=previous_objects,
                                                        z_scale=z_scale)
            # [batch_size * n_kp, ch + 1, patch_size, patch_size]
        else:
            x_repeated = x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1)  # [batch_size, n_kp, ch, image_size, image_size]
            x_repeated = x_repeated.view(-1, *x.shape[1:])  # [batch_size * n_kp, ch, image_size, image_size]
            if z_scale is None:
                z_scale = (self.patch_size / img_size) * torch.ones_like(kp)
            else:
                # assume unnormalized z_scale
                z_scale = torch.sigmoid(z_scale)
            z_pos = kp.reshape(-1, kp.shape[-1])
            z_scale = z_scale.view(-1, z_scale.shape[-1])
            out_dims = (batch_size * n_kp, x.shape[1], self.patch_size, self.patch_size)
            cropped_objects = spatial_transform(x_repeated, z_pos, z_scale, out_dims, inverse=False)
            # [batch_size * n_kp, ch, patch_size, patch_size]

        # encode objects - fc
        _, cropped_objects_cnn = self.cnn(cropped_objects)
        cropped_objects_flat = cropped_objects_cnn.reshape(batch_size, n_kp, -1)  # flatten
        # backbone features
        backbone_features = cropped_objects_flat
        # projection
        backbone_features = self.projection(backbone_features)


        # backbone features
        backbone_features = self.backbone(backbone_features)
        # projection to output
        stats_x = self.x_head(backbone_features)
        stats_x = stats_x.view(batch_size, n_kp, 2)
        mu_x, logvar_x = stats_x.chunk(chunks=2, dim=-1)

        stats_y = self.y_head(backbone_features)
        stats_y = stats_y.view(batch_size, n_kp, 2)
        mu_y, logvar_y = stats_y.chunk(chunks=2, dim=-1)

        mu = torch.cat([mu_x, mu_y], dim=-1)
        logvar = torch.cat([logvar_x, logvar_y], dim=-1)

        scale_xy = self.scale_xy_head(backbone_features)
        scale_xy = scale_xy.view(batch_size, n_kp, -1)
        mu_scale, logvar_scale = torch.chunk(scale_xy, chunks=2, dim=-1)

        if self.kp_activation == "tanh":
            mu = self.max_offset * torch.tanh(mu)
        elif self.kp_activation == "sigmoid":
            mu = self.max_offset * torch.sigmoid(mu)

        obj_on = self.obj_on_head(backbone_features)
        obj_on = obj_on.view(batch_size, n_kp, 2)
        lobj_on_a, lobj_on_b = torch.chunk(obj_on, chunks=2, dim=-1)  # log alpha, beta of Beta dist
        lobj_on_a = lobj_on_a.squeeze(-1)
        lobj_on_b = lobj_on_b.squeeze(-1)

        psf_on = self.psf_on_head(backbone_features)
        psf_on = psf_on.view(batch_size, n_kp, 2)
        lpsf_on_a, lpsf_on_b = torch.chunk(psf_on, chunks=2, dim=-1)  # log alpha, beta of Beta dist
        lpsf_on_a = lpsf_on_a.squeeze(-1)
        lpsf_on_b = lpsf_on_b.squeeze(-1)

        depth = self.depth_head(backbone_features)
        depth = depth.view(batch_size, n_kp, 2)
        mu_depth, logvar_depth = torch.chunk(depth, 2, dim=-1)


        defocus = self.defocus_offset_head(backbone_features)
        defocus = defocus.view(batch_size, n_kp, 2)
        mu_defocus_offset, logvar_defocus_offset = torch.chunk(defocus, 2, dim=-1)


        spatial_out = {'mu': mu, 'logvar': logvar, 'mu_scale': mu_scale, 'logvar_scale': logvar_scale,
                       'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b, 'obj_on': obj_on,
                       'lpsf_on_a': lpsf_on_a, 'lpsf_on_b': lpsf_on_b, 'psf_on': psf_on,
                       'mu_depth': mu_depth, 'logvar_depth': logvar_depth,
                       'mu_defocus_offset': mu_defocus_offset, 'logvar_defocus_offset': logvar_defocus_offset}

        return spatial_out


class ParticleFeaturesEncoder(nn.Module):


    def __init__(self, anchor_size, features_dim, image_size, cnn_channels=(16, 16, 32), margin=0, ch=3,
                 use_resblock=False, hidden_dims=(256, 256)):
        super().__init__()
        # use_correlation_heatmaps: use correlation heatmaps as input to model particle properties (e.g., xy offset)
        self.anchor_size = anchor_size
        self.channels = cnn_channels
        self.image_size = image_size
        if anchor_size < 1:
            self.patch_size = np.round(anchor_size * (image_size - 1)).astype(int)
        else:
            self.patch_size = image_size
        self.margin = margin
        self.crop_size = self.patch_size + 2 * margin
        self.ch = ch
        self.use_resblock = use_resblock
        self.features_dim = features_dim
        self.hidden_dims = hidden_dims
        hidden_dim_1 = hidden_dims[0]
        hidden_dim_2 = hidden_dims[1]

        in_ch = ch
        self.cnn = KeyPointCNNOriginal(cdim=in_ch, channels=cnn_channels, image_size=self.crop_size, n_kp=32,
                                       pad_mode='replicate', use_resblock=self.use_resblock,
                                       first_conv_kernel_size=3)

        feature_map_size = (self.crop_size // 2 ** (len(cnn_channels) - 1)) ** 2
        fc_in_dim = 32 * feature_map_size
        self.projection = nn.Linear(fc_in_dim, hidden_dim_2)

        self.backbone = nn.Sequential(nn.Linear(hidden_dim_2, hidden_dim_1),
                                      nn.ReLU(True),
                                      nn.Linear(hidden_dim_1, hidden_dim_2),
                                      nn.ReLU(True))
        self.features_head = nn.Linear(hidden_dim_2, 2 * self.features_dim)  # mu_features, logvar_features

    def forward(self, x, kp, z_scale=None):
        # x: [bs, ch, image_size, image_size]
        # kp: [bs, n_kp, 2] in [-1, 1]
        batch_size = x.shape[0]
        n_kp = kp.shape[1]
        img_size = x.shape[-1]
        x_repeated = x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1)  # [batch_size, n_kp, ch, image_size, image_size]
        x_repeated = x_repeated.view(-1, *x.shape[1:])  # [batch_size * n_kp, ch, image_size, image_size]
        if z_scale is None:
            z_scale = (self.patch_size / img_size) * torch.ones_like(kp)
        else:
            # assume unnormalized z_scale
            z_scale = torch.sigmoid(z_scale)
        z_pos = kp.reshape(-1, kp.shape[-1])
        z_scale = z_scale.view(-1, z_scale.shape[-1])
        out_dims = (batch_size * n_kp, x.shape[1], self.patch_size, self.patch_size)
        cropped_objects = spatial_transform(x_repeated, z_pos, z_scale, out_dims, inverse=False)
        # [batch_size * n_kp, ch, patch_size, patch_size]

        # encode objects - fc
        _, cropped_objects_cnn = self.cnn(cropped_objects)
        cropped_objects_flat = cropped_objects_cnn.reshape(batch_size, n_kp, -1)  # flatten
        # backbone features
        backbone_features = cropped_objects_flat
        # projection
        backbone_features = self.projection(backbone_features)


        backbone_features = self.backbone(backbone_features)
        # projection to output

        features = self.features_head(backbone_features)
        features = features.view(batch_size, n_kp, -1)
        mu_features, logvar_features = torch.chunk(features, 2, dim=-1)

        cropped_objects = cropped_objects.view(batch_size, -1, *cropped_objects.shape[1:])
        # [batch_size, n_kp, ch, crop_size, crop_size]
        spatial_out = {'mu_features': mu_features, 'logvar_features': logvar_features,
                       'cropped_objects': cropped_objects}
        return spatial_out


class ObjectDecoderCNN(nn.Module):
    def __init__(self, patch_size, num_chans=4, bottleneck_size=128, pad_mode='reflect', embed_position=False,
                 use_resblock=False):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.num_chans = num_chans
        self.embed_position = embed_position
        self.use_resblock = use_resblock
        if self.embed_position:
            self.position_embedding = nn.Linear(2, bottleneck_size)

        self.in_ch = 32

        if self.patch_size[0] == 52:
            sz = 13
        elif self.patch_size[0] == 40:
            sz = 10
        else:
            sz = 8

        self.sz = sz

        fc_out_dim = self.in_ch * sz * sz
        fc_in_dim = bottleneck_size if not self.embed_position else 2 * bottleneck_size

        self.fc = nn.Sequential(nn.Linear(fc_in_dim, 256, bias=True),
                                nn.ReLU(True),
                                nn.Linear(256, 256),
                                nn.ReLU(True),
                                nn.Linear(256, fc_out_dim),
                                nn.ReLU(True))

        num_upsample = int(np.log2(patch_size[0])) - 3
        self.channels = [self.in_ch]
        for i in range(num_upsample):
            self.channels.append(64)
        cc = self.channels[-1]


        self.main = nn.Sequential()
        if num_upsample > 0:
            self.main.add_module('depth_up',
                                 ConvBlock(self.in_ch, cc, kernel_size=3, pad=1, upsample=True, pad_mode=pad_mode,
                                           use_resblock=self.use_resblock))
            for ch in reversed(self.channels[1:-1]):
                self.main.add_module('conv_to_{}'.format(sz * 2), ConvBlock(cc, ch, kernel_size=3, pad=1, upsample=True,
                                                                            pad_mode=pad_mode,
                                                                            use_resblock=self.use_resblock))
                cc, sz = ch, sz * 2

        self.main.add_module('conv_to_{}'.format(sz * 2),
                             ConvBlock(cc, self.channels[0], kernel_size=3, pad=1,
                                       upsample=False, pad_mode=pad_mode, use_resblock=False))
        self.main.add_module('final_conv', ConvBlock(self.channels[0], num_chans, kernel_size=1, bias=True,
                                                     activation=False, batchnorm=False, use_resblock=False,
                                                     pad_mode=pad_mode))
        self.decode = self.main

    def forward(self, x, kp=None):
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        if kp is not None and self.embed_position:
            pos_embed = self.position_embedding(kp)  # [bs, n_kp, bottleneck_size]
            pos_embed = pos_embed.view(-1, pos_embed.shape[-1])
            x = torch.cat([x, pos_embed], dim=-1)
        conv_in = self.fc(x)
        conv_in = conv_in.view(-1, 32, self.sz, self.sz)
        out = self.decode(conv_in).view(-1, self.num_chans, *self.patch_size)
        out = torch.sigmoid(out)
        return out


class FCToCNN(nn.Module):
    def __init__(self, target_hw=16, n_ch=8, pad_mode='replicate', features_dim=2, use_resblock=False):
        super(FCToCNN, self).__init__()
        # features_dim : 2 [logvar] + additional features
        self.features_dim = features_dim  # logvar, features
        self.n_ch = n_ch
        self.fmap_size = int(target_hw/2)
        self.use_resblock = use_resblock
        fc_out_dim = self.n_ch * (self.fmap_size ** 2)

        self.mlp = nn.Sequential(nn.Linear(self.features_dim, 256),
                                 nn.ReLU(True),
                                 nn.Linear(256, 256),
                                 nn.ReLU(True),
                                 nn.Linear(256, fc_out_dim),
                                 nn.ReLU(True))

        num_upsample = int(np.log(target_hw) // np.log(2)) - 3
        self.cnn = nn.Sequential()
        for i in range(num_upsample):
            self.cnn.add_module(f'depth_up_{i}', ConvBlock(n_ch, n_ch, kernel_size=3, pad=1,
                                                           upsample=True, pad_mode=pad_mode,
                                                           use_resblock=self.use_resblock))

    def forward(self, features):
        # features [batch_size, features_dim]
        features = features.view(-1, features.shape[-1])  # [batch_size * n_kp, features]
        h = self.mlp(features)
        h = h.view(-1, self.n_ch, self.fmap_size, self.fmap_size)  # [batch_size, n_kp, 4, 4]
        cnn_out = self.cnn(h)  # [batch_size, n_kp, target_hw, target_hw]
        return cnn_out
