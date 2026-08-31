"""
Generate a synthetic labeled dataset from a trained PILPEL checkpoint.
Writes <OUTPUT_ROOT>/<dataset>_<run>/im0.tiff, ... and labels.pickle for DeepSTORM3D.
"""
import itertools
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.io import imread

from loader import load
from utils.util_func import get_config, normalize_01


DEVICE = 'cuda:0'  # GPU index for this machine; falls back to CPU if unavailable
RUN_NAME = 'runs/250826_145106_microtubules_3d/'
OUTPUT_ROOT = 'generated'
SEED = 1

PLOT_SANITY_CHECK = True    # overlay the labels on the first 3 examples, then carry on
BATCH_SIZE = 16              # batch size for latent extraction
MAX_SAMPLES = 10000          # stop after this many generated images


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_checkpoint(run_name):
    run_path = Path(run_name)
    checkpoint_dir = run_path if run_path.exists() else Path('runs') / run_name

    config_path = checkpoint_dir / 'hparams.json'
    saves_dir = checkpoint_dir / 'saves'
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    if not saves_dir.exists():
        raise FileNotFoundError(f'Saves directory not found: {saves_dir}')

    pth_files = sorted(saves_dir.glob('*.pth'))
    if not pth_files:
        raise FileNotFoundError(f'No .pth files found in {saves_dir}')

    checkpoint_path = next((p for p in pth_files if 'best' in p.name), pth_files[0])
    return checkpoint_dir, checkpoint_path, config_path


def batch_image_paths(dataset_root, batch_size):
    image_paths = sorted(Path(dataset_root).glob('*.tiff'), key=lambda p: int(p.stem[2:]))
    if not image_paths:
        raise ValueError(f'No TIFF images found in dataset root: {dataset_root}')

    num_batches = len(image_paths) // batch_size
    return [image_paths[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]


def load_image_batch(image_paths, image_size, device):
    img_batch = torch.zeros((len(image_paths), 1, image_size, image_size), device=device)
    for i, image_path in enumerate(image_paths):
        im = imread(image_path).astype('float32')
        im = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).to(device)
        img_batch[i] = normalize_01(transforms.CenterCrop(image_size)(im))
    return img_batch


def encode_batch(model, img_batch, norm01):
    with torch.no_grad():
        model_dict = model(img_batch, norm01=norm01, deterministic=True)

    return {
        'z': model_dict['z'],
        'z_depth': model_dict['z_depth'],
        'z_defocus': model_dict['z_defocus'],
        'z_scale': model_dict['z_scale'],
        'obj_on': model_dict['obj_on'],
        'z_features': model_dict['z_features'],
        'z_bg': model_dict['z_bg'],
        'psf_on': model_dict['psf_on'],
        'z_read_noise': model_dict['z_read_noise'] if model.read_noise else [],
    }


def perturb_latents(model, latents, psf_threshold=0.1):
    z = latents['z']
    z_psf = torch.clamp(z + torch.randn_like(z) / 80, -1, 1)

    if model.fg_module.psf_model == '2d':
        z_defocus = torch.zeros_like(latents['z_defocus'])
    else:
        z_min = model.optics_dict['z_range'][0]
        z_max = model.optics_dict['z_range'][1] - 0.001
        z_defocus = (z_max - z_min) * torch.rand_like(latents['z_defocus']) + z_min

    psf_on = torch.nn.Threshold(psf_threshold, 0.0)(latents['psf_on'])
    if not (psf_on > 0).any():
        return None

    return {'z_psf': z_psf, 'z_defocus': z_defocus, 'psf_on': psf_on}


def decode_batch(model, latents, perturbed, norm01):
    with torch.no_grad():
        decoder_dict = model.decode_all_generate_image(
            latents['z'],
            perturbed['z_psf'],
            latents['z_features'],
            latents['z_bg'],
            latents['obj_on'],
            perturbed['z_defocus'],
            perturbed['psf_on'],
            latents['z_read_noise'],
            z_depth=latents['z_depth'],
            z_scale=latents['z_scale'],
            blur_bg=False,
            norm01=norm01,
        )

    return decoder_dict


def psf_coordinates(model, perturbed, psf_on):
    half_fov_um = model.image_size // 2 * model.optics_dict['pixel_size_CCD'] / model.optics_dict['M']
    z_psf = perturbed['z_psf']
    z_defocus = perturbed['z_defocus']

    is_2d = model.fg_module.psf_model == '2d'

    xyz = []
    for b, row in enumerate(psf_on):
        active = torch.nonzero(row, as_tuple=True)[0]
        xx = z_psf[b, active, 1].cpu().numpy() * half_fov_um
        yy = z_psf[b, active, 0].cpu().numpy() * half_fov_um
        if is_2d:
            xyz.append(np.stack([xx, yy], axis=1))
        else:
            zz = z_defocus[b, active, 0].cpu().numpy()
            xyz.append(np.stack([xx, yy, zz], axis=1))

    return xyz


def plot_sanity_check(model, xyz_localized, decoder_dict, idx=0):
    um_per_pixel = model.optics_dict['pixel_size_CCD'] / model.optics_dict['M']
    half = model.image_size // 2
    xx = xyz_localized[:, 0] / um_per_pixel + half
    yy = xyz_localized[:, 1] / um_per_pixel + half

    for key in ('rec', 'PSFs'):
        plt.figure()
        plt.imshow(decoder_dict[key][idx].squeeze().detach().cpu().numpy())
        plt.scatter(xx, yy, color='r')
        plt.title(f'{key} [{idx}]')
        plt.show()


def generate_samples(model, image_path_batches, device, norm01, plot_sanity=False):
    for image_path_batch in itertools.cycle(image_path_batches):
        img_batch = load_image_batch(image_path_batch, model.image_size, device)
        latents = encode_batch(model, img_batch, norm01)

        perturbed = perturb_latents(model, latents)
        if perturbed is None:
            continue

        decoder_dict = decode_batch(model, latents, perturbed, norm01)
        rec = decoder_dict['rec']
        xyz = psf_coordinates(model, perturbed, decoder_dict['psf_on'])

        if plot_sanity:
            plotted = 0
            for i, xyz_localized in enumerate(xyz):
                if len(xyz_localized) == 0:
                    continue
                plot_sanity_check(model, xyz_localized, decoder_dict, i)
                plotted += 1
                if plotted == 3:
                    break
            plot_sanity = False

        for idx, xyz_localized in enumerate(xyz):
            if len(xyz_localized) == 0:
                continue
            yield rec[idx].squeeze().cpu().numpy(), xyz_localized


def main():
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

    checkpoint_dir, checkpoint_path, config_path = resolve_checkpoint(RUN_NAME)
    print(f'Using checkpoint: {checkpoint_path}')

    model = load(str(checkpoint_path), device, str(config_path))
    ds_config = get_config(str(config_path))

    dataset_root = ds_config.get('root')
    if dataset_root is None:
        raise ValueError('No dataset root found in checkpoint config.')
    dataset_root = str(dataset_root)
    dataset_name = Path(dataset_root.rstrip('/')).name

    norm01 = ds_config.get('norm01_flag', False)
    image_path_batches = batch_image_paths(dataset_root, BATCH_SIZE)

    out_dir = Path(OUTPUT_ROOT) / f'{dataset_name}_{checkpoint_dir.name}'
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = generate_samples(model, image_path_batches, device, norm01, PLOT_SANITY_CHECK)

    labels_dict = {}
    for label_index, (image, xyz_localized) in enumerate(itertools.islice(samples, MAX_SAMPLES)):
        Image.fromarray(image).save(out_dir / f'im{label_index}.tiff')
        labels_dict[str(label_index)] = np.expand_dims(xyz_localized, 0)
        print(f'Example [{label_index}]')

    with open(out_dir / 'labels.pickle', 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=4)


if __name__ == '__main__':
    main()