import glob
import os

import torchvision.transforms as transforms
from skimage.io import imread
from torch.utils.data import Dataset

from utils.util_func import normalize_01


class ExpDataset(Dataset):
    def __init__(self, root, mode, image_size=128, transform=None, val_fraction=0.1):
        super().__init__()
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'valid':
            mode = 'val'
        self.mode = mode

        image_paths = sorted(glob.glob(os.path.join(root, '*.tiff')))
        if not image_paths:
            raise FileNotFoundError(f'No TIFF images found in: {root}')

        num_images = len(image_paths)
        val_size = max(1, int(round(val_fraction * num_images))) if num_images > 1 else 0
        test_size = val_size if num_images > 2 else 0
        train_end = max(0, num_images - val_size - test_size)
        val_end = train_end + val_size

        if mode == 'train':
            self.im_list = image_paths[:train_end] if train_end > 0 else image_paths
        elif mode == 'val':
            self.im_list = image_paths[train_end:val_end] if val_size > 0 else image_paths[:1]
        else:
            self.im_list = image_paths[val_end:] if test_size > 0 else image_paths[-1:]

        print(
            f'loaded data with shape: {num_images}, '
            f'train: {len(image_paths[:train_end])}, '
            f'val: {len(image_paths[train_end:val_end])}, '
            f'test: {len(image_paths[val_end:])}, '
            f'active split ({mode}): {len(self.im_list)}'
        )

        self.image_size = image_size
        if transform is None:
            self.input_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
        else:
            self.input_transform = transform

    def __getitem__(self, index):
        image = imread(self.im_list[index]).astype('float32')
        image = self.input_transform(image)
        image01 = normalize_01(image)
        return image01, index

    def __len__(self):
        return len(self.im_list)
