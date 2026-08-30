"""Prepare a PILPEL training dataset: save crops of raw acquisition frames.

Writes <output_folder>/im0000.tiff, im0001.tiff, ...

You can randomly sample the FOV, or define a fixed center. Either way we suggest
going over the dataset afterwards to make sure the crops actually contain
emitters, and to clean out the ones that are empty.
"""
import os
import glob
import random
import numpy as np
from PIL import Image
from skimage.io import imread
from utils.util_func import to_uint16


random.seed(0)
np.random.seed(0)

crop_size = 129
num_total_crops = 5000
input_folder = 'raw/my_acquisition/'
output_folder = 'data/my_dataset'

# Where in the raw frame to take crops from.
# Crop centers are sampled uniformly inside these bounds, (low, high) in pixels.
# Keep them at least crop_size // 2 from the frame edge.
center_y_range = (225, 395)
center_x_range = (225, 395)

# ...or use the same window in every frame: set fixed_center to a (y, x) pair,
# e.g. (135, 260). Leave it None to sample randomly from the ranges above.
fixed_center = None

os.makedirs(output_folder, exist_ok=True)

image_paths = glob.glob(os.path.join(input_folder, '*.tif'))

crop_idx = 0
while crop_idx < num_total_crops:
    image = imread(random.choice(image_paths))

    if fixed_center is not None:
        center_y, center_x = fixed_center
    else:
        center_y = np.random.randint(*center_y_range)
        center_x = np.random.randint(*center_x_range)

    y = center_y - crop_size // 2
    x = center_x - crop_size // 2

    crop = image[y:y + crop_size, x:x + crop_size]

    img1 = Image.fromarray(to_uint16(crop))
    img1.save(os.path.join(output_folder, f'im{crop_idx:04d}.tiff'))

    crop_idx += 1
