# PILPEL - Physics-Informed Self-Supervised Generative Model for Localization Microscopy

Official implementation of the paper:

**Self-supervised generation of realistic training data enables nanoscale localization in challenging conditions**

Ofri Goldenberg, Tal Daniel, Dafei Xiao, Yael Shalev Ezra, Onit Alalouf, and Yoav Shechtman
[bioRxiv 2026](https://www.biorxiv.org/content/10.1101/2025.07.16.665148v3)

## Overview

PILPEL is a physics-informed self-supervised generative model for 2D and 3D single-molecule localization microscopy. It extends the Deep Latent Particles (DLP) framework by incorporating a physical model of the Point Spread Function (PSF) into the decoder, enabling it to disentangle learned realistic environments from individual emitters.

Trained directly on unlabeled experimental images, PILPEL generates fully labeled, realistic synthetic training datasets with known emitter locations. These datasets substantially improve the performance of supervised localization networks, particularly in challenging scenarios such as complex backgrounds and low signal-to-noise ratios.

## Installation

```bash
conda env create -f environment.yml
conda activate pilpel
```

Alternatively, use `pip` to install `requirements.txt`.

## Setup

### Phase masks
Copy your `.mat` phase mask files into the `phase_masks/` folder. Phase mask paths are referenced in experiment config files.

### 2D and 3D
`optics_dict` determines the physical model of the PSF. The default, `"psf_model": "3d"`, renders each emitter through
the phase mask at its estimated depth and needs `phase_mask_root`, `NFP` and `z_range`. Setting
`"psf_model": "2d"` renders a mask-free in-focus PSF instead, and every
emitter is localized in x and y only. `configs/microtubules_2d.json` is a complete 2D example.

### Data
Copy your TIFF image datasets into `data/`. The expected structure is:

```
data/
└── <dataset_name>/
    ├── im0001.tiff
    └── ...
```

This repository contains two 250-image samples of experimental microtubules data, so that
`train_pilpel.py` runs immediately after cloning: `data/microtubules_3d/` for the 3D configuration and
`data/microtubules_2d/` for the 2D one. To train on your own data, add it under `data/` in the same layout
and point a config at it.

We provide the script `data_prep.py` to save crops of your raw acquisition into this layout. 

### Dataset configs
Each dataset should have a self-contained JSON file in `configs/` holding every training hyperparameter along with
data root and optics. See `configs/microtubules_3d.json` for a 3D example and
`configs/microtubules_2d.json` for a 2D one. To add a dataset, copy an existing config and edit `ds`,
`root` and `optics_dict`.


## Usage

### 1. Train PILPEL

Open `train_pilpel.py`, point `CONFIG` at your dataset config and set `DEVICE` for this machine:

```python
CONFIG = 'configs/microtubules_3d.json'
DEVICE = 'cuda:0'
```

Optionally - use the `config.update({...})` block in `build_config()` for run-specific configuration update.

The model checkpoint, config, run log and per-epoch figures are saved to `runs/<DDMMYY_HHMMSS>_<ds>/`.

### 2. Generate a labeled dataset

Open `generate_dataset.py`, point `RUN_NAME` at a trained run, set `DEVICE` for your machine, and choose number of training samples wanted.

This produces synthetic TIFF images and a pickle file of ground-truth localizations, which can be used to train a supervised localization network such as [DeepSTORM3D](https://github.com/EliasNehme/DeepSTORM3D). The labels are `x, y, z` under a 3D config and `x, y` under a 2D one.

## Acknowledgements

This work builds on the following open-source projects:

- **DLPv2** — the self-supervised object-centric generative model that serves as the base framework for PILPEL.
  [https://github.com/taldatech/ddlp](https://github.com/taldatech/ddlp)

- **DeepSTORM3D** — the supervised CNN-based 3D localization network used for evaluation, and the source of the physical layer and simulation utilities.
  [https://github.com/EliasNehme/DeepSTORM3D](https://github.com/EliasNehme/DeepSTORM3D)

- **VIPR** — phase retrieval method used to estimate experimental PSFs from calibration bead measurements.
  [https://github.com/Borisfer/VIPR-Vectorial-Phase-Retrieval-for-microscopy](https://github.com/Borisfer/VIPR-Vectorial-Phase-Retrieval-for-microscopy)

## Citation

If you use this code, please cite:

```bibtex
@article{goldenberg2025pilpel,
  title={Physics-Informed Self-Supervised Generative Model for 3D Localization Microscopy},
  author={Goldenberg, Ofri and Daniel, Tal and Xiao, Dafei and Shalev Ezra, Yael and Shechtman, Yoav},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.07.16.665148}
}
```
