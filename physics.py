import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from math import pi
from torchvision import transforms


class Normalize01(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, result_noisy):

        # number of samples in each batch
        Nbatch = result_noisy.size(0)
        Nparticles = result_noisy.size(1)

        # [0,1] normalization to prevent scaling vulnerability
        result_noisy_01 = torch.zeros_like(result_noisy)
        for i in range(Nbatch):
            for j in range(Nparticles):
                min_val = (result_noisy[i, j, :, :]).min()
                max_val = (result_noisy[i, j, :, :]).max()
                result_noisy_01[i, j, :, :] = (result_noisy[i, j, :, :] - min_val) / (max_val - min_val + 1e-6)

        return result_noisy_01


class Croplayer(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.H, self.W = W, W

    def forward(self, im):
        # size of the PSFs array
        Nbatch, C, Him, Wim = im.size()

        # crop the central portion in the xy plane to remove redundant zeros
        Xlow = int(np.floor(Wim / 2) - np.floor(self.W / 2))
        Xhigh = int(np.floor(Wim / 2) + np.floor(self.W / 2) + 1)
        Ylow = int(np.floor(Him / 2) - np.floor(self.H / 2))
        Yhigh = int(np.floor(Him / 2) + np.floor(self.H / 2) + 1)
        images4D_crop = im[:, :, Ylow:Yhigh, Xlow:Xhigh].contiguous()

        return images4D_crop


def calc_bfp_grids(phase_mask, optics_dict):
    if 'Hmask' in optics_dict:
        N = optics_dict['Hmask']
    else:
        N = phase_mask.shape[0]

    NFP = optics_dict['NFP']
    lamda = optics_dict['lamda']
    NA, M, f_4f = optics_dict['NA'], optics_dict['M'], optics_dict['f_4f']
    pixel_size_CCD, pixel_size_SLM = optics_dict['pixel_size_CCD'], optics_dict['pixel_size_SLM']
    noil, nwater = optics_dict['noil'], optics_dict['nwater']

    # SLM, and back focal plane properties
    BFPdiam_um = 2 * f_4f * NA / np.sqrt(M ** 2 - NA ** 2)  # [um]
    px_per_um = 1 / pixel_size_SLM  # [1/um]
    BFPdiam_px = BFPdiam_um * px_per_um  # BFP diameter in pixels
    CCD_size_px = lamda * f_4f * px_per_um / pixel_size_CCD  # final size of FFT on camera
    pad_px = int(np.round(0.5 * (CCD_size_px - N)))

    # cartesian and polar grid in BFP
    Xphys = np.linspace(-(N - 1) / 2, (N - 1) / 2, N) / px_per_um
    XI, ETA = np.meshgrid(Xphys, Xphys)  # cartesian physical coords. in SLM space [um]
    Xang = np.linspace(-1, 1, N) * NA / noil * N / BFPdiam_px  # each pixel is NA/(BFPdiam_px/2*n1) wide
    XX, YY = np.meshgrid(Xang, Xang)
    r = np.sqrt(XX ** 2 + YY ** 2)  # radial coordinate s.t. r = NA/noil at edge of E field support

    # defocus aberration in oil
    koil = 2 * pi * noil / lamda  # [1/um]
    sin_theta_oil = r  # um
    circ_oil = sin_theta_oil < NA / noil
    circ_oil = circ_oil.astype("float32")
    cos_theta_oil = np.emath.sqrt(1 - (sin_theta_oil * circ_oil) ** 2) * circ_oil
    defocusAberr = np.exp(1j * koil * (-NFP) * cos_theta_oil)

    # create the circular aperture
    kwater = 2 * pi * nwater / lamda
    sin_theta_water = noil / nwater * sin_theta_oil
    circ_water = sin_theta_water < 1
    circ_water = circ_water.astype("float32")
    sin_theta_water = sin_theta_water * circ_water
    cos_theta_water = np.sqrt(1 - sin_theta_water ** 2) * circ_water

    # circular aperture to impose on the mask in the SLM
    circ = circ_water.astype("float32")

    # Inputs for the calculation of lateral and axial phases
    Xgrid = 2 * pi * XI * M / (lamda * f_4f)
    Ygrid = 2 * pi * ETA * M / (lamda * f_4f)
    Zgrid = kwater * cos_theta_water

    # save grids to bfp_grids dictionary
    bfp_grids = {}
    bfp_grids['circ'] = torch.from_numpy(circ)
    bfp_grids['Xgrid'], bfp_grids['Ygrid'], bfp_grids['Zgrid'] = torch.from_numpy(Xgrid), torch.from_numpy(
        Ygrid), torch.from_numpy(Zgrid)
    bfp_grids['defocusAberr'] = torch.from_numpy(defocusAberr)
    bfp_grids['pad_px'] = pad_px

    return bfp_grids


def calc_bfp_grids2D(optics_dict):

    lamda = optics_dict['lamda']
    NA, M, f_4f = optics_dict['NA'], optics_dict['M'], optics_dict['f_4f']
    pixel_size_CCD, pixel_size_SLM = optics_dict['pixel_size_CCD'], optics_dict['pixel_size_SLM']
    noil, nwater = optics_dict['noil'], optics_dict['nwater']

    # SLM, and back focal plane properties
    BFPdiam_um = 2 * f_4f * NA / np.sqrt(M ** 2 - NA ** 2)  # [um]
    px_per_um = 1 / pixel_size_SLM  # [1/um]
    BFPdiam_px = BFPdiam_um * px_per_um  # BFP diameter in pixels
    CCD_size_px = lamda * f_4f * px_per_um / pixel_size_CCD  # final size of FFT on camera
    N = int(CCD_size_px)

    # cartesian and polar grid in BFP
    Xphys = np.linspace(-(N - 1) / 2, (N - 1) / 2, N) / px_per_um
    XI, ETA = np.meshgrid(Xphys, Xphys)  # cartesian physical coords. in SLM space [um]
    Xang = np.linspace(-1, 1, N) * NA / noil * N / BFPdiam_px  # each pixel is NA/(BFPdiam_px/2*n1) wide
    XX, YY = np.meshgrid(Xang, Xang)
    r = np.sqrt(XX ** 2 + YY ** 2)  # radial coordinate s.t. r = NA/noil at edge of E field support

    # defocus aberration in oil
    koil = 2 * pi * noil / lamda  # [1/um]
    sin_theta_oil = r  # um
    circ_oil = sin_theta_oil < NA / noil
    circ_oil = circ_oil.astype("float32")
    cos_theta_oil = np.emath.sqrt(1 - (sin_theta_oil * circ_oil) ** 2) * circ_oil

    # create the circular aperture
    kwater = 2 * pi * nwater / lamda
    sin_theta_water = noil / nwater * sin_theta_oil
    circ_water = sin_theta_water < 1
    circ_water = circ_water.astype("float32")
    sin_theta_water = sin_theta_water * circ_water
    cos_theta_water = np.sqrt(1 - sin_theta_water ** 2) * circ_water

    # circular aperture to impose on the mask in the SLM
    circ = circ_water.astype("float32")

    # Inputs for the calculation of lateral and axial phases
    Xgrid = 2 * pi * XI * M / (lamda * f_4f)
    Ygrid = 2 * pi * ETA * M / (lamda * f_4f)
    Zgrid = kwater * cos_theta_water

    # save grids to bfp_grids dictionary
    bfp_grids = {}
    bfp_grids['circ'] = torch.from_numpy(circ)
    bfp_grids['Xgrid'], bfp_grids['Ygrid'], bfp_grids['Zgrid'] = torch.from_numpy(Xgrid), torch.from_numpy(
        Ygrid), torch.from_numpy(Zgrid)

    return bfp_grids

class MaskPhasesToPSFs(nn.Module):
    def __init__(self, mask, optics_dict):
        super().__init__()
        self.mask = mask
        bfp_grids = calc_bfp_grids(self.mask, optics_dict)
        self.circ = nn.Parameter(bfp_grids['circ'], requires_grad=False)
        self.defocusAberr = nn.Parameter(bfp_grids['defocusAberr'], requires_grad=False)
        self.Xgrid = nn.Parameter(bfp_grids['Xgrid'], requires_grad=False)
        self.Ygrid = nn.Parameter(bfp_grids['Ygrid'], requires_grad=False)
        self.Zgrid = nn.Parameter(bfp_grids['Zgrid'], requires_grad=False)
        self.pad_px = bfp_grids['pad_px']

    def forward(self, xy, z):

        phase_lat = torch.exp(1j * (xy[:, :, [0]].unsqueeze(-1) * self.Xgrid + xy[:, :, [1]].unsqueeze(-1) * self.Ygrid))
        phase_ax = torch.exp(1j * z.unsqueeze(-1) * self.Zgrid)
        phase_emitter_location = phase_lat * phase_ax

        # total constant phase due to emitter location and general defocus aberration
        phase_emitter = phase_emitter_location * self.defocusAberr * self.circ

        # padding
        pad_px = self.pad_px
        phase_emitter = F.pad(phase_emitter, (pad_px, pad_px, pad_px, pad_px))

        newphase = phase_emitter * torch.exp(1j * self.mask)

        fft_res = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(newphase), norm='forward'))

        # take the absolute value squared
        fft_abs_square = torch.abs(fft_res) ** 2

        PSF = F.pad(fft_abs_square, (-pad_px, -pad_px, -pad_px, -pad_px))

        return PSF.type(torch.float)

class MaskPhasesToPSFs2D(nn.Module):
    def __init__(self, optics_dict):
        super().__init__()
        bfp_grids = calc_bfp_grids2D(optics_dict)
        self.circ = nn.Parameter(bfp_grids['circ'], requires_grad=False)
        self.Xgrid = nn.Parameter(bfp_grids['Xgrid'], requires_grad=False)
        self.Ygrid = nn.Parameter(bfp_grids['Ygrid'], requires_grad=False)
        self.Zgrid = nn.Parameter(bfp_grids['Zgrid'], requires_grad=False)

    def forward(self, xy, angles=None):

        phase_lat = torch.exp(1j * (xy[:, :, [0]].unsqueeze(-1) * self.Xgrid + xy[:, :, [1]].unsqueeze(-1) * self.Ygrid))

        phase_emitter_location = phase_lat

        # total constant phase due to emitter location and general defocus aberration
        phase_emitter = phase_emitter_location * self.circ

        fft_res = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(phase_emitter), norm='forward'))

        # take the absolute value squared
        fft_abs_square = torch.abs(fft_res) ** 2

        # depth-wise normalization factor
        PSF = fft_abs_square

        return PSF.type(torch.float)

class PhysicalLayer_mask_rec(nn.Module):
    def __init__(self, optics_dict, W=None):
        super(PhysicalLayer_mask_rec, self).__init__()
        phase_mask = sio.loadmat(optics_dict['phase_mask_root'])['maskRec']
        self.mask = nn.Parameter(torch.from_numpy(phase_mask).type(torch.FloatTensor), requires_grad=False)
        self.psf_crop = W
        self.MaskPhasesToPSFs = MaskPhasesToPSFs(self.mask, optics_dict)
        self.crop = Croplayer(self.psf_crop)
        self.blur = transforms.GaussianBlur(5, sigma=optics_dict['est_gblur'])
        self.norm01 = Normalize01()

    def forward(self, xy, z):
        PSF4D = self.MaskPhasesToPSFs(xy, z)

        # blur each emitter with slightly different gaussian
        images4D_blur = self.blur(PSF4D)

        # crop relevant FOV area
        images4D_blur_crop = self.crop(images4D_blur)

        PSFs = self.norm01(images4D_blur_crop)

        return PSFs

class PhysicalLayer_mask_rec2D(nn.Module):
    def __init__(self, optics_dict, W=None):
        super(PhysicalLayer_mask_rec2D, self).__init__()
        self.psf_crop = W
        self.MaskPhasesToPSFs = MaskPhasesToPSFs2D(optics_dict)
        # self.crop = transforms.CenterCrop(W)
        self.crop = Croplayer(self.psf_crop)
        self.blur = transforms.GaussianBlur(5, sigma=optics_dict['est_gblur'])
        self.norm01 = Normalize01()

    def forward(self, xy, angles=None):
        PSF4D = self.MaskPhasesToPSFs(xy, angles=angles)

        # blur each emitter with slightly different gaussian
        images4D_blur = self.blur(PSF4D)

        # crop relevant FOV area
        images4D_blur_crop = self.crop(images4D_blur)

        PSFs = self.norm01(images4D_blur_crop)

        return PSFs
