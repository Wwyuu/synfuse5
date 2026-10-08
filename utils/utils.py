#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:12:52
LastEditTime: 2020-11-25 23:00:57
@Description: file content
'''
import os, math, torch,cv2
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from utils.vgg import VGG
import torch.nn.functional as F
# from model.deepfuse import MEF_SSIM_Loss
class newLoss(nn.Module):
    def __init__(self,r1=1.1,r2=1000,offset=0.9995):
        super(newLoss, self).__init__()
        self.r1=r1
        self.r2=r2
        self.offset=offset
        print("new loss maked")
    def forward(self, input,target):
        del_x=torch.abs(input-target)
        compare_mut=torch.ones_like(del_x)
        y = ((del_x*255)**self.r1/255)*del_x*1#((torch.minimum(self.offset+del_x,compare_mut))**self.r2)
        return torch.sum(y)



def maek_optimizer(opt_type, cfg, params):
    if opt_type == "ADAM":
        optimizer = torch.optim.Adam(params, lr=cfg['schedule']['lr'], betas=(cfg['schedule']['beta1'], cfg['schedule']['beta2']), eps=cfg['schedule']['epsilon'])
    elif opt_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=cfg['schedule']['lr'], momentum=cfg['schedule']['momentum'])
    elif opt_type == "RMSprop":
        optimizer = torch.optim.RMSprop(params, lr=cfg['schedule']['lr'], alpha=cfg['schedule']['alpha'])
    else:
        raise ValueError
    return optimizer

def make_loss(loss_type):
    # loss = {}
    if loss_type == "MSE":
        loss = nn.MSELoss(reduction='sum')
    elif loss_type == "L1":
        loss = nn.L1Loss(reduction='sum')
    elif loss_type == "MEF_SSIM":
        loss = MEF_SSIM_Loss()
    elif loss_type == "VGG22":
        loss = VGG(loss_type[3:], rgb_range=255)
    elif loss_type == "VGG54":
        loss = VGG(loss_type[3:], rgb_range=255)
    elif loss_type == "newloss":
        loss = newLoss()

    else:
        raise ValueError
    return loss

def get_path(subdir):
    return os.path.join(subdir)

def save_config(time, log):
    log_dir = os.path.join('./log', str(time))
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'records.txt'), 'a') as log_file:
        log_file.write(str(log) + '\n')

def save_net_config(time, log):
    log_dir = os.path.join('./log', str(time))
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'net.txt'), 'a') as log_file:
        log_file.write(str(log) + '\n')

def calculate_psnr(img1, img2, pixel_range=255, color_mode='rgb'):
    # transfer color channel to y
    if color_mode == 'rgb':
        img1 = (img1 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
        img2 = (img2 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
    elif color_mode == 'yuv':
        img1 = img1[:, 0, :, :]
        img2 = img2[:, 0, :, :]
    elif color_mode == 'y':
        img1 = img1
        img2 = img2
    # img1 and img2 have range [0, pixel_range]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(pixel_range / math.sqrt(mse))

def ssim(img1, img2, pixel_range=255, color_mode='rgb'):
    C1 = (0.01 * pixel_range)**2
    C2 = (0.03 * pixel_range)**2

    # transfer color channel to y
    if color_mode == 'rgb':
        img1 = (img1 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
        img2 = (img2 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
    elif color_mode == 'yuv':
        img1 = img1[:, 0, :, :]
        img2 = img2[:, 0, :, :]
    elif color_mode == 'y':
        img1 = img1
        img2 = img2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, pixel_range=255):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2, pixel_range)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2, pixel_range))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2), pixel_range)
    else:
        raise ValueError('Wrong input image dimensions.')


# --- pansharpening ref metrics ---
# Same formulas as WINet/FAME `cpsnr`/`cssim` and `py-tra/metrics.py`
# (IQA_pansharpening_python). Full HWC, all bands, no RGB->Y.

def cpsnr(img1, img2, dynamic_range=255):
    """PSNR over all bands. uint8 range 255; uint16 often 2047."""
    if img1.shape != img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    mse = np.mean((img1_ - img2_) ** 2)
    if mse <= 1e-10:
        return np.inf
    return 20 * np.log10(dynamic_range / (np.sqrt(mse) + np.finfo(np.float64).eps))


def _band_ssim(img1, img2, dynamic_range=255):
    """SSIM for one band, shape (H, W)."""
    C1 = (0.01 * dynamic_range) ** 2
    C2 = (0.03 * dynamic_range) ** 2
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1_, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2_, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1_ ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_ ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def cssim(img1, img2, dynamic_range=255):
    """SSIM: per-band then mean, all channels. Same as py-tra/metrics.ssim."""
    if img1.shape != img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return _band_ssim(img1, img2, dynamic_range)
    if img1.ndim == 3:
        return np.mean([_band_ssim(img1[..., i], img2[..., i], dynamic_range)
                        for i in range(img1.shape[2])])
    raise ValueError('Wrong input image dimensions.')


def csam(img1, img2):
    """SAM in radians, mean over pixels. Same as py-tra/metrics.sam."""
    if img1.shape != img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    inner = (img1_ * img2_).sum(axis=2)
    n1 = np.sqrt((img1_ ** 2).sum(axis=2))
    n2 = np.sqrt((img2_ ** 2).sum(axis=2))
    cos_theta = (inner / (n1 * n2 + np.finfo(np.float64).eps)).clip(0, 1)
    return np.mean(np.arccos(cos_theta))


def cergas(img_fake, img_real, scale=4):
    """ERGAS, same as py-tra/metrics.ergas."""
    if img_fake.shape != img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_) ** 2)
        return 100.0 / scale * np.sqrt(mse / (mean_real ** 2 + np.finfo(np.float64).eps))
    means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
    mses = ((img_fake_ - img_real_) ** 2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
    return 100.0 / scale * np.sqrt((mses / (means_real ** 2 + np.finfo(np.float64).eps)).mean())