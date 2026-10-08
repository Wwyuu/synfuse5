# -*- coding: utf-8 -*-
"""Full-resolution no-reference metrics aligned to pansharpening papers:

  D_lambda  光谱失真指数 (Spectral distortion index)  — lower better
  D_s       空间失真指数 (Spatial distortion index)   — lower better
  QNR       无参考质量 = (1-D_lambda)*(1-D_s)         — higher better

Adapted from py-tra/metrics.py (IQA pansharpening reference).
Images: HxWxC, uint8 [0,255] or float. pan is HxWx1.
"""
from __future__ import print_function
import numpy as np
import cv2
from scipy import ndimage


def _qindex(img1, img2, block_size=8):
    assert block_size > 1
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size, block_size)) / (block_size ** 2)
    pad_topleft = int(np.floor(block_size / 2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(img1_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1_ ** 2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_ ** 2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu2_sq
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_mu2
    qindex_map = np.ones(sigma12.shape)
    idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    qindex_map[idx] = ((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    return np.mean(qindex_map)


def _gaussian2d(N, std):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2)
    t1, t2 = np.meshgrid(t, t)
    std = np.double(std)
    w = np.exp(-0.5 * (t1 / std) ** 2) * np.exp(-0.5 * (t2 / std) ** 2)
    return w


def _kaiser2d(N, beta):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2) / np.double(N - 1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0
    return w


def _fir_filter_wind(Hd, w):
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = h / np.sum(h)
    return h


def _GNyq2win(GNyq, scale=4, N=41):
    fcut = 1 / scale
    alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq)))
    H = _gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = _kaiser2d(N, 0.5)
    h = _fir_filter_wind(Hd, w)
    return np.real(h)


def _satellite_gnyq(satellite):
    """Return (GNyq_bands[4], GNyq_pan). Unknown sensors fall back to QuickBird."""
    name = (satellite or 'QuickBird').lower().replace('-', '').replace('_', '')
    # Band order assumed B,G,R,NIR
    table = {
        'quickbird': ([0.34, 0.32, 0.30, 0.22], 0.15),
        'ikonos': ([0.26, 0.28, 0.29, 0.28], 0.17),
        # Common practice in this codebase family: alias GF/WV to QuickBird MTF
        'gaofen2': ([0.34, 0.32, 0.30, 0.22], 0.15),
        'gf2': ([0.34, 0.32, 0.30, 0.22], 0.15),
        'worldview2': ([0.34, 0.32, 0.30, 0.22], 0.15),
        'wv2': ([0.34, 0.32, 0.30, 0.22], 0.15),
        'worldview3': ([0.34, 0.32, 0.30, 0.22], 0.15),
        'wv3': ([0.34, 0.32, 0.30, 0.22], 0.15),
    }
    if name not in table:
        return table['quickbird']
    return table[name]


def mtf_resize(img, satellite='QuickBird', scale=4):
    scale = int(scale)
    GNyq, GNyqPan = _satellite_gnyq(satellite)
    img_ = img.squeeze().astype(np.float64)
    if img_.ndim == 2:
        H, W = img_.shape
        lowpass = _GNyq2win(GNyqPan, scale, N=41)
    elif img_.ndim == 3:
        H, W, _ = img.shape
        lowpass = np.stack([_GNyq2win(g, scale, N=41) for g in GNyq], axis=-1)
    else:
        raise ValueError('img must be 2D or 3D')
    # scipy API compatibility
    if hasattr(ndimage, 'correlate'):
        img_ = ndimage.correlate(img_, lowpass, mode='nearest')
    else:
        img_ = ndimage.filters.correlate(img_, lowpass, mode='nearest')
    output_size = (H // scale, W // scale)
    img_ = cv2.resize(img_, dsize=output_size, interpolation=cv2.INTER_NEAREST)
    return img_


def D_lambda(img_fake, img_lm, block_size=32, p=1):
    assert img_fake.ndim == img_lm.ndim == 3
    C_f = img_fake.shape[2]
    assert C_f == img_lm.shape[2]
    Q_fake, Q_lm = [], []
    for i in range(C_f):
        for j in range(i + 1, C_f):
            Q_fake.append(_qindex(img_fake[..., i], img_fake[..., j], block_size=block_size))
            Q_lm.append(_qindex(img_lm[..., i], img_lm[..., j], block_size=block_size))
    Q_fake = np.array(Q_fake)
    Q_lm = np.array(Q_lm)
    return float(((np.abs(Q_fake - Q_lm) ** p).mean()) ** (1.0 / p))


def D_s(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, q=1):
    assert img_fake.ndim == img_lm.ndim == 3
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert H_f // H_r == W_f // W_r == scale
    assert C_f == C_r
    assert pan.ndim == 3 and pan.shape[2] == 1
    assert H_f == pan.shape[0] and W_f == pan.shape[1]
    pan_lr = mtf_resize(pan, satellite=satellite, scale=scale)
    Q_hr, Q_lr = [], []
    for i in range(C_f):
        Q_hr.append(_qindex(img_fake[..., i], pan[..., 0], block_size=block_size))
        Q_lr.append(_qindex(img_lm[..., i], pan_lr, block_size=block_size))
    Q_hr = np.array(Q_hr)
    Q_lr = np.array(Q_lr)
    return float(((np.abs(Q_hr - Q_lr) ** q).mean()) ** (1.0 / q))


def qnr(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, p=1, q=1, alpha=1, beta=1):
    dl = D_lambda(img_fake, img_lm, block_size, p)
    ds = D_s(img_fake, img_lm, pan, satellite, scale, block_size, q)
    return float(((1 - dl) ** alpha) * ((1 - ds) ** beta)), dl, ds


def no_ref_evaluate(pred, pan, lms, satellite='QuickBird', scale=4):
    """pred HxWxC, pan HxWx1, lms hxwxC — all uint8 preferred."""
    qnr_v, dl, ds = qnr(pred, lms, pan, satellite=satellite, scale=scale)
    return {'D_lambda': dl, 'D_s': ds, 'QNR': qnr_v}
