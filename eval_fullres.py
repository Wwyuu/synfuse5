#!/usr/bin/env python
# coding=utf-8
"""Full-resolution (real-world) no-reference evaluation: D_lambda / D_s / QNR.

Datasets on this project server (data/yaogan):
  fullGF2_data/test128/{ms,pan}     MS 32x32, PAN 128x128, ~200 pairs
  fullWV2_data_small/{ms,pan}       same geometry, ~200 pairs
  (no fullWV3 in current yaogan dump)

Example:
  python eval_fullres.py \\
    --algorithm nullpan \\
    --checkpoint checkpoint/nullpan_4_1790861156/bestPSNR.pth \\
    --data_dir data/yaogan/fullGF2_data \\
    --satellite GaoFen2 \\
    --save_dir result/fullres_nullpan_GF2

Notes:
  - No GT HRMS; only no-reference metrics.
  - Train still uses reduced-res (train128); this script is for paper Table "full-resolution".
"""
from __future__ import print_function
import argparse
import os
import json
import time

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from data.data import get_fullres_data
from research import import_algorithm
from research.losses.iso import unwrap_hrms
from utils.noref_metrics import no_ref_evaluate


def load_cfg(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def tensor_to_uint8_hwc(t):
    """BxCxHxW float[0,1] -> HxWxC uint8 (first item)."""
    x = t.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    x = x.clamp(0, 1).numpy().transpose(1, 2, 0)
    return np.uint8(np.round(x * 255.0))


def save_tif_cmyk(arr_hwc_u8, path):
    Image.fromarray(arr_hwc_u8, mode='CMYK').save(path)


def main():
    ap = argparse.ArgumentParser(description='Full-res QNR evaluation')
    ap.add_argument('--option_path', default='option.yml')
    ap.add_argument('--algorithm', default=None, help='override algorithm name')
    ap.add_argument('--checkpoint', required=True, help='path to .pth (contains net)')
    ap.add_argument('--data_dir', required=True, help='fullGF2_data or fullWV2_data_small root')
    ap.add_argument('--satellite', default='GaoFen2', help='GaoFen2|WorldView2|QuickBird|...')
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--save_dir', default='', help='optional dir to save fused tifs')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--max_samples', type=int, default=0, help='0 = all')
    args = ap.parse_args()

    cfg = load_cfg(args.option_path)
    if args.algorithm:
        cfg['algorithm'] = args.algorithm
    cfg['data']['upsacle'] = args.scale
    cfg.setdefault('test', {})
    cfg['test'].setdefault('source_ms', cfg.get('source_ms', 'ms'))
    cfg['test'].setdefault('source_pan', cfg.get('source_pan', 'pan'))

    algo = cfg['algorithm'].lower()
    lib = import_algorithm(algo)
    model = lib.Net(
        num_channels=cfg['data']['n_colors'],
        base_filter=64,
        args=cfg,
    )
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
    # strip possible module. prefix
    new_state = {}
    for k, v in state.items():
        new_state[k[7:] if k.startswith('module.') else k] = v
    model.load_state_dict(new_state, strict=True)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    model = model.to(device).eval()

    dataset = get_fullres_data(cfg, args.data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print('Full-res samples:', len(dataset), 'data_dir=', args.data_dir)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    rows = []
    t_all = []
    for i, batch in enumerate(loader):
        if args.max_samples and i >= args.max_samples:
            break
        # Data_fullres returns: bms, lms, pan, bms, file
        _, lms, pan, bms, name = batch
        lms = lms.to(device)
        pan = pan.to(device)
        bms = bms.to(device)
        t0 = time.time()
        with torch.no_grad():
            pred = unwrap_hrms(model(lms, bms, pan))
        t_all.append(time.time() - t0)

        pred_u8 = tensor_to_uint8_hwc(pred)
        lms_u8 = tensor_to_uint8_hwc(lms)
        pan_u8 = tensor_to_uint8_hwc(pan)
        if pan_u8.ndim == 2:
            pan_u8 = pan_u8[:, :, None]
        elif pan_u8.shape[2] != 1:
            pan_u8 = pan_u8[:, :, :1]

        metrics = no_ref_evaluate(
            pred_u8, pan_u8, lms_u8, satellite=args.satellite, scale=args.scale
        )
        fname = name[0] if isinstance(name, (list, tuple)) else str(name)
        rows.append({'file': fname, **metrics})
        print('[%03d] %s  D_λ(光谱失真)=%.4f  D_s(空间失真)=%.4f  QNR=%.4f' % (
            i, fname, metrics['D_lambda'], metrics['D_s'], metrics['QNR']))

        if args.save_dir:
            save_tif_cmyk(pred_u8, os.path.join(args.save_dir, fname if fname.lower().endswith('.tif') else fname + '.tif'))

    if not rows:
        raise RuntimeError('No samples evaluated. Check data_dir layout.')

    mean = {
        'D_lambda': float(np.mean([r['D_lambda'] for r in rows])),
        'D_s': float(np.mean([r['D_s'] for r in rows])),
        'QNR': float(np.mean([r['QNR'] for r in rows])),
        'n': len(rows),
        'avg_sec': float(np.mean(t_all)),
        'algorithm': algo,
        'checkpoint': args.checkpoint,
        'data_dir': args.data_dir,
        'satellite': args.satellite,
    }
    print('==== MEAN (real-world full-resolution, no-reference) ====')
    print('dataset: GaoFen2-style full-res | satellite=%s | n=%d' % (args.satellite, mean['n']))
    print('  光谱失真指数 D_lambda = %.4f  (lower better)' % mean['D_lambda'])
    print('  空间失真指数 D_s      = %.4f  (lower better)' % mean['D_s'])
    print('  无参考质量   QNR      = %.4f  (higher better)' % mean['QNR'])
    print('  timer %.3fs/img' % mean['avg_sec'])
    print('CSV: D_lambda,D_s,QNR')
    print('CSV: %.6f,%.6f,%.6f' % (mean['D_lambda'], mean['D_s'], mean['QNR']))

    out_json = os.path.join(args.save_dir or '.', 'fullres_metrics_%s.json' % algo)
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'mean': mean, 'per_image': rows}, f, indent=2, ensure_ascii=False)
    print('Wrote', out_json)


if __name__ == '__main__':
    main()
