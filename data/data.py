#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-16 19:22:41
LastEditTime: 2021-01-19 20:55:10
@Description: file content
'''
from os.path import join
from torchvision.transforms import Compose, ToTensor
from .dataset import Data, Data_test, Data_eval, Data_fullres
from torchvision import transforms
import torch, numpy  #h5py, 
import torch.utils.data as data
import os

def transform():
    return Compose([
        ToTensor(),
    ])
    
def get_data(cfg, mode):
    data_dir_ms = join(mode, cfg['source_ms'])
    data_dir_pan = join(mode, cfg['source_pan'])
    cfg = cfg
    return Data(data_dir_ms, data_dir_pan, cfg, transform=transform())
    
def get_test_data(cfg, mode):
    data_dir_ms = join(mode, cfg['test']['source_ms'])
    data_dir_pan = join(mode, cfg['test']['source_pan'])
    cfg = cfg
    return Data_test(data_dir_ms, data_dir_pan, cfg, transform=transform())

def _resolve_fullres_dirs(data_dir, source_ms='ms', source_pan='pan'):
    """Accept either root/{ms,pan} or root/test128/{ms,pan}."""
    direct_ms = join(data_dir, source_ms)
    direct_pan = join(data_dir, source_pan)
    if os.path.isdir(direct_ms) and os.path.isdir(direct_pan):
        return direct_ms, direct_pan
    nested = join(data_dir, 'test128')
    nested_ms = join(nested, source_ms)
    nested_pan = join(nested, source_pan)
    if os.path.isdir(nested_ms) and os.path.isdir(nested_pan):
        return nested_ms, nested_pan
    raise FileNotFoundError(
        'Full-res data not found under %s (need ms/ + pan/ or test128/ms + test128/pan)' % data_dir
    )

def get_eval_data(cfg, data_dir, upscale_factor=None):
    source_ms = cfg.get('test', {}).get('source_ms', cfg.get('source_ms', 'ms'))
    source_pan = cfg.get('test', {}).get('source_pan', cfg.get('source_pan', 'pan'))
    data_dir_ms, data_dir_pan = _resolve_fullres_dirs(data_dir, source_ms, source_pan)
    return Data_fullres(data_dir_ms, data_dir_pan, cfg, transform=transform())

def get_fullres_data(cfg, data_dir):
    return get_eval_data(cfg, data_dir)