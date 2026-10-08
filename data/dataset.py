#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-23 14:57:22
LastEditTime: 2021-01-19 20:57:29
@Description: file content
'''
import torch.utils.data as data
import torch, random, os
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torch.nn.functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF'])


def load_img(filepath):
    img = Image.open(filepath)
    #img = Image.open(filepath)
    return img

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def get_patch(ms_image, lms_image, pan_image, bms_image, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = lms_image.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    lms_image = lms_image.crop((iy,ix,iy + ip, ix + ip))
    ms_image = ms_image.crop((ty,tx,ty + tp, tx + tp))
    pan_image = pan_image.crop((ty,tx,ty + tp, tx + tp))
    bms_image = bms_image.crop((ty,tx,ty + tp, tx + tp))
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return ms_image, lms_image, pan_image, bms_image, info_patch

def augment(ms_image, lms_image, pan_image, bms_image, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        ms_image = ImageOps.flip(ms_image)
        lms_image = ImageOps.flip(lms_image)
        pan_image = ImageOps.flip(pan_image)
        bms_image = ImageOps.flip(bms_image)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            ms_image = ImageOps.mirror(ms_image)
            lms_image = ImageOps.mirror(lms_image)
            pan_image = ImageOps.mirror(pan_image)
            bms_image = ImageOps.mirror(bms_image)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            ms_image = ms_image.rotate(180)
            lms_image = lms_image.rotate(180)
            pan_image = pan_image.rotate(180)
            bms_image = pan_image.rotate(180)
            info_aug['trans'] = True
            
    return ms_image, lms_image, pan_image, bms_image, info_aug

class Data(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None):
        super(Data, self).__init__()
    
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        
        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor, ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lms_image = ms_image.resize((int(ms_image.size[0]/self.upscale_factor),int(ms_image.size[1]/self.upscale_factor)), Image.BICUBIC)       
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)
           
        ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image, self.patch_size, scale=self.upscale_factor)
        
        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)
        
        if self.transform:
            ms_image = self.transform(ms_image)
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1

        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)

class Data_test(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None):
        super(Data_test, self).__init__()
    
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        
        ms_image = load_img(self.ms_image_filenames[index])
        #print(type(ms_image)) 'PIL.TiffImagePlugin.TiffImageFile'>
        #test_img=np.array(ms_image)
        #print(test_img.max())
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor, ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lms_image = ms_image.resize((int(ms_image.size[0]/self.upscale_factor), int(ms_image.size[1]/self.upscale_factor)), Image.BICUBIC)
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)

        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)
        
        if self.transform:
            ms_image = self.transform(ms_image)
            #print(ms_image.max())
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)
            #print(torch.max(ms_image))
            #print(torch.min(ms_image)) 

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1
 
        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)

class Data_fullres(data.Dataset):
    """Full-resolution (real-world) pairs: LRMS + HR PAN, no GT HRMS.

    Expected layout (either works):
      root/ms/*.tif , root/pan/*.tif
      or root/test128/ms , root/test128/pan
    MS is already low-res (e.g. 32x32); PAN is high-res (e.g. 128x128).
    """

    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None):
        super(Data_fullres, self).__init__()
        ms_names = sorted([x for x in listdir(data_dir_ms) if is_image_file(x)])
        pan_set = set([x for x in listdir(data_dir_pan) if is_image_file(x)])
        self.ms_image_filenames = []
        self.pan_image_filenames = []
        for name in ms_names:
            if name not in pan_set:
                continue
            self.ms_image_filenames.append(join(data_dir_ms, name))
            self.pan_image_filenames.append(join(data_dir_pan, name))
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        lms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])

        # Align to integer scale multiples; do NOT further downsample MS
        scale = self.upscale_factor
        lw = lms_image.size[0] - (lms_image.size[0] % 1)
        lh = lms_image.size[1] - (lms_image.size[1] % 1)
        lms_image = lms_image.crop((0, 0, lw, lh))
        pw = (pan_image.size[0] // scale) * scale
        ph = (pan_image.size[1] // scale) * scale
        pan_image = pan_image.crop((0, 0, pw, ph))
        # If PAN/LMS scale mismatch, resize LMS target to pan/scale
        target_lms = (pw // scale, ph // scale)
        if lms_image.size != target_lms:
            lms_image = lms_image.resize(target_lms, Image.BICUBIC)
        bms_image = rescale_img(lms_image, scale)

        if self.transform:
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1

        # Keep API close to Data_test: placeholder ms (=lms upsampled conceptually unused)
        return bms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)


# Backward-compatible alias (old Data_eval was broken)
Data_eval = Data_fullres