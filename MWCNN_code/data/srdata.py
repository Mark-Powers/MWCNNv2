import os

from data import common

import numpy as np
import scipy.misc as misc
import scipy.io as sio
from scipy.misc import imresize

import torch
import torch.utils.data as data
import h5py

import rawpy
import imageio
import glob
import math
import time

class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0

        # since we have too much data, we don't want to load it all at once
        if train and False:
            #mat = h5py.File('../MWCNN/imdb_gray.mat')
            #self.args.ext = 'mat'
            #self.hr_data = mat['images']['labels'][:,:,:,:]
            dng_list = [];
            for f in files:
                with rawpy.imread(f) as raw:
                    dng_list.append(raw.postprocess())
            self.hr_data = np.array(dng_list)

            self.num = self.hr_data.shape[0]
            #print(self.hr_data.shape)

        if self.split == 'test':
            self._set_filesystem(args.dir_data)

        # this should be a list of file names
        self.images_hr = self._scan()
        self.num = len(self.images_hr)


    def _scan(self):
        raise NotImplementedError
    #
    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    # def _name_hrbin(self):
    #     raise NotImplementedError

    # def _name_lrbin(self, scale):
    #     raise NotImplementedError

    def __getitem__(self, idx):
        hr, filename = self._load_file(idx)
        if self.train:
            lr, hr, scale = self._get_patch(hr, filename)
            lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
            return lr_tensor, hr_tensor, filename
        else:
            lr, hr, _ = self._get_patch(hr, filename)

            lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)

            return lr_tensor, hr_tensor, filename


    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        t = time.time()
        idx = self._get_index(idx)
        subimage_idx = idx%36
        filename_idx = math.floor(idx/36)
        # lr = self.images_lr[self.idx_scale][idx]
        filename = self.images_hr[filename_idx]
        with rawpy.imread(filename) as raw:
            hr = raw.postprocess()

        # running this gets unmatched dimensions error
        #hr = hr[:3024, :4032, :]

        # running this gets out of memory error
        #hr = hr[:3024, :3024, :]
        subimage_x_index = subimage_idx % 6
        subimage_y_index = math.floor(subimage_idx / 6)
        # we do min to make sure we don't bleed over edge of image
        x = min(subimage_x_index * 500, hr.shape[0]-512)
        y = min(subimage_y_index * 500, hr.shape[1]-512)
        hr = hr[x:x+512, y:y+512, :]
        print("reading", filename, " - ", time.time()-t," seconds")
        return hr, filename

    def _get_patch(self, hr, filename):
        patch_size = self.args.patch_size

        if self.train:
            scale = self.scale[0]
            #print("GETTING PATCH", hr.shape)
            if self.args.task_type == 'denoising':
                lr, hr = common.get_patch_noise(
                    hr, patch_size, scale
                )
            if self.args.task_type == 'SISR':
                lr, hr = common.get_patch_bic(
                    hr, patch_size, scale
                )
            if self.args.task_type == 'JIAR':
                lr, hr = common.get_patch_compress(
                    hr, patch_size, scale
                )
            if self.args.task_type == 'clip':
                lr, hr = common.get_patch_clip(
                    hr, 1.2, 0, 255
                )
            lr, hr = common.augment([lr, hr])
            return lr, hr, scale
        else:
            scale = self.scale[0]
            if self.args.task_type == 'denoising':
                lr, hr = common.get_img_noise(
                    hr, patch_size, scale
                )
            if self.args.task_type == 'SISR':
                lr, hr = self._get_patch_test(
                    hr, patch_size, scale
                )
            if self.args.task_type == 'JIAR':
                lr, hr = common.get_img_compress(
                    hr, patch_size, scale
                )
            if self.args.task_type == 'clip':
                lr, hr = common.get_patch_clip(
                    hr, 1.2, 0, 255
                )
            return lr, hr, scale
            # lr = common.add_noise(lr, self.args.noise)


    def _get_patch_test(self, hr, scale):

        ih, iw = hr.shape[0:2]
        lr = imresize(imresize(hr, [int(ih/scale), int(iw/scale)], 'bicubic'), [ih, iw], 'bicubic')
        ih = ih // 8 * 8
        iw = iw // 8 * 8
        hr = hr[0:ih, 0:iw, :]
        lr = lr[0:ih, 0:iw, :]

        return lr, hr




    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

