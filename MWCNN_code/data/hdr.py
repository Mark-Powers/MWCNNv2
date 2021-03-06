import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

import glob

class HDR(srdata.SRData):
    def __init__(self, args, train=True):
        self.glob_argument = args.hdr_train_dir + "*.dng"
        super(HDR, self).__init__(args, train)
        self.repeat = 1#args.test_every // (args.n_train // args.batch_size)

    def _scan(self):
        print("scan in HDR")
        list_hr = sorted(glob.glob(self.glob_argument))

        # we do 512x512 images with ridge 500
        self.num = len(list_hr) * 6 * 6
        self.num_samples = self.num
        print("number of training samples: ", self.num)
        return list_hr#[i for i in range(self.num)]#, list_lr

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat * 6 * 6
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % (len(self.images_hr) * 6 * 6)
        else:
            return idx

    # do noting
    def _set_filesystem(self, dir_data):
        return # raise NotImplementedError
