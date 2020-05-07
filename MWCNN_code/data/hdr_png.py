import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

import glob

class HDR_png(srdata.SRData):
    def __init__(self, args, train=True):
        self.glob_argument = args.hdr_train_dir + "*"
        super(HDR_png, self).__init__(args, train)
        self.repeat = 1#args.test_every // (args.n_train // args.batch_size)

    def _scan(self):
        print("scan in HDR")
        list_hr = sorted(glob.glob(self.glob_argument))

        self.num = len(list_hr)
        self.num_samples = self.num
        print("number of training samples: ", self.num)
        return list_hr#[i for i in range(self.num)]#, list_lr

    def __len__(self):
        return len(self.images_hr) * 2 * 2

    def _get_index(self, idx):
        return idx % (len(self.images_hr) * 2 * 2)

    # do noting
    def _set_filesystem(self, dir_data):
        return # raise NotImplementedError
