import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

import glob

class HDRTest(srdata.SRData):
    def __init__(self, args, train=True):
        self.glob_argument = args.hdr_test_dir + "*"
        super(HDRTest, self).__init__(args, train)
        self.repeat = 1#args.test_every // (args.n_train // args.batch_size)

    def _scan(self):
        list_hr = sorted(glob.glob(self.glob_argument))

        self.num = len(list_hr)
        self.num_samples = self.num
        print("number of training samples: ", self.num)
        return list_hr#[i for i in range(self.num)]#, list_lr

    def __len__(self):
        if self.args.test_only:
            return len(self.images_hr) * 2 * 2
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        return idx % (len(self.images_hr) * 2 * 2)

    # do noting
    def _set_filesystem(self, dir_data):
        return # raise NotImplementedError
