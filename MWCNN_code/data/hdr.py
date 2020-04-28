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
        print("HDR init start")
        super(HDR, self).__init__(args, train)
        self.repeat = 1#args.test_every // (args.n_train // args.batch_size)
        self.num_samples = 1
        print("HDR init ent")

    def _scan(self):
        print("scan in HDR")
        if self.train and False:
            list_hr = [i for i in range(self.num)]
        else:
            # list_hr = []
            # # list_lr = [[] for _ in self.scale]
            #for entry in os.scandir(self.dir_hr):
            #    filename = os.path.splitext(entry.name)[0]
            #    list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            list_hr = sorted(glob.glob("/home/mppowers/train/*/*.dng"))

            #list_hr.sort()
        self.num = len(list_hr)
        self.num_samples = self.num
        print("self.num", self.num)
        return list_hr#[i for i in range(self.num)]#, list_lr

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx
    # do noting
    def _set_filesystem(self, dir_data):
        return # raise NotImplementedError
