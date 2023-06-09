from __future__ import print_function
from __future__ import division
import os
import torch
import torchvision
import numpy as np
import PIL.Image


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes, transform = None, img_size = 224):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.img_size = (img_size,img_size)
        self.ys, self.im_paths, self.I = [], [], []
        self.raw_image_flag = False

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = PIL.Image.open(self.im_paths[index])
        # convert gray to rgb
        if len(list(im.split())) == 1 : im = im.convert('RGB')
        if(self.raw_image_flag):
            raw_image = np.asarray(im.resize(self.img_size))
        if self.transform is not None:
            im = self.transform(im)
        if(self.raw_image_flag):
            return im, self.ys[index], index, raw_image
        else:
            return im, self.ys[index], index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]
