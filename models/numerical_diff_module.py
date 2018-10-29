# coding: utf-8

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from copy import deepcopy

from .numerical_diff_autograd_v2 import NumDiffAutoGradFn
#from .numerical_diff_autograd import NumDiffAutoGradFn
from helpers.layers import str_to_activ_module, get_encoder


class NumericalDifferentiator(nn.Module):
    ''' takes an inlay and a latent vector(z) and returns a crop prediction '''
    def __init__(self, config):
        super(NumericalDifferentiator, self).__init__()
        self.autograd_fn = NumDiffAutoGradFn.apply
        self.sobel_x = torch.from_numpy(
            np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.from_numpy(
            np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        self.config = config

    def fp16(self):
        pass

    def parallel(self):
        pass

    def forward(self, z, crops):
        ''' forward passes through the numerical grad func '''
        assert z.shape[1] >= 3, "posterior sample needs to be >3 dimensional"
        if z.is_cuda and not self.sobel_x.is_cuda:
            self.sobel_x = self.sobel_x.cuda()
            self.sobel_y = self.sobel_y.cuda()

        # return self.autograd_fn(z[:, 0:3], crops,
        #                         self.config['window_size'],
        #                         self.config['crop_padding'])

        return self.autograd_fn(z[:, 0:3], crops,
                                self.sobel_x, self.sobel_y,
                                self.config['window_size'],
                                self.config['crop_padding'])
