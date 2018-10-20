import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from copy import deepcopy

from .numerical_diff_autograd import NumDiffAutoGradFn
from helpers.layers import str_to_activ_module, get_encoder


class NumericalDifferentiator(nn.Module):
    ''' takes an inlay and a latent vector(z) and returns a crop prediction '''
    def __init__(self, config):
        super(NumericalDifferentiator, self).__init__()
        self.autograd_fn = NumDiffAutoGradFn.apply
        self.config = config

    def fp16(self):
        pass

    def parallel(self):
        pass

    def forward(self, z, crops):
        ''' forward passes through the numerical grad func '''
        assert z.shape[1] >= 3, "posterior sample needs to be >3 dimensional"
        return self.autograd_fn(z[:, 0:3], crops, self.config['window_size'],
                                self.config['crop_padding'])
