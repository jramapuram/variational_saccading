import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from copy import deepcopy

from .saccade_autograd_v2 import SaccadeAutoGradFn
from helpers.layers import str_to_activ_module, get_encoder


class InlayToCropProjector(nn.Module):
    ''' takes an inlay and a latent vector(z) and returns a crop prediction '''
    def __init__(self):
        super(InlayToCropProjector, self).__init__()
        self.autograd_fn = SaccadeAutoGradFn.apply

    def fp16(self):
        pass

    def parallel(self):
        pass

    def forward(self, crops):
        ''' forward passes through the numerical grad func '''
        return self.autograd_fn(crops)
