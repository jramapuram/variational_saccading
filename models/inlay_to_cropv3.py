import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from copy import deepcopy

from .spatial_transformer import SpatialTransformer
from helpers.layers import str_to_activ_module, get_encoder
from helpers.distributions import nll


class InlayToCropProjector(nn.Module):
    ''' takes an inlay and a latent vector(z) and returns a crop prediction '''
    def __init__(self, reparam_size, config):
        super(InlayToCropProjector, self).__init__()
        self.config = config
        self.reparam_size = reparam_size
        self.spatial_transformer = SpatialTransformer(config)
        self.z_proj = self._get_dense(name="z_proj")(
            reparam_size, 3, latent_size=32, nlayers=1,
            normalization_str=self.config['dense_normalization'],
            activation_fn=str_to_activ_module(self.config['activation'])
        )

    def fp16(self):
        self.z_proj = self.z_proj.half()
        self.spatial_transformer = self.spatial_transformer.half()

    def parallel(self):
        self.z_proj = nn.DataParallel(self.z_proj)
        self.spatial_transformer = nn.DataParallel(self.spatial_transformer)

    def _get_dense(self, name):
        config = deepcopy(self.config)
        config['encoder_layer_type'] = 'dense'
        return get_encoder(config, name=name)

    def loss_function(self, crop_pred, crop):
        batch_size = crop_pred[0].size(0)
        crop_loss = [nll(crop_i, crop_pred_i, nll_type=self.config['nll_type'])
                     for crop_pred_i, crop_i in zip(crop_pred, crop)]

        # reduce mean on time-dimension
        crop_loss = torch.cat([cl.unsqueeze(0) for cl in crop_loss], 0)
        return torch.mean(crop_loss, 0)

    def forward(self, inlay, z):
        '''concats the image produced by the conv-decoder
           and the inlay and projects '''
        z_proj = self.z_proj(z)
        #return self.spatial_transformer(torch.tanh(z_proj), inlay)
        return self.spatial_transformer(z_proj, inlay)
