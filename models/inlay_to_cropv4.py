import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from helpers.layers import ResnetBlock, get_decoder, str_to_activ_module
from helpers.distributions import nll


class InlayToCropProjector(nn.Module):
    ''' takes an inlay and a latent vector(z) and returns a crop prediction '''
    def __init__(self, posterior_size, config):
        super(InlayToCropProjector, self).__init__()
        self.config = config
        self.posterior_size = posterior_size
        self.z_to_img_net = self._build_model()

    def parallel(self):
        self.z_to_img_net = nn.DataParallel(self.z_to_img_net)

    def _build_model(self):
        #assert self.config['window_size'] == 32, "can only operate over window size of 32"
        return get_decoder(self.config, reupsample=True, name='z_to_img_decoder')(
            input_size=self.posterior_size,
            output_shape=[self.config['img_shp'][0], # channels derived from original
                          self.config['window_size'],
                          self.config['window_size']]#,
            #activation_fn=str_to_activ_module(self.config['activation'])
        )

    def loss_function(self, crop_pred, crop):
        batch_size = crop_pred[0].size(0)
        crop_loss = [nll(crop_i, crop_pred_i, nll_type=self.config['nll_type'])
                     for crop_pred_i, crop_i in zip(crop_pred, crop)]

        # reduce mean on time-dimension
        crop_loss = torch.cat([cl.unsqueeze(0) for cl in crop_loss], 0)
        return torch.mean(crop_loss, 0)

    def forward(self, z):
        ''' generates crops from z '''
        return self.z_to_img_net(z)
