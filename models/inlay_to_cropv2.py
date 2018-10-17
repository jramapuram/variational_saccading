import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

from copy import deepcopy

from helpers.layers import get_encoder, str_to_activ_module, add_normalization, \
    GatedConv2d, Squeeze


class InlayToCropProjector(nn.Module):
    ''' takes an inlay and a latent vector(z) and returns a crop prediction '''
    def __init__(self, config):
        super(InlayToCropProjector, self).__init__()
        self.config = config
        self.z_proj, self.inlay_proj, self.joint_net = self._build_model()

    def parallel(self):
        self.z_proj = nn.DataParallel(self.z_proj)
        self.inlay_proj = nn.DataParallel(self.inlay_proj)
        self.joint_net = nn.DataParallel(self.joint_net)

    def _get_dense(self, name='imsp'):
        config = deepcopy(self.config)
        config['encoder_layer_type'] = 'dense'
        return get_encoder(config, name=name)

    def _build_model(self):
        assert self.config['differentiable_image_size'] == 128, "only supports 128x128 imgs"
        assert self.config['window_size'] == 32, "can only operate over window size of 32"

        # take z and project to latent_size
        z_proj = nn.Sequential(
            self._get_dense(name='z_proj')(3, self.config['latent_size']),
            str_to_activ_module(self.config['activation'])()
        )

        # take inlay and project to latent_size
        conv_fn = nn.Conv2d if self.config['disable_gated'] else GatedConv2d
        num_groups = max(int(min(np.ceil(self.config['latent_size'] / 2), 32)), 1)
        inlay_shp = [self.config['img_shp'][0],
                     self.config['differentiable_image_size'],
                     self.config['differentiable_image_size']]
        inlay_proj = nn.Sequential(
            get_encoder(self.config, name='inlay_proj')(
                input_shape=inlay_shp,
                output_size=self.config['latent_size'],
                activation_fn=str_to_activ_module(self.config['activation']),
                num_layers=4, # 7 projects to 3x3
                bilinear_size=inlay_shp[1:]
            ),
            str_to_activ_module(self.config['activation'])(),

            add_normalization(conv_fn(self.config['latent_size'], self.config['latent_size'], 5, stride=2),
                              self.config['conv_normalization'], 2, self.config['latent_size'], num_groups=num_groups),
            str_to_activ_module(self.config['activation'])(),
            add_normalization(conv_fn(self.config['latent_size'], self.config['latent_size'], 1, stride=1),
                              self.config['conv_normalization'], 2, self.config['latent_size'], num_groups=num_groups),
            str_to_activ_module(self.config['activation'])(),

            add_normalization(conv_fn(self.config['latent_size'], self.config['latent_size'], 5, stride=2),
                              self.config['conv_normalization'], 2, self.config['latent_size'], num_groups=num_groups),
            str_to_activ_module(self.config['activation'])(),
            add_normalization(conv_fn(self.config['latent_size'], self.config['latent_size'], 1, stride=1),
                              self.config['conv_normalization'], 2, self.config['latent_size'], num_groups=num_groups),
            str_to_activ_module(self.config['activation'])(),

            add_normalization(conv_fn(self.config['latent_size'], self.config['latent_size'], 4, stride=2),
                              self.config['conv_normalization'], 2, self.config['latent_size'], num_groups=num_groups),
            add_normalization(conv_fn(self.config['latent_size'], self.config['latent_size'], 1, stride=1),
                              self.config['conv_normalization'], 2, self.config['latent_size'], num_groups=num_groups),
            str_to_activ_module(self.config['activation'])(),
            Squeeze()
        )

        # takes z_proj and inlay_proj and runs through another dnn
        joint_net = self._get_dense(name='joint_crop_proj')(self.config['latent_size']*2,
                                                            self.config['latent_size'])
        return z_proj, inlay_proj, joint_net

    def loss_function(self, crop_pred, crop):
        crop_pred = torch.cat([cp.unsqueeze(1) for cp in crop_pred], 1)
        crop = torch.cat([cp.unsqueeze(1) for cp in crop], 1)
        return F.mse_loss(input=crop_pred, target=crop, size_average=True)

    def forward(self, inlay, z):
        '''concats the image produced by the conv-decoder
           and the inlay and projects '''
        z_projected = self.z_proj(z)
        inlay_projected = self.inlay_proj(inlay)
        print("z_proj = ", z_projected.shape, " | inlay_proj = ", inlay_projected.shape)
        return self.joint_net(
            torch.cat([z_projected, inlay_projected], -1)
        )
