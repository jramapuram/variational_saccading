import torch
import torch.nn as nn

from copy import deepcopy

from helpers.layers import get_encoder, str_to_activ, str_to_activ_module


class ImageStateProjector(nn.Module):
    def __init__(self, output_size, config):
        super(ImageStateProjector, self).__init__()
        self.config = config
        self.output_size = output_size
        self.conv, self.state_proj, self.out_proj = self._build_model()

    def forward(self, x, state):
        conv_features = self.conv(x)
        # combined = str_to_activ(self.config['activation'])(
        #     torch.cat([state, conv_features], -1)
        # )
        combined = torch.cat([state, conv_features], -1)
        return self.state_proj(combined)

    def fp16(self):
        self.conv = self.conv.half()
        self.state_proj = self.state_proj.half()
        self.out_proj = self.out_proj.half()

    def parallel(self):
        self.conv = nn.DataParallel(self.conv)
        self.state_proj = nn.DataParallel(self.state_proj)
        self.out_proj = nn.DataParallel(self.out_proj)

    def get_output(self, accumulator):
        return self.out_proj(accumulator)

    def _get_dense(self, name='imsp'):
        config = deepcopy(self.config)
        config['encoder_layer_type'] = 'dense'
        return get_encoder(config, name=name)

    def _build_model(self):
        ''' helper function to build convolutional or dense decoder
            chans * 2 because we want to do relationships'''
        crop_size = [self.config['img_shp'][0],
                     self.config['window_size'],
                     self.config['window_size']]

        # main function approximator to extract crop features
        nlayer_map = {
            'conv': { 32: 4, 70: 6, 100: 7 },
            'resnet': { 32: None, 64: None, 70: None, 100: None },
            'dense': { 32: 3, 64: 3, 70: 3, 100: 3 }
        }
        bilinear_size = (self.config['window_size'], self.config['window_size'])
        conv = get_encoder(self.config, name='crop_feat')(
            crop_size, self.config['latent_size'],
            activation_fn=str_to_activ_module(self.config['activation']),
            num_layers=nlayer_map[
                self.config['encoder_layer_type']][self.config['window_size']
                ],
            bilinear_size=bilinear_size
        )

        # takes the state + output of conv and projects it
        # the +1 is for ACT
        state_projector = nn.Sequential(
            self._get_dense(name='state_proj')(
                self.config['latent_size']*2, self.config['latent_size']+1,
                normalization_str=self.config['dense_normalization'],
                activation_fn=str_to_activ_module(self.config['activation'])
            )
        )

        # takes the finally aggregated vector and projects to output dims
        output_projector = self._get_dense(name='output_proj')(
            self.config['latent_size'], self.output_size,
            normalization_str=self.config['dense_normalization'],
            activation_fn=str_to_activ_module(self.config['activation'])
        )

        return conv, state_projector, output_projector
