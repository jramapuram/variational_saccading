import torch
import torch.nn as nn

from helpers.layers import View, build_dense_encoder, build_conv_encoder


class ImageStateProjector(nn.Module):
    def __init__(self, latent_size, output_size, config):
        super(ImageStateProjector, self).__init__()
        self.config = config
        self.output_size = output_size
        self.latent_size = latent_size
        self.conv, self.state_proj, self.out_proj = self._build_model()

    def forward(self, x, state):
        conv_features = self.conv(x)
        combined = torch.cat([state, conv_features], -1)
        return self.state_proj(combined)

    def parallel(self):
        self.conv = nn.DataParallel(self.conv)
        self.state_proj = nn.DataParallel(self.state_proj)
        self.out_proj = nn.DataParallel(self.out_proj)

    def get_output(self, accumulator):
        return self.out_proj(accumulator)

    def _build_model(self):
        ''' helper function to build convolutional or dense decoder
            chans * 2 because we want to do relationships'''
        from helpers.layers import build_conv_encoder, build_dense_encoder
        crop_size = [self.config['img_shp'][0],
                     self.config['window_size'],
                     self.config['window_size']]

        # main function approximator to decode the crop
        builder_fn = build_dense_encoder \
                if self.config['layer_type'] == 'dense' else build_conv_encoder
        decoder = nn.Sequential(
            builder_fn(crop_size, self.latent_size,
                       normalization_str=self.config['normalization']),
            # nn.SELU()# ,
            #nn.Dropout(p=0.5)
        )

        # takes the state + output of conv and projects it
        state_projector = nn.Sequential(
            build_dense_encoder(self.latent_size + self.latent_size, self.latent_size,
                                normalization_str='batchnorm'),
            # nn.SELU()# ,
            # nn.Dropout(p=0.5)
        )

        # takes the finally aggregated vector and projects to output dims
        output_projector = build_dense_encoder(self.latent_size, self.output_size,
                                               normalization_str='batchnorm')
        return decoder, state_projector, output_projector
