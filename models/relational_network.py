import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

from helpers.utils import to_data, expand_dims, \
    int_type, float_type, long_type, add_weight_norm
from helpers.layers import build_conv_encoder, build_dense_encoder

class RelationalNetwork(nn.Module):
    def __init__(self, hidden_size, output_size, config):
        ''' Relational network that takes a list of imgs, projects
            each with a conv and then take each i-j tuple and runs an RN
            over it. Finally, the model returns a DNN over the summed
            values of the RN.
        '''
        super(RelationalNetwork, self).__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.output_size = output_size

        # build the projector and the rn models
        self.proj = self._build_proj_model()
        self.rn = self._build_rn_model()

    def _build_proj_model(self):
        return build_dense_encoder(self.hidden_size, self.output_size,
                                   normalization_str='batchnorm')#self.config['normalization'])

    def _lazy_build_image_model(self, input_size):
        if not hasattr(self, 'conv'):
            builder_fn = build_dense_encoder \
                if self.config['layer_type'] == 'dense' else build_conv_encoder
            self.img_model =  nn.Sequential(
                builder_fn(input_size, self.hidden_size,
                           normalization_str=self.config['normalization']),
                nn.SELU())

            if self.config['cuda']:
                self.img_model = self.img_model.cuda()

        return self.img_model

    def _build_rn_model(self):
        return nn.Sequential(
            build_dense_encoder(self.hidden_size*2, self.hidden_size,
                                normalization_str='batchnorm'),#self.config['normalization']),
            nn.SELU())

    def forward(self, imgs):
        assert type(imgs) == list, "need a set of input images"

        # get all the conv outputs
        self._lazy_build_image_model(list(imgs[0].size())[1:])
        conv_output = [self.img_model(img) for img in imgs]

        # project each x_i : x_j tuple through the RN
        rn_buffer = [self.rn(torch.cat([img_r, img_l], -1)).unsqueeze(0)
                     for img_r in conv_output
                     for img_l in conv_output]
        rn_buffer = torch.mean(torch.cat(rn_buffer, 0), 0)

        # return the summed buffer projected through a DNN
        return self.proj(rn_buffer)
