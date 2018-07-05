import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from helpers.layers import build_conv_encoder, build_dense_encoder
from helpers.utils import same_type, expand_dims


class RNNPosterior(nn.Module):
    def __init__(self, image_size, posterior_input_size, posterior_output_size,
                 hidden_size=256, num_layers=2, bidirectional=True, **kwargs):
        super(RNNPosterior, self).__init__()
        self.posterior_input_size = posterior_input_size
        self.posterior_output_size = posterior_output_size
        self.image_size = image_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.config = kwargs['kwargs'] # grab the meta-config
        self.image_encoder, self.rnn_model, self.output_projection, \
            self.h, self.c = self.build_model()

    def state_size(self):
        bidirectional_multiplier = 2 if self.bidirectional else 1
        return [self.num_layers * bidirectional_multiplier,
                -1,
                self.hidden_size]

    def _init_state(self, batch_size):
        '''return h or c initialized to 0'''
        state_size = self.state_size()
        state_size[1] = batch_size
        return Variable(same_type(self.config['half'], self.config['cuda'])(
            *state_size
        ).zero_())

    def init_states(self, batch_size):
        ''' helper to return h and c'''
        h = self._init_state(batch_size)
        c = self._init_state(batch_size)
        return (h, c)

    def build_model(self):
        ''' helper to build the model based on conv / dense
            in addition to the lstm to encode to the posterior
            at time T+1 '''
        if self.config['layer_type'] == 'conv':
            image_encoder = build_conv_encoder(self.image_size, self.posterior_output_size)
        else:
            image_encoder = build_dense_encoder(int(np.prod(self.image_size)),
                                                self.posterior_output_size)

        # build the main RNN model that takes
        # the previous posterior and image crop
        rnn_model = nn.LSTM(
            input_size=self.posterior_output_size*2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=False,
            bidirectional=self.bidirectional  # TODO: evaluate if useful
        )
        h, c = self.init_states(self.config['batch_size'])
        bidirectional_multiplier = 2 if self.bidirectional else 1
        output_projection = nn.Linear(self.hidden_size*bidirectional_multiplier,
                                      self.posterior_input_size)


        if self.config['cuda']:
            rnn_model = rnn_model.cuda()
            image_encoder = image_encoder.cuda()
            output_projection = output_projection.cuda()

        return image_encoder, rnn_model, output_projection, h, c

    def forward(self, x_tm1, posterior_tm1, reset_state=False):
        '''x_tm1: generally image crop at t-1
           posterior_tm1: posterior at t-1
           reset_state: resets states to 0'''
        if reset_state:
            self.h, self.c = self.init_states(x_tm1.size(0))

        # encode the image
        encoded_xtm1 = self.image_encoder(x_tm1)

        # concat the encoded image-crop and the previous posterior
        input_tm1 = torch.cat([encoded_xtm1, posterior_tm1], -1)
        if len(list(input_tm1.size())) < 3: # expand time dimension
            input_tm1 = expand_dims(input_tm1, 0)

        # project via RNN
        rnn_output_t, state_t = self.rnn_model(input_tm1, (self.h, self.c))

        # project via linear layer
        output_t = self.output_projection(rnn_output_t[-1])
        return output_t, state_t
