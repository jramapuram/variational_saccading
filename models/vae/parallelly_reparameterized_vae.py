from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from models.reparameterizers.gumbel import GumbelSoftmax
from models.reparameterizers.mixture import Mixture
from models.reparameterizers.isotropic_gaussian import IsotropicGaussian
from models.vae.abstract_vae import AbstractVAE


class ParallellyReparameterizedVAE(AbstractVAE):
    ''' This implementation uses a parallel application of
        the reparameterizer via the mixture type. '''
    def __init__(self, input_shape, output_size,
                 activation_fn=nn.ELU, **kwargs):
        super(ParallellyReparameterizedVAE, self).__init__(input_shape,
                                                           output_size,
                                                           activation_fn=activation_fn,
                                                           **kwargs)

        # build the reparameterizer
        if self.config['reparam_type'] == "isotropic_gaussian":
            print("using isotropic gaussian reparameterizer")
            self.reparameterizer = IsotropicGaussian(self.config)
        elif self.config['reparam_type'] == "discrete":
            raise NotImplementedError("discrete not implemented")
            # self.config['discrete_size'] = deepcopy(self.input_shape[-2:])
            # print("using gumbel softmax reparameterizer")
            # self.reparameterizer = GumbelSoftmax(self.config, dim=1)
        elif self.config['reparam_type'] == "mixture":
            print("using mixture reparameterizer")
            self.reparameterizer = Mixture(num_discrete=self.config['discrete_size'],
                                           num_continuous=self.config['continuous_size'],
                                           config=self.config)
        else:
            raise Exception("unknown reparameterization type")

        # build the encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def get_name(self):
        if self.config['reparam_type'] == "mixture":
            reparam_str = "mixturecat{}gauss{}_".format(
                str(self.config['discrete_size']),
                str(self.config['continuous_size'])
            )
        elif self.config['reparam_type'] == "isotropic_gaussian":
            reparam_str = "cont{}_".format(str(self.config['continuous_size']))
        elif self.config['reparam_type'] == "discrete":
            reparam_str = "disc{}_".format(str(self.config['discrete_size']))
        else:
            raise Exception("unknown reparam type")

        return 'parvae_' + super(ParallellyReparameterizedVAE, self).get_name(reparam_str)

    def has_discrete(self):
        ''' True is we have a discrete reparameterization '''
        return self.config['reparam_type'] == 'mixture' \
            or self.config['reparam_type'] == 'discrete'

    def get_reparameterizer_scalars(self):
        ''' basically returns tau from reparameterizers for now '''
        reparam_scalar_map = {}
        if isinstance(self.reparameterizer, GumbelSoftmax):
            reparam_scalar_map['tau_scalar'] = self.reparameterizer.tau
        elif isinstance(self.reparameterizer, Mixture):
            reparam_scalar_map['tau_scalar'] = self.reparameterizer.discrete.tau

        return reparam_scalar_map

    def decode(self, features):
        '''returns logits '''
        return self.decoder(features.contiguous())

    def posterior(self, x):
        z_logits = self.encode(x)
        return self.reparameterize(z_logits)

    def reparameterize(self, logits):
        ''' reparameterizes the latent logits appropriately '''
        return self.reparameterizer(logits)

    def encode(self, x):
        ''' encodes via a convolution
            and lazy init's a dense projector'''
        conv = self.encoder(x)         # do the convolution

        if self.config['use_relational_encoder']:
            # build a relational net as the encoder projection
            self._lazy_init_relational(self.reparameterizer.input_size, name='enc_proj')
        # else:
        #     # project via linear layer [if necessary!]
        #     conv_output_shp = int(np.prod(conv.size()[1:]))
        #     self._lazy_init_dense(conv_output_shp,
        #                           num_models*num_meta_model_layers,
        #                           name='enc_proj')

        # return projected units
        #return self.enc_proj(conv)#.view(-1, num_meta_model_layers, num_models)

        return conv

    def generate(self, z):
        ''' reparameterizer for sequential is different '''
        return self.decode(z)

    def kld(self, dist_a):
        ''' KL divergence between dist_a and prior '''
        return torch.sum(self.reparameterizer.kl(dist_a), dim=-1)

    def mut_info(self, dist_params):
        ''' helper to get mutual info '''
        mut_info = None
        if self.config['reparam_type'] == 'mixture' \
           or self.config['reparam_type'] == 'discrete':
            mut_info = self.reparameterizer.mutual_info(dist_params)

        return mut_info

    def loss_function(self, recon_x, x, params):
        ''' evaluates the loss of the model '''
        #mut_info = self.mut_info(params)
        mut_info = None
        return super(ParallellyReparameterizedVAE, self).loss_function(recon_x, x, params,
                                                                       mut_info=mut_info)
