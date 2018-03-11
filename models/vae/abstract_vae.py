from __future__ import print_function
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict, Counter

from helpers.utils import float_type
from models.relational_network import RelationalNetwork
from helpers.layers import View, flatten_layers, \
    build_conv_encoder, build_dense_encoder, \
    build_relational_conv_encoder, build_conv_decoder, \
    build_dense_decoder
from helpers.distributions import nll_activation as nll_activation_fn
from helpers.distributions import nll as nll_fn


class AbstractVAE(nn.Module):
    ''' abstract base class for VAE, both sequentialVAE and parallelVAE inherit this '''
    def __init__(self, input_shape, output_size, activation_fn=nn.ELU, **kwargs):
        super(AbstractVAE, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.activation_fn = activation_fn
        self.is_color = input_shape[0] > 1
        self.chans = 3 if self.is_color else 1

        # grab the meta config and print for
        self.config = kwargs['kwargs']
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)

        # placeholder in order to sequentialize model
        self.full_model = None

    def get_name(self, reparam_str):
        ''' helper to get the name of the model '''
        es_str = "es" + str(int(self.config['early_stop'])) if self.config['early_stop'] \
                 else "epochs" + str(self.config['epochs'])
        full_hash_str = "_{}_{}act{}_input{}_batch{}_mut{}_filter{}_nll{}_lr{}_{}_ngpu{}".format(
            str(self.config['layer_type']),
            reparam_str,
            str(self.activation_fn.__name__),
            str(self.input_shape),
            str(self.config['batch_size']),
            str(self.config['mut_reg']),
            str(self.config['filter_depth']),
            str(self.config['nll_type']),
            str(self.config['lr']),
            es_str,
            str(self.config['ngpu'])
        )
        full_hash_str = full_hash_str.strip().lower().replace('[', '')  \
                                                     .replace(']', '')  \
                                                     .replace(' ', '')  \
                                                     .replace('{', '') \
                                                     .replace('}', '') \
                                                     .replace(',', '_') \
                                                     .replace(':', '') \
                                                     .replace('(', '') \
                                                     .replace(')', '') \
                                                     .replace('\'', '')
        task_cleaned = AbstractVAE._clean_task_str(self.config['task'])
        return task_cleaned + full_hash_str

    @staticmethod
    def _clean_task_str(task_str):
        ''' helper to reduce string length.
            eg: mnist+svhn+mnist --> mnist2svhn1 '''
        result_str = ''
        if '+' in task_str:
            splits = Counter(task_str.split('+'))
            for k, v in splits.items():
                result_str += '{}{}'.format(k, v)

            return result_str

        return task_str

    def build_encoder(self):
        ''' helper function to build convolutional or dense encoder '''
        encoder_output_size = self.reparameterizer.input_size
        if self.config['layer_type'] == 'conv':
            if self.config['use_relational_encoder']:
                encoder = build_relational_conv_encoder(input_shape=self.input_shape,
                                                        filter_depth=self.config['filter_depth'],
                                                        activation_fn=self.activation_fn)
            else:
                encoder = build_conv_encoder(input_shape=self.input_shape,
                                             output_size=encoder_output_size,
                                             filter_depth=self.config['filter_depth'],
                                             activation_fn=self.activation_fn)
        elif self.config['layer_type'] == 'dense':
            print("is = ", self.input_shape)
            encoder = build_dense_encoder(input_shape=self.input_shape,
                                          output_size=encoder_output_size,
                                          latent_size=512,
                                          activation_fn=self.activation_fn)
        else:
            raise Exception("unknown layer type requested")

        if self.config['ngpu'] > 1:
            encoder = nn.DataParallel(encoder)

        if self.config['cuda']:
            encoder = encoder.cuda()

        return encoder

    def build_decoder(self):
        ''' helper function to build convolutional or dense decoder'''
        decoder_input_size = self.reparameterizer.output_size
        if self.config['layer_type'] == 'conv':
            decoder = build_conv_decoder(input_size=decoder_input_size,
                                         output_shape=self.input_shape,
                                         filter_depth=self.config['filter_depth'],
                                         activation_fn=self.activation_fn)
        elif self.config['layer_type'] == 'dense':
            decoder = build_dense_decoder(input_size=decoder_input_size,
                                          output_shape=self.input_shape,
                                          activation_fn=self.activation_fn)
        else:
            raise Exception("unknown layer type requested")

        if self.config['ngpu'] > 1:
            decoder = nn.DataParallel(decoder)

        if self.config['cuda']:
            decoder = decoder.cuda()

        return decoder

    def _lazy_init_dense(self, input_size, output_size, name='enc_proj'):
        '''initialize the dense linear projection lazily
           because determining convolutional output size
           is annoying '''
        if not hasattr(self, name):
            # build a simple linear projector
            setattr(self, name, nn.Sequential(
                View([-1, input_size]),
                nn.Linear(input_size, output_size)
            ))

            if self.config['ngpu'] > 1:
                setattr(self, name,
                        nn.DataParallel(getattr(self, name))
                )

            if self.config['cuda']:
                setattr(self, name, getattr(self, name).cuda())

    def _lazy_init_relational(self, output_size, name='enc_proj'):
        '''initialize a relational network lazily
           because determining convolutional output size
           is annoying '''
        if not hasattr(self, name):
            setattr(self, name, RelationalNetwork(hidden_size=512, #XXX
                                                  output_size=output_size,
                                                  cuda=self.config['cuda'],
                                                  ngpu=self.config['ngpu']))

            if self.config['ngpu'] > 1:
                setattr(self, name,
                        nn.DataParallel(getattr(self, name))
                )

            if self.config['cuda']:
                setattr(self, name, getattr(self, name).cuda())

    def compile_full_model(self):
        ''' takes all the submodules and module-lists
            and returns one gigantic sequential_model '''
        if not self.full_model:
            full_model_list, _ = flatten_layers(self)
            self.full_model = nn.Sequential(OrderedDict(full_model_list))

    def nll_activation(self, logits):
        return F.log_softmax(logits)

    def forward(self, x):
        ''' params is a map of the latent variable's parameters'''
        z, params = self.posterior(x)
        return self.decode(z), params

    def loss_function(self, reconstructions, x, params, mut_info=None):
        ''' the loss function here is P(y | f(x; theta)) '''
        #nll = F.cross_entropy(input=predictions, target=labels, reduce=False)
        nll = nll_fn(x, reconstructions, self.config['nll_type'])
        kld = self.kld(params)
        elbo = nll + kld

        # handle the mutual information term
        if mut_info is None:
            mut_info = Variable(
                float_type(self.config['cuda'])(x.size(0)).zero_()
            )
        else:
            # Clamping strategies: 2 and 3 are about the same [empirically in ELBO]
            # mut_info = self.config['mut_reg'] * torch.clamp(mut_info, min=0, max=torch.norm(kld, p=2).data[0])
            # mut_info = self.config['mut_reg'] * mut_info
            mut_info = self.config['mut_reg'] * (mut_info / torch.norm(mut_info, p=2))

        loss = elbo + mut_info
        return {
            'loss': loss,
            'loss_mean': torch.mean(loss),
            'elbo_mean': torch.mean(elbo),
            'nll_mean': torch.mean(nll),
            'kld_mean': torch.mean(kld),
            'mu_mean': params['gaussian']['mu_mean'],
            'logvar_mean': params['gaussian']['logvar_mean'],
            'mut_info_mean': torch.mean(mut_info)
        }

    def has_discrete(self):
        ''' returns True if the model has a discrete
            as it's first (in the case of parallel) reparameterizer'''
        raise NotImplementedError("has_discrete not implemented")

    def get_reparameterizer_scalars(self):
        ''' returns a map of the scalars of the reparameterizers.
            This is useful for visualization purposes'''
        raise NotImplementedError("get_reparameterizer_scalars not implemented")

    def reparameterize(self, logits):
        ''' reparameterizes the latent logits appropriately '''
        raise NotImplementedError("reparameterize not implemented")

    def decode(self, z):
        '''returns logits '''
        raise NotImplementedError("decode not implemented")

    def posterior(self, x):
        ''' get a reparameterized Q(z|x) for a given x '''
        z_logits = self.encode(x)
        return self.reparameterize(z_logits)

    def encode(self, x):
        ''' encodes via a convolution and returns logits '''
        raise NotImplementedError("encode not implemented")

    def generate(self, z):
        ''' returns a generation for a given z '''
        raise NotImplementedError("generate not implemented")

    def kld(self, dist_params):
        ''' KL divergence between dist_a and prior '''
        raise NotImplementedError("kld not implemented")

    def mut_info(self, dist_params):
        ''' helper to get the mutual info to add to the loss '''
        raise NotImplementedError("mut_info not implemented")
