import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.vae.abstract_vae import AbstractVAE
from models.reparameterizers.gumbel import GumbelSoftmax
from models.reparameterizers.mixture import Mixture
from models.reparameterizers.beta import Beta
from models.reparameterizers.isotropic_gaussian import IsotropicGaussian
from helpers.distributions import nll_activation as nll_activation_fn
from helpers.distributions import nll as nll_fn
from helpers.utils import eps as eps_fn
from helpers.utils import same_type, zeros_like, expand_dims, zeros, nan_check_and_break
from helpers.layers import build_gated_conv_encoder, build_conv_encoder, \
    build_dense_encoder, build_gated_conv_decoder, build_conv_decoder, \
    build_dense_decoder, build_gated_dense_encoder, build_gated_dense_decoder, \
    build_pixelcnn_decoder, str_to_activ_module


class VRNNMemory(nn.Module):
    ''' Helper object to contain states and outputs for the RNN'''
    def __init__(self, h_dim, n_layers, bidirectional,
                 config, rnn=None, cuda=False):
        super(VRNNMemory, self).__init__()
        self.model = rnn
        self.config = config
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.h_dim = h_dim
        self.use_cuda = cuda
        self.memory_buffer = []

    @staticmethod
    def _state_from_tuple(tpl):
        _, state = tpl
        return state

    @staticmethod
    def _output_from_tuple(tpl):
        output, _ = tpl
        return output

    def _append_to_buffer(self, tpl):
        output_t, state_t = tpl
        self.memory_buffer.append([output_t.clone(), (state_t[0].clone(),
                                                      state_t[1].clone())])

    def clear(self):
        self.memory_buffer.clear()

    def init_state(self, batch_size):
        def _init(batch_size):
            ''' return a single initialized state'''
            num_directions = 2 if self.bidirectional else 1
            if self.training and self.config['use_noisy_rnn_state']: # add some noise to initial state
                # nn.init.xavier_uniform_(
                return same_type(self.config['half'], self.config['cuda'])(
                    num_directions * self.n_layers, batch_size, self.h_dim
                ).normal_(0, 0.01).requires_grad_()

            # return zeros for testing
            return same_type(self.config['half'], self.config['cuda'])(
                num_directions * self.n_layers, batch_size, self.h_dim
            ).zero_().requires_grad_()

        self.state = ( # LSTM state is (h, c)
            _init(batch_size),
            _init(batch_size)
        )

    def init_output(self, batch_size, seqlen):
        self.outputs = same_type(self.config['half'], self.config['cuda'])(
            seqlen, batch_size, self.h_dim
        ).zero_().requires_grad_()

    def update(self, tpl):
        self._append_to_buffer(tpl)
        self.outputs, self.state = tpl

    def forward(self, input_t, reset_state=False):
        batch_size = input_t.size(0)
        if reset_state:
            self.init_state(batch_size)

        input_t = input_t.contiguous()

        # if not self.config['half']:
        self.update(self.model(input_t, self.state))
        # else:
        # self.update(self.model(input_t, collect_hidden=True))

        return self.get_output()

    def get_state(self):
        assert hasattr(self, 'state'), "do a forward pass first"
        return self.state

    def get_repackaged_state(self, h=None):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if h is None:
            return self.get_repackaged_state(self.state)

        if isinstance(h, torch.Tensor):
            return h.detach()

        return tuple(self.get_repackaged_state(v) for v in h)

    def get_output(self):
        assert hasattr(self, 'outputs'), "do a forward pass first"
        return self.outputs

    def get_merged_memory(self):
        ''' merges over nlayers of the state which is [nlayer, batch, latent]'''
        assert hasattr(self, 'memory_buffer'), "do a forward pass first"
        mem_concat = torch.cat([self._state_from_tuple(mem)[0]
                                for mem in self.memory_buffer], 0)
        return torch.mean(mem_concat, 0)

    def get_final_memory(self):
        assert hasattr(self, 'memory_buffer'), "do a forward pass first"
        return self._state_from_tuple(self.memory_buffer[-1])[0]


class VRNN(AbstractVAE):
    def __init__(self, input_shape, latent_size,
                 normalization="none", dense_activation="identity",
                 n_layers=2, bidirectional=False, **kwargs):
        """implementation of the Variational Recurrent
           Neural Network (VRNN) from https://arxiv.org/abs/1506.02216

           params:

             normalization: use a separate normalization for the VRNN [eg: GN doesnt work]
             dense_activation: separate from 'activation' provided from argparse for dense nets
             n_layers: number of RNN layers
             bidirectional: bidirectional RNN
        """
        super(VRNN, self).__init__(input_shape, **kwargs)
        self.latent_size = latent_size
        self.bidirectional = bidirectional
        self.normalization = normalization

        self.n_layers = n_layers

        # grab the activation nn.Module from the string
        self.dense_activation_fn = str_to_activ_module(dense_activation)

        # build the reparameterizer
        if self.config['reparam_type'] == "isotropic_gaussian":
            print("using isotropic gaussian reparameterizer")
            self.reparameterizer = IsotropicGaussian(self.config)
        elif self.config['reparam_type'] == "discrete":
            print("using gumbel softmax reparameterizer")
            self.reparameterizer = GumbelSoftmax(self.config)
        elif self.config['reparam_type'] == "beta":
            print("using beta reparameterizer")
            self.reparameterizer = Beta(self.config)
        elif self.config['reparam_type'] == "mixture":
            print("using mixture reparameterizer")
            self.reparameterizer = Mixture(num_discrete=self.config['discrete_size'],
                                           num_continuous=self.config['continuous_size'],
                                           config=self.config)
        else:
            raise Exception("unknown reparameterization type")

        # build the entire model
        self._build_model()

    def _build_phi_x_model(self):
        ''' simple helper to build the feature extractor for x'''
        return self._lazy_build_phi_x(self.input_shape)

    def _lazy_build_phi_x(self, input_shape):
        ''' simple helper to build the feature extractor for x
            NOTE: Expects input_shape as [chan, xdim, ydim]'''
        if self.config['layer_type'] == 'conv':    # build a conv encoder
            conv_builder = build_gated_conv_encoder \
                if self.config['disable_gated'] is False else build_conv_encoder
            phi_x = nn.Sequential(
                conv_builder(input_shape=[input_shape[0], -np.inf, -np.inf],  # ensures to use upsampler
                             output_size=self.latent_size,
                             filter_depth=self.config['filter_depth'],
                             activation_fn=self.activation_fn,
                             normalization_str=self.config['normalization']),
                nn.SELU()
            )
        else:  # conv encoder
            input_size = int(np.prod(input_shape))
            dense_builder = build_gated_dense_encoder \
                if self.config['disable_gated'] is False else build_dense_encoder
            phi_x = nn.Sequential(
                #nn.Upsample(size=(self.input_shape[1], self.input_shape[2]), mode='bilinear'),
                dense_builder(input_size, self.latent_size,
                              activation_fn=self.dense_activation_fn,
                              normalization_str=self.normalization,
                              nlayers=2),
                nn.SELU()
            )

        return phi_x.cuda() if self.config['cuda'] else phi_x


    def _lazy_rnn_lambda(self, x, state,
                          model_type='lstm',
                          bias=True,
                          dropout=0):
        ''' automagically builds[if it does not exist]
            and returns the output of an RNN lazily '''
        if not hasattr(self, 'rnn'):
            self.rnn = self._build_rnn_memory_model(input_size=x.size(-1),
                                                    model_type=model_type,
                                                    bias=bias,
                                                    dropout=dropout)

        return self.rnn(x, state)


    def _build_model(self):
        input_size = int(np.prod(self.input_shape))

        # feature-extracting transformations
        self.phi_x = self._build_phi_x_model()
        self.phi_x_i = []
        self.phi_z = nn.Sequential(
            build_dense_encoder(self.reparameterizer.output_size, self.latent_size,
                                activation_fn=self.dense_activation_fn,
                                normalization_str=self.normalization,
                                nlayers=2),
            nn.SELU()
        )

        # prior
        self.prior = build_dense_encoder(self.latent_size, self.reparameterizer.input_size,
                                         activation_fn=self.dense_activation_fn,
                                         normalization_str=self.normalization,
                                         nlayers=2)

        # decoder
        self.dec = self._build_decoder(input_size=self.latent_size*2,
                                       reupsample=True)

        # memory module that contains the RNN or DNC
        self.memory = VRNNMemory(h_dim=self.latent_size,
                                 n_layers=self.n_layers,
                                 bidirectional=self.bidirectional,
                                 config=self.config,
                                 rnn=self._lazy_rnn_lambda,
                                 cuda=self.config['cuda'])

        # This needs to be done, else weightnorm throws a random error
        # if self.config['cuda']:
        #     modules = [self.phi_x, self.phi_z,
        #                self.enc, self.prior, self.dec]
        #     for i in range(len(modules)):
        #         modules[i] = modules[i].cuda()
        #         if self.config['ngpu'] > 1:
        #             modules[i] = nn.DataParallel(modules[i])

        # initialize our weights
        # self = init_weights(self)

    def get_name(self):
        if self.config['reparam_type'] == "mixture":
            reparam_str = "mixturecat{}gauss{}_".format(
                str(self.config['discrete_size']),
                str(self.config['continuous_size'])
            )
        elif self.config['reparam_type'] == "isotropic_gaussian" or self.config['reparam_type'] == "beta":
            reparam_str = "cont{}_".format(str(self.config['continuous_size']))
        elif self.config['reparam_type'] == "discrete":
            reparam_str = "disc{}_".format(str(self.config['discrete_size']))
        else:
            raise Exception("unknown reparam type")

        return 'vrnn_{}ts_'.format(self.config['max_time_steps']) \
            + super(VRNN, self).get_name(reparam_str)

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

    def _build_rnn_memory_model(self, input_size, model_type='lstm', bias=True, dropout=0):
        if self.config['half']:
            import apex

        model_fn_map = {
            'gru': torch.nn.GRU if not self.config['half'] else apex.RNN.GRU,
            'lstm': torch.nn.LSTM if not self.config['half'] else apex.RNN.LSTM,
        }
        rnn = model_fn_map[model_type](
            input_size=input_size,
            hidden_size=self.latent_size,
            num_layers=self.n_layers,
            bidirectional=self.bidirectional,
            bias=bias, dropout=dropout
        )

        if self.config['cuda'] and not self.config['half']:
            rnn.flatten_parameters()

        return rnn.cuda() if self.config['cuda'] else rnn

    def reparameterize(self, logits_map):
        '''reparameterize the encoder output and the prior'''
        nan_check_and_break(logits_map['encoder_logits'], "enc_logits")
        nan_check_and_break(logits_map['prior_logits'], "prior_logits")
        z_enc_t, params_enc_t = self.reparameterizer(logits_map['encoder_logits'])

        # XXX: hardcode softplus on prior logits std-dev
        feat_size = logits_map['prior_logits'].size(-1)
        logits_map['prior_logits'] = torch.cat(
            [logits_map['prior_logits'][:, 0:feat_size//2],
            F.sigmoid(logits_map['prior_logits'][:, feat_size//2:])],
        -1)
        z_prior_t, params_prior_t = self.reparameterizer(logits_map['prior_logits'])

        z = {  # reparameterization
            'prior': z_prior_t,
            'posterior': z_enc_t,
            'x_features': logits_map['x_features']
        }
        params = {  # params of the posterior
            'prior': params_prior_t,
            'posterior': params_enc_t
        }

        return z, params

    def decode(self, z_t, produce_output=False, reset_state=False):
        # grab state from RNN
        # final_state = self.memory.get_repackaged_state()[0]
        final_state = torch.mean(self.memory.get_state()[0], 0)
        # final_state = self.memory.get_output().squeeze()
        nan_check_and_break(final_state, "final_rnn_output[decode]")

        # feature transform for z_t
        phi_z_t = self.phi_z(z_t['posterior'])
        nan_check_and_break(phi_z_t, "phi_z_t")

        # concat and run through RNN to update state
        input_t = torch.cat([z_t['x_features'], phi_z_t], -1).unsqueeze(0)
        self.memory(input_t.contiguous(), reset_state=reset_state)

        # decode only if flag is set
        dec_t = None
        if produce_output:
            dec_input_t = torch.cat(
                [phi_z_t, final_state], -1
            )#.transpose(1, 0).squeeze()
            dec_t = self.dec(dec_input_t)
            dec_t = self._project_decoder_for_variance(dec_t)

        return dec_t

    def _extract_features(self, x, *xargs):
        ''' accepts x and any number of extra x items and returns
            each of them projected through it's own NN,
            creating any networks as needed
        '''
        phi_x_t = self.phi_x(x)
        for i, x_item in enumerate(xargs):
            if len(self.phi_x_i) < i + 1:
                # add a new model at runtime if needed
                self.phi_x_i.append(self._lazy_build_phi_x(x_item.size()[1:]))
                print("increased length of feature extractors to {}".format(len(self.phi_x_i)))

            # use the model and concat on the feature dimension
            phi_x_i = self.phi_x_i[i](x_item)
            phi_x_t = torch.cat([phi_x_t, phi_x_i], -1)

        nan_check_and_break(phi_x_t, "phi_x_t")
        return phi_x_t

    def _lazy_build_encoder(self, input_size):
        ''' lazy build the encoder based on the input size'''
        if not hasattr(self, 'enc'):
            self.enc = build_dense_encoder(input_size, self.reparameterizer.input_size,
                                           activation_fn=self.dense_activation_fn,
                                           normalization_str=self.normalization,
                                           nlayers=2)

        return self.enc.cuda() if self.config['cuda'] else self.enc

    def encode(self, x, *xargs):
        # get the memory trace
        batch_size = x.size(0)
        final_state = torch.mean(self.memory.get_state()[0], 0)
        #final_state = self.memory.get_output().squeeze()
        #final_state = self.memory.get_repackaged_state()[0]
        nan_check_and_break(final_state, "final_rnn_output")

        # extract input data features
        phi_x_t = self._extract_features(x, *xargs)

        # encoder projection; TODO: evaluate n_layer repeat logic below
        #phi_x_t_expanded = phi_x_t.unsqueeze(0).repeat(self.memory.n_layers, 1, 1)
        enc_input_t = torch.cat(
            [phi_x_t, final_state], -1
        )#.transpose(1, 0).squeeze().contiguous()
        enc_t = self._lazy_build_encoder(enc_input_t.size(-1))(enc_input_t)
        nan_check_and_break(enc_t, "enc_t")

        # prior projection
        prior_t = self.prior(final_state.transpose(1, 0).contiguous())# + eps_fn(self.config['cuda']))
        nan_check_and_break(prior_t, "priot_t")

        # enc_t = self.enc(torch.cat([phi_x_t, final_state], -1).contiguous())
        # nan_check_and_break(enc_t, "enc_t")
        # prior_t = self.prior(final_state)
        # nan_check_and_break(prior_t, "priot_t")


        return {
            'encoder_logits': enc_t,
            'prior_logits': prior_t,
            'x_features': phi_x_t
        }


    def generate_synthetic_samples(self, batch_size, **kwargs):
        assert 'seq_len' in kwargs, "VRNN needs seq_len passed into kwargs"
        samples = zeros([batch_size] + self.config['img_shp'],
                        cuda=self.config['cuda'])

        self.memory.init_state()  # reset memory
        z_t, _ = self.posterior(samples)
        return self.nll_activation(self.decode(z_t))

    def generate(self, z_t, reset_state=False):
        ''' reparameterizer for sequential is different '''
        return self.decode(z_t, reset_state=reset_state)

    def posterior(self, *x_args):
        logits_map = self.encode(*x_args)
        return self.reparameterize(logits_map)

    # def kld(self, dist_list):
    #     ''' KL divergence between dist_a and prior '''
    #     reparams = [self.reparameterizer.kl(kl['posterior'], kl['prior']).unsqueeze(0)
    #                 for kl in dist_list]
    #     return torch.cat(reparams, 0)

    def _ensure_same_size(self, prediction_list, target_list):
        ''' helper to ensure that image sizes in both lists match '''
        assert len(prediction_list) == len(target_list), "#preds[{}] != #targets[{}]".format(
            len(prediction_list), len(target_list))
        for i in range(len(target_list)):
            if prediction_list[i].size() != target_list[i].size():
                if prediction_list[i].size() > target_list[i].size():
                    larger_size = prediction_list[i].size()
                    target_list[i] = F.upsample(target_list[i],
                                                size=tuple(larger_size[2:]),
                                                mode='bilinear')

                else:
                    larger_size = target_list[i].size()
                    prediction_list[i] = F.upsample(prediction_list[i],
                                                    size=tuple(larger_size[2:]),
                                                    mode='bilinear')

        return prediction_list, target_list

    def kld(self, dist):
        ''' KL divergence between dist_a and prior as well as constrain prior to hyper-prior'''
        return self.reparameterizer.kl(dist['posterior'], dist['prior'])  # \
             # + self.reparameterizer.kl(dist['prior']) / self.config['max_time_steps']

    # def nll(self, prediction_list, target_list):
    #     prediction_list, target_list = self._ensure_same_size(prediction_list, target_list)
    #     nll = [nll_fn(targets, predictions, self.config['nll_type']).unsqueeze(0)
    #            for targets, predictions in zip(target_list, prediction_list)]
    #     return torch.cat(nll, 0)

    def mut_info(self, dist_params_container):
        ''' helper to get mutual info '''
        mut_info = None
        if (self.config['continuous_mut_info'] > 0
            or self.config['discrete_mut_info'] > 0):
            # only grab the mut-info if the scalars above are set
            mut_info = [self.reparameterizer.mutual_info(params['posterior']).unsqueeze(0)
                        for params in dist_params_container]
            mut_info = torch.sum(torch.cat(mut_info, 0), 0)

        return mut_info

    @staticmethod
    def _add_loss_map(loss_t, loss_aggregate_map):
        ''' helper to add two maps and keep counts
            of the total samples for reduction later'''
        if loss_aggregate_map is None:
            return {**loss_t, 'count': 1}

        for (k, v) in loss_t.items():
            loss_aggregate_map[k] += v

        # increment total count
        loss_aggregate_map['count'] += 1
        return loss_aggregate_map

    @staticmethod
    def _mean_map(loss_aggregate_map):
        ''' helper to reduce all values by the key count '''
        for k in loss_aggregate_map.keys():
            loss_aggregate_map[k] /= loss_aggregate_map['count']

        return loss_aggregate_map

    def loss_function(self, recon_x_container, x_container, params_map):
        ''' evaluates the loss of the model '''
        loss_aggregate_map = None
        for recon_x, x, params in zip(recon_x_container, x_container, params_map):
            mut_info_t = self.mut_info(params)
            loss_t = super(VRNN, self).loss_function(recon_x, x, params,
                                                     mut_info=mut_info_t)
            loss_aggregate_map = self._add_loss_map(loss_t, loss_aggregate_map)

        return self._mean_map(loss_aggregate_map)

    # def loss_function(self, recon_x, x, params_container):
    #     # elbo = -log_likelihood + latent_kl
    #     # cost = elbo + consistency_kl - self.mutual_info_reg * mutual_info_regularizer
    #     nll = self.nll(recon_x, x) # multiply by self.config['max_time_steps']?
    #     nan_check_and_break(nll, "nll")

    #     kld = self.config['kl_reg'] * self.kld(params_container)
    #     nan_check_and_break(kld, "kld")

    #     #elbo = torch.sum(nll + kld, 0)
    #     elbo = torch.mean(nll + kld, 0)

    #     # tabulate mutual information
    #     mut_info = self.mut_info(params_container)

    #     # handle the mutual information term
    #     if mut_info is None:
    #         mut_info = same_type(self.config['half'], self.config['cuda'])(
    #             x[0].size(0)
    #         ).zero_().requires_grad_()
    #     else:
    #         # Clamping strategies
    #         mut_clamp_strategy_map = {
    #             'none': lambda mut_info: mut_info,
    #             'norm': lambda mut_info: mut_info / torch.norm(mut_info, p=2),
    #             'clamp': lambda mut_info: torch.clamp(mut_info,
    #                                                   min=-self.config['mut_clamp_value'],
    #                                                   max=self.config['mut_clamp_value'])
    #         }
    #         mut_info = mut_clamp_strategy_map[self.config['mut_clamp_strategy'].strip().lower()](mut_info)

    #     nan_check_and_break(mut_info, "mut_info")
    #     loss = elbo - mut_info
    #     nan_check_and_break(loss, "vrnn_loss")
    #     return {
    #         'loss': loss,
    #         'loss_mean': torch.mean(loss),
    #         'elbo_mean': torch.mean(elbo),
    #         'nll_mean': torch.mean(nll),
    #         'kld_mean': torch.mean(kld),
    #         'mut_info_mean': torch.mean(mut_info)
    #     }
