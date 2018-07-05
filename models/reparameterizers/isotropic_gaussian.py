from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from torch.autograd import Variable

from helpers.utils import zeros_like, ones_like, same_type, float_type, nan_check_and_break
from helpers.utils import eps as eps_fn


class IsotropicGaussian(nn.Module):
    ''' isotropic gaussian reparameterization '''
    def __init__(self, config):
        super(IsotropicGaussian, self).__init__()
        self.config = config
        self.input_size = self.config['continuous_size']
        assert self.config['continuous_size'] % 2 == 0
        self.output_size = self.config['continuous_size'] // 2

    def prior(self, batch_size):
        return Variable(
            same_type(self.config['half'], self.config['cuda'])(
                batch_size, self.output_size
            ).normal_()
        )

    def _reparametrize_gaussian(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp()
            eps = float_type(self.config['cuda'])(std.size()).normal_().type(
                    same_type(self.config['half'], self.config['cuda'])
                )
            eps = Variable(eps)
            nan_check_and_break(logvar, "logvar")
            return eps.mul(std).add_(mu), {'mu': mu, 'logvar': logvar}

        return mu, {'mu': mu, 'logvar': logvar}

    def reparmeterize(self, logits):
        eps = eps_fn(self.config['half'])
        feature_size = logits.size(-1)
        assert feature_size % 2 == 0 and feature_size // 2 == self.output_size
        if logits.dim() == 2:
            mu = logits[:, 0:int(feature_size/2)]
            nan_check_and_break(mu, "mu")
            sigma = logits[:, int(feature_size/2):] + eps
            # sigma = F.softplus(logits[:, int(feature_size/2):]) + eps
            # sigma = F.hardtanh(logits[:, int(feature_size/2):], min_val=-6.,max_val=2.)
        elif logits.dim() == 3:
            mu = logits[:, :, 0:int(feature_size/2)]
            sigma = logits[:, :, int(feature_size/2):] + eps
        else:
            raise Exception("unknown number of dims for isotropic gauss reparam")

        return self._reparametrize_gaussian(mu, sigma)

    def mutual_info(self, params, eps=1e-9):
        # I(z_d; x) ~ H(z_prior, z_d) + H(z_prior)
        z_true = D.Normal(params['gaussian']['mu'],
                          params['gaussian']['logvar'])
        z_match = D.Normal(params['q_z_given_xhat']['gaussian']['mu'],
                           params['q_z_given_xhat']['gaussian']['logvar'])
        kl_proxy_to_xent = torch.sum(D.kl_divergence(z_match, z_true), dim=-1)
        return kl_proxy_to_xent

    @staticmethod
    def _kld_gaussian_N_0_1(mu, logvar):
        standard_normal = D.Normal(zeros_like(mu), ones_like(logvar))
        normal = D.Normal(mu, logvar)
        return torch.sum(D.kl_divergence(normal, standard_normal), -1)

    def kl(self, dist_a, prior=None):
        if prior == None: # use default prior
            return IsotropicGaussian._kld_gaussian_N_0_1(
                dist_a['gaussian']['mu'], dist_a['gaussian']['logvar']
            )

        # we have two distributions provided (eg: VRNN)
        return torch.sum(D.kl_divergence(
            D.Normal(dist_a['gaussian']['mu'], dist_a['gaussian']['logvar']),
            D.Normal(prior['gaussian']['mu'], prior['gaussian']['logvar'])
        ), -1)

    # def kl(self, dist_a, prior=None):
    #     if prior == None: # use default prior
    #         return IsotropicGaussian._kld_gaussian_N_0_1(
    #             F.sigmoid(dist_a['gaussian']['mu']), F.softplus(dist_a['gaussian']['logvar'])
    #         )

    #     # we have two distributions provided (eg: VRNN)
    #     return D.kl_divergence(
    #         D.Normal(F.sigmoid(dist_a['gaussian']['mu']), F.softplus(dist_a['gaussian']['logvar'])),
    #         D.Normal(F.sigmoid(prior['gaussian']['mu']), F.softplus(prior['gaussian']['logvar']))
    #     )

    def log_likelihood(self, z, params):
        return D.Normal(params['gaussian']['mu'],
                        params['gaussian']['logvar']).log_prob(z)

    def forward(self, logits):
        z, gauss_params = self.reparmeterize(logits)
        gauss_params['mu_mean'] = torch.mean(gauss_params['mu'])
        gauss_params['logvar_mean'] = torch.mean(gauss_params['logvar'])
        return z, { 'z': z, 'gaussian':  gauss_params }
