# coding: utf-8

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from torch.autograd import Variable

from helpers.utils import zeros_like, ones_like, same_type
from helpers.utils import eps as eps_fn


class Beta(nn.Module):
    ''' beta reparameterization '''
    def __init__(self, config):
        super(Beta, self).__init__()
        self.config = config
        self.input_size = self.config['continuous_size']
        assert self.config['continuous_size'] % 2 == 0
        self.output_size = self.config['continuous_size'] // 2

    def prior(self, batch_size):
        ''' Kerman, J. (2011). Neutral noninformative and informative
        conjugate beta and gamma prior distributions. Electronic
        Journal of Statistics, 5, 1450-1470.'''
        conc1 = Variable(
            same_type(self.config['half'], self.config['cuda'])(
                batch_size, self.output_size
            ).zero_() + 1/3
        )
        conc2 = Variable(
            same_type(self.config['half'], self.config['cuda'])(
                batch_size, self.output_size
            ).zero_() + 1/3
        )
        return D.Beta(conc1, conc2).sample()

    def _reparametrize_beta(self, conc1, conc2):
        if self.training:
            # rsample is CPU only ¯\_(ツ)_/¯ , but ok
            # because it is outside the computational graph
            beta_cpu = D.Beta(conc1.cpu(), conc2.cpu()).rsample()
            beta = beta_cpu.cuda() if self.config['cuda'] else beta_cpu
            # once we have GPU rsample just use the following:
            # beta = D.Beta(conc1, conc2).rsample()
            return beta, {'conc1': conc1, 'conc2': conc2}

        return D.Beta(conc1, conc2).mean, {'conc1': conc1, 'conc2': conc2}

    def reparmeterize(self, logits):
        eps = eps_fn(self.config['half'])
        feature_size = logits.size(-1)
        assert feature_size % 2 == 0 and feature_size // 2 == self.output_size
        if logits.dim() == 2:
            conc1 = F.sigmoid(logits[:, 0:int(feature_size/2)] + eps)
            conc2 = F.sigmoid(logits[:, int(feature_size/2):] + eps)
        elif logits.dim() == 3:
            conc1 = F.sigmoid(logits[:, :, 0:int(feature_size/2)] + eps)
            conc2 = F.sigmoid(logits[:, :, int(feature_size/2):] + eps)
        else:
            raise Exception("unknown number of dims for isotropic gauss reparam")

        return self._reparametrize_beta(conc1, conc2)

    def _kld_beta_kerman_prior(self, conc1, conc2):
        prior = D.Beta(zeros_like(conc1) + 1/3,
                       zeros_like(conc2) + 1/3)
        beta = D.Beta(conc1, conc2)
        return torch.sum(D.kl_divergence(beta, prior), -1)

    def kl(self, dist_a, prior=None):
        if prior == None:  # use standard reparamterizer
            return self._kld_beta_kerman_prior(
                dist_a['beta']['conc1'], dist_a['beta']['conc2']
            )

        # we have two distributions provided (eg: VRNN)
        return torch.sum(D.kl_divergence(
            D.Beta(dist_a['beta']['conc1'], dist_a['beta']['conc2']),
            D.Beta(prior['beta']['conc1'], prior['beta']['conc2'])
        ), -1)


    def mutual_info(self, params, eps=1e-9):
        # I(z_d; x) ~ H(z_prior, z_d) + H(z_prior)
        z_true = D.Beta(params['beta']['conc1'],
                        params['beta']['conc2'])
        z_match = D.Beta(params['q_z_given_xhat']['beta']['conc1'],
                         params['q_z_given_xhat']['beta']['conc2'])
        kl_proxy_to_xent = torch.sum(D.kl_divergence(z_match, z_true), dim=-1)
        return  kl_proxy_to_xent

    def log_likelihood(self, z, params):
        return D.Beta(params['beta']['conc1'],
                      params['beta']['conc2']).log_prob(z)

    def forward(self, logits):
        z, beta_params = self.reparmeterize(logits)
        beta_params['conc1_mean'] = torch.mean(beta_params['conc1'])
        beta_params['conc2_mean'] = torch.mean(beta_params['conc2'])
        return z, { 'z': z, 'beta':  beta_params }
