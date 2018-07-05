from __future__ import print_function
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable

from helpers.utils import float_type, one_hot, ones_like


class GumbelSoftmax(nn.Module):
    def __init__(self, config, dim=-1):
        super(GumbelSoftmax, self).__init__()
        self._setup_anneal_params()
        self.dim = dim
        self.iteration = 0
        self.config = config
        self.input_size = self.config['discrete_size']
        self.output_size = self.config['discrete_size']

    def prior(self, batch_size):
        uniform_probs = float_type(self.config['cuda'])(1, self.output_size[-1]).zero_()
        uniform_probs += 1.0 / self.output_size
        cat = torch.distributions.Categorical(uniform_probs)
        sample = cat.sample((batch_size*self.output_size[0],))
        return Variable(
            one_hot(self.output_size[-1], sample, use_cuda=self.config['cuda'])
        ).type(float_type(self.config['cuda']))

    def _setup_anneal_params(self):
        # setup the base gumbel rates
        # TODO: parameterize this
        self.tau, self.tau0 = 1.0, 1.0
        self.anneal_rate = 3e-6
        self.min_temp = 0.5

    def anneal(self, anneal_interval=10):
        ''' Helper to anneal the categorical distribution'''
        if self.training \
           and self.iteration > 0 \
           and self.iteration % anneal_interval == 0:

            # smoother annealing
            rate = -self.anneal_rate * self.iteration
            self.tau = np.maximum(self.tau0 * np.exp(rate),
                                  self.min_temp)
            # hard annealing
            # self.tau = np.maximum(0.9 * self.tau, self.min_temp)

    def reparmeterize(self, logits):
        logits_shp = logits.size()
        log_q_z = F.log_softmax(logits, dim=self.dim)
        z, z_hard = self.sample_gumbel(logits, self.tau,
                                       hard=True,
                                       dim=self.dim,
                                       use_cuda=self.config['cuda'])
        return z.view(logits_shp), z_hard.view(logits_shp), log_q_z

    def mutual_info_analytic(self, params, eps=1e-9):
        # I(z_d; x) ~ H(z_prior, z_d) + H(z_prior)
        targets = torch.argmax(params['discrete']['z_hard'].type(long_type(self.config['cuda'])), dim=-1)
        crossent_loss = -F.cross_entropy(input=params['q_z_given_xhat']['discrete']['logits'],
                                         target=targets, reduce=False)
        ent_loss = -torch.sum(D.OneHotCategorical(logits=params['discrete']['z_hard']).entropy(), -1)
        return ent_loss + crossent_loss

    def mutual_info_monte_carlo(self, params, eps=1e-9):
        # I(z_d; x) ~ H(z_prior, z_d) + H(z_prior)
        log_q_z_given_x = params['q_z_given_xhat']['discrete']['logits'] + eps
        # log_q_z_given_x = params['discrete']['log_q_z'] + eps
        p_z = self.prior(log_q_z_given_x.size()[0])
        # p_z = params['discrete']['z_soft'] + eps
        crossent_loss = -torch.sum(log_q_z_given_x * p_z, dim=-1)
        ent_loss = -torch.sum(torch.log(p_z + eps) * p_z, dim=-1)
        return ent_loss + crossent_loss

    def mutual_info(self, params, eps=1e-9):
        ''' analytic mutual information regularizer'''
        targets = torch.argmax(params['discrete']['z_hard'].type(long_type(self.config['cuda'])), dim=-1)
        # soft_targets = F.softmax(
        #     params['discrete']['logits'], -1
        # ).type(long_type(self.config['cuda']))
        # targets = torch.argmax(params['discrete']['log_q_z'], -1) # 3rd change, havent tried
        crossent_loss = -F.cross_entropy(input=params['q_z_given_xhat']['discrete']['logits'],
                                         target=targets, reduce=False)
        ent_loss = -torch.sum(D.OneHotCategorical(logits=params['discrete']['z_hard']).entropy(), -1)
        return ent_loss + crossent_loss

    @staticmethod
    def _kld_categorical_uniform(log_q_z, dim=-1, eps=1e-9):
        shp = log_q_z.size()
        p_z = 1.0 / shp[dim]
        log_p_z = np.log(p_z)
        kld_element = log_q_z.exp() * (log_q_z - log_p_z)
        #return torch.sum(kld_element.view(shp[0], -1), -1)
        return kld_element

    def kl(self, dist_a, prior=None):
        if prior == None:  # use standard uniform prior
            return GumbelSoftmax._kld_categorical_uniform(
                dist_a['discrete']['log_q_z'], dim=self.dim
            )

        # we have two distributions provided (eg: VRNN)
        return D.kl_divergence(
            D.OneHotCategorical(logits=dist_a['discrete']['log_q_z']),
            D.OneHotCategorical(prior['discrete']['log_q_z'])
        )


    @staticmethod
    def _gumbel_softmax(x, tau, eps=1e-9, dim=-1, use_cuda=False):
        noise = torch.rand(x.size())
        # -ln(-ln(U + eps) + eps)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        if use_cuda:
            noise = noise.cuda()

        noise = Variable(noise)
        x = (x + noise) / tau
        x = F.softmax(x + eps, dim=dim)
        return x.view_as(x)

    @staticmethod
    def sample_gumbel(x, tau, hard=False, dim=-1, use_cuda=True):
        y = GumbelSoftmax._gumbel_softmax(x, tau, dim=dim, use_cuda=use_cuda)

        if hard:
            y_max, _ = torch.max(y, dim=dim, keepdim=True)
            y_hard = Variable(
                torch.eq(y_max.data, y.data).type(float_type(use_cuda))
            )
            y_hard_diff = y_hard - y
            y_hard = y_hard_diff.detach() + y
            return y.view_as(x), y_hard.view_as(x)

        return y.view_as(x), None

    def log_likelihood(self, z, params):
        print("log = ", params['discrete']['logits'].size(), " | z = ", z.size())
        return D.Categorical(logits=params['discrete']['logits']).log_prob(z)

    def forward(self, logits):
        self.anneal()  # anneal first
        z, z_hard, log_q_z = self.reparmeterize(logits)
        params = {
            'z_hard': z_hard,
            'logits': logits,
            'log_q_z': log_q_z,
            'tau_scalar': self.tau
        }
        self.iteration += 1

        if self.training:
            # return the reparameterization
            # and the params of gumbel
            return z, { 'z': z, 'discrete': params }

        return z_hard, { 'z': z, 'discrete': params }
