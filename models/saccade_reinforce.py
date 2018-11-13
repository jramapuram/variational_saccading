import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from copy import deepcopy

from helpers.utils import expand_dims, check_or_create_dir, \
    zeros_like, int_type, nan_check_and_break, zeros, get_dtype, \
    plot_tensor_grid, add_noise_to_imgs
from helpers.layers import View, Identity, get_encoder, str_to_activ_module
from .image_state_projector import ImageStateProjector
from .numerical_diff_module import NumericalDifferentiator
from .spatial_transformer import SpatialTransformer
from datasets.crop_dual_imagefolder import CropLambdaPool, USE_LIB

from saccade import Saccader


class SaccaderReinforce(Saccader):
    def __init__(self, vae, output_size, **kwargs):
        Saccader.__init__(self, vae, output_size, **kwargs)

    def forward(self, x, x_related):
            ''' encode with VAE, then use posterior sample to
                gather image from original space (eg: 4k image) '''
            batch_size, chans = x_related.size(0), x_related.size(1)

            def _forward_internal(x_related, inference_only=False):
                params, crops, crops_true, inlays, decodes = [], [], [], [], []

                # reset the state, output and the truncate window
                self.vae.memory.init_state(batch_size, cuda=x_related.is_cuda)
                self.vae.memory.init_output(batch_size, seqlen=1, cuda=x_related.is_cuda)

                # accumulator for predictions and ACT
                x_preds = zeros((batch_size, self.config['latent_size']),
                                cuda=x_related.is_cuda,
                                dtype=get_dtype(x_related)).requires_grad_()
                act = zeros((batch_size, 1), cuda=x_related.is_cuda,
                            dtype=get_dtype(x_related)).squeeze().requires_grad_()

                for i in range(self.config['max_time_steps']):
                    # get posterior and params, expand 0'th dim for seqlen
                    x_related_inference = add_noise_to_imgs(x_related) \
                        if self.config['add_img_noise'] else x_related

                    z_t, params_t = self.vae.posterior(x_related_inference)

                    # call forward baseline here
                    base_score = self.vae.get_baseline()

                    # locator forward
                    mu, l_t = self.vae.locator()

                    nan_check_and_break(x_related_inference, "x_related_inference")
                    nan_check_and_break(z_t['prior'], "prior")
                    nan_check_and_break(z_t['posterior'], "posterior")
                    nan_check_and_break(z_t['x_features'], "x_features")

                    # to modify, new locator/cropper here
                    # extract the required crop from original image

                    x_trunc_t = self._z_to_image(z_t['posterior'], x)

                    # to modify
                    # do preds and sum

                    state = torch.mean(self.vae.memory.get_state()[0], 0)
                    
                    crops_pred_perturbed = add_noise_to_imgs(x_trunc_t['crops_pred']) \
                        if self.config['add_img_noise'] else x_trunc_t['crops_pred']
                    state_proj = self.latent_projector(crops_pred_perturbed, state)
                    x_preds = x_preds + state_proj[:, 0:-1]  # last bit is for ACT

                    # decode the posterior
                    decoded_t = self.vae.decode(z_t, produce_output=True)
                    nan_check_and_break(decoded_t, "decoded_t")

                    # cache for loss function & visualization
                    params.append(params_t)
                    crops.append(x_trunc_t['crops_pred'])
                    decodes.append(decoded_t)

                    # only add these if we are in the lambda setting:
                    if 'crops_true' in x_trunc_t and 'inlay' in x_trunc_t:
                        crops_true.append(x_trunc_t['crops_true'])
                        inlays.append(x_trunc_t['inlay'])

                    # conditionally break away based on ACT
                    # act = act + torch.sigmoid(state_proj[:, -1])
                    # if torch.max(act / max(i, 1)) >= 0.9998:
                    #     break

                preds = self.latent_projector.get_output(
                    x_preds / self.config['max_time_steps']
                )
                return {
                    'act': act / max(i, 1),
                    'saccades_scalar': i,
                    'decoded': decodes,
                    'params': params,
                    'preds': preds,
                    'inlays': inlays,
                    'crops': crops,
                    'crops_true': crops_true
                }

            # forward pass with decoding [normal]
            standard_forward_pass = _forward_internal(x_related, inference_only=False)

            #  re-eval posterior for mut-info using decoded from above
            if (self.config['continuous_mut_info'] > 0
                    or self.config['discrete_mut_info'] > 0):
                ''' append the posterior of re-forward passing'''
                q_z_given_x_hat = _forward_internal(standard_forward_pass['decoded'],
                                                    inference_only=True)
                for param, q_z_given_xhat_param in zip(standard_forward_pass['params'],
                                                       q_z_given_x_hat['params']):
                    param['posterior'] = {
                        **param['posterior'],
                        'q_z_given_xhat': q_z_given_xhat_param['posterior']
                    }

            # return both the standard forward pass and the mut-info one
            # after clearing the cached memory
            self.vae.memory.clear()
            return standard_forward_pass



    # Reinforce loss
    def loss_function(self, x, labels, output_map):

        # build and call module here
        # build custom module

        # Predicted class
        loss_fn = F.binary_cross_entropy_with_logits \
            if len(labels.shape) > 1 else F.cross_entropy
        pred_loss = loss_fn(
            input=output_map['preds'],
            target=labels,
            reduction='none'
        )
        pred_loss = torch.sum(pred_loss, -1) if len(pred_loss.shape) > 1 else pred_loss

        # Possibly not correct
        predicted = output_map['preds']

        # Log_pi depend on hidden state, log_pi = log N (mu, std)
        log_pi =

        # Baselines
        baselines = 

        # Calculate reward
        R = (predicted.detach() == labels).float()
        # Possible unroll

        # Losses for differential modules
        loss_actions = vae_loss_map['nll_mean']
        # Baseline compensation
        loss_baseline = F.mse_loss(baselines, R)

        # Compute reinforce loss, sum over time-steps and average the batch
        adjusted_reward = R - baselines.detach()
        loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
        loss_reinforce = torch.mean(loss_reinforce, dim=0)

        return loss_reinforce
