import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributions as D

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

            locs = []
            log_pi = []
            baselines = []

            for i in range(self.config['max_time_steps']):
                # get posterior and params, expand 0'th dim for seqlen
                x_related_inference = add_noise_to_imgs(x_related) \
                    if self.config['add_img_noise'] else x_related

                z_t, params_t = self.vae.posterior(x_related_inference)

                # TODO:
                # check locator function
                # check std param
                # locator forward
                mu, l_t = self.vae.locator()

                p = D.Normal(mu, self.std).log_prob(l_t)
                p = torch.sum(p, dim=1)

                # foveate / cropper code here

                nan_check_and_break(x_related_inference, "x_related_inference")
                nan_check_and_break(z_t['prior'], "prior")
                nan_check_and_break(z_t['posterior'], "posterior")
                nan_check_and_break(z_t['x_features'], "x_features")

                # hard cropper
                # x_trunc_t = self._z_to_image(z_t['posterior'], x)
                x_trunc_t = self._hard_crop(self, x, l_t, self.config['window_size'])

                # call forward baseline here
                base_score = self.vae.get_baseline()

                # to modify
                # do preds and sum

                state = torch.mean(self.vae.memory.get_state()[0], 0)

                crops_pred_perturbed = add_noise_to_imgs(x_trunc_t) \
                    if self.config['add_img_noise'] else x_trunc_t

                state_proj = self.latent_projector(crops_pred_perturbed, state)
                x_preds = x_preds + state_proj[:, 0:-1]  # last bit is for ACT

                # decode the posterior
                decoded_t = self.vae.decode(z_t, produce_output=True)
                nan_check_and_break(decoded_t, "decoded_t")

                # cache for loss function & visualization
                params.append(params_t)
                crops.append(x_trunc_t)
                decodes.append(decoded_t)

                # conditionally break away based on ACT
                # act = act + torch.sigmoid(state_proj[:, -1])
                # if torch.max(act / max(i, 1)) >= 0.9998:
                #     break

                # Keep locations and baselines per time-step

                locs.append(l_t)
                log_pi.append(p)
                baselines.append(base_score)

            # After last time step
            preds = self.latent_projector.get_output(
                x_preds / self.config['max_time_steps']
            )

            return {
                'location': locs,
                'log_pi': log_pi,
                'baselines': baselines,
                'act': act / max(i, 1),
                'saccades_scalar': i,
                'decoded': decodes,
                'params': params,
                'preds': preds,
                'crops': x_trunc_t,
                'crops_true': crops_true
            }

        # forward pass with decoding [normal]
        standard_forward_pass = _forward_internal(x_related, inference_only=False)

        #  re-eval posterior for mut-info using decoded from above
        if (self.config['continuous_mut_info'] > 0 or
                self.config['discrete_mut_info'] > 0):
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

        # Need
        # 1)
        # soft predictions, log_probas
        # 2)
        # Log_pi depend on hidden state, log_pi = log N (mu, std)
        # log_pi = Normal(mu, self.std).log_prob(l_t)
        # log_pi = torch.sum(log_pi, dim=1)
        # log_pi =
        # 3) predicted
        # Hard class prediction
        # 4) baselines

        log_probas = output_map['preds']
        predicted = torch.max(log_probas, 1)[1]

        log_pi = output_map['log_pi']
        log_pi = torch.stack(log_pi).transpose(1, 0)

        baselines = torch.stack(baselines).transpose(1, 0)

        # Calculate reward, needs possible unroll
        R = (predicted.detach() == labels).float()

        # Losses for differential modules
        loss_actions = vae_loss_map['nll_mean']
        # Baseline compensation
        loss_baseline = F.mse_loss(baselines, R)

        # Compute reinforce loss, sum over time-steps and average the batch
        adjusted_reward = R - baselines.detach()
        loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
        loss_reinforce = torch.mean(loss_reinforce, dim=0)

        return loss_reinforce

    def _hard_crop(self, x, l, size):
        """
        Modified code from:
        https://github.com/kevinzakka/recurrent-visual-attention

        Extract a single patch for each image in the
        minibatch `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l: a 2D Tensor of shape (B, 2).
        - size: a scalar defining the size of the extracted patch.

        Returns
        -------
        - patch: a 4D Tensor of shape (B, size, size, C)
        """
        B, C, H, W = x.shape

        # denormalize coords of patch center
        coords = self._denormalize(H, l)

        # compute top left corner of patch
        patch_x = coords[:, 0] - (size // 2)
        patch_y = coords[:, 1] - (size // 2)

        # loop through mini-batch and extract
        patch = []
        for i in range(B):
            im = x[i].unsqueeze(dim=0)
            T = im.shape[-1]

            # compute slice indices
            from_x, to_x = patch_x[i], patch_x[i] + size
            from_y, to_y = patch_y[i], patch_y[i] + size

            # cast to ints
            from_x, to_x = from_x.item(), to_x.item()
            from_y, to_y = from_y.item(), to_y.item()

            # pad tensor in case exceeds
            if self.exceeds(from_x, to_x, from_y, to_y, T):
                pad_dims = (
                    size // 2 + 1, size // 2 + 1,
                    size // 2 + 1, size // 2 + 1,
                    0, 0,
                    0, 0,
                )
                im = F.pad(im, pad_dims, "constant", 0)

                # add correction factor
                from_x += (size // 2 + 1)
                to_x += (size // 2 + 1)
                from_y += (size // 2 + 1)
                to_y += (size // 2 + 1)

            # and finally extract
            patch.append(im[:, :, from_y:to_y, from_x:to_x])

        # concatenate into a single tensor
        patch = torch.cat(patch)

        return patch

    def _denormalize(self, T, coords):
        """
        Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
        return (0.5 * ((coords + 1.0) * T)).long()

    def _exceeds(self, from_x, to_x, from_y, to_y, T):
        """
        Check whether the extracted patch will exceed
        the boundaries of the image of size `T`.
        """
        if (
            (from_x < 0) or (from_y < 0) or (to_x > T) or (to_y > T)
        ):
            return True
        return False
