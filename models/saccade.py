import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from copy import deepcopy
from torch.autograd import Variable

from helpers.utils import expand_dims, check_or_create_dir, \
    zeros_like, int_type, nan_check_and_break, zeros, get_dtype
from helpers.layers import View
from .relational_network import RelationalNetwork
from .image_state_projector import ImageStateProjector
from datasets.crop_dual_imagefolder import CropLambdaPool

# the folowing few helpers are from pyro example for AIR
def ng_ones(*args, **kwargs):
    """
    :param torch.Tensor type_as: optional argument for tensor type

    A convenience function for Variable(torch.ones(...), requires_grad=False)
    """
    retype = kwargs.pop('type_as', None)
    p_tensor = torch.ones(*args, **kwargs)
    return Variable(p_tensor if retype is None else p_tensor.type_as(retype), requires_grad=False)


def ng_zeros(*args, **kwargs):
    """
    :param torch.Tensor type_as: optional argument for tensor type

    A convenience function for Variable(torch.ones(...), requires_grad=False)
    """
    retype = kwargs.pop('type_as', None)
    p_tensor = torch.zeros(*args, **kwargs)
    return Variable(p_tensor if retype is None else p_tensor.type_as(retype), requires_grad=False)


def expand_z_where(z_where):
    # Take a batch of three-vectors, and massages them into a batch of
    # 2x3 matrices with elements like so:
    # [s,x,y] -> [[s,0,x],
    #             [0,s,y]]
    n = z_where.size(0)
    out = torch.cat((ng_zeros([1, 1]).type_as(z_where).expand(n, 1), z_where), 1)
    expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])
    ix = Variable(expansion_indices)
    if z_where.is_cuda:
        ix = ix.cuda()

    out = torch.index_select(out, 1, ix)
    out = out.view(n, 2, 3)
    return out


def z_where_inv(z_where, clip_scale=5.0):
    # Take a batch of z_where vectors, and compute their "inverse".
    # That is, for each row compute:
    # [s,x,y] -> [1/s,-x/s,-y/s]
    # These are the parameters required to perform the inverse of the
    # spatial transform performed in the generative model.
    n = z_where.size(0)
    out = torch.cat((ng_ones([1, 1]).type_as(z_where).expand(n, 1),
                     -z_where[:, 1:]), 1)

    # Divide all entries by the scale. abs(scale) ensures images arent flipped
    scale = torch.max(torch.abs(z_where[:, 0:1]),
                      zeros_like(z_where[:, 0:1]) + clip_scale)
    if torch.sum(scale == 0) > 0:
        print("tensor scale of {} dim was 0!!".format(scale.shape))
        exit(-1)

    nan_check_and_break(scale, "scale")
    out = out / scale
    # out = out / z_where[:, 0:1]

    return out

def window_to_image(z_where, window_size, image_size, windows):
    n = windows.size(0)
    assert windows.size(1) == window_size ** 2, 'Size mismatch.'
    theta = expand_z_where(z_where)
    grid = F.affine_grid(theta, torch.Size((n, 1, image_size, image_size)))
    out = F.grid_sample(windows.view(n, 1, window_size, window_size), grid)
    return out.view(n, 1, image_size, image_size)


def image_to_window(z_where, window_size, images, max_image_percentage=0.15):
    ''' max_percentage is the maximum scale possible for the window

        example sizes:
            grid=  torch.Size([300, 32, 32, 2])  | images =  torch.Size([300, 1, 64, 64])
            theta_inv =  torch.Size([300, 2, 3])
            nonzero grid =  tensor(0, device='cuda:0')
    '''

    n, image_size = images.size(0), list(images.size())[1:]
    assert images.size(2) == images.size(3) == image_size[1] == image_size[2], 'Size mismatch.'
    max_scale = image_size[1] / (image_size[1] * max_image_percentage)
    theta_inv = expand_z_where(z_where_inv(z_where, clip_scale=max_scale))
    grid = F.affine_grid(theta_inv, torch.Size((n, 1, window_size, window_size)))
    out = F.grid_sample(images.view(n, *image_size), grid)
    return out

def image_to_window_discrete(z_where, window_size, image_size, images, scale=None):
    n = images.size(0)
    assert images.size(2) == images.size(3) == image_size[1] == image_size[2], 'Size mismatch.'
    x_where, y_where = (z_where > 0).nonzero()
    if scale: # scale the window size
        window_size *= scale

    # tabulate upper left and bottom right positions
    window_upper_left = [x_where, y_where] - (window_size // 2)
    window_lower_right = [x_where, y_where] + (window_size // 2)
    assert window_upper_left > [0, 0]
    assert window_lower_right < [images.size(2), images.size(3)]

    # index out
    return images[:, :, window_upper_left[0]:window_lower_right[0], # x
                  window_upper_left[1]:window_lower_right[1]]       # y


class Saccader(nn.Module):
    def __init__(self, vae, output_size, **kwargs):
        super(Saccader, self).__init__()
        self.vae = vae
        self.output_size = output_size
        self.config = kwargs['kwargs']

        # build the pool object
        self.pool = CropLambdaPool(self.config['batch_size'])

        # build the projection to softmax from RNN state
        # self.latent_projector = ImageStateProjector(output_size=self.output_size,
        #                                             config=self.config)
        config_copy = deepcopy(self.config)
        config_copy['latent_size'] = self.output_size
        self.latent_projector = ImageStateProjector(output_size=self.output_size,
                                                    config=config_copy)

    def parallel(self):
        self.latent_projector.parallel()
        self.vae.parallel()

    def get_name(self):
        return "{}_win{}_us{}_dscale{}_{}".format(
            str(self.config['uid']),
            str(self.config['window_size']),
            str(self.config['synthetic_upsample_size']) if self.config['synthetic_upsample_size'] > 0 else "",
            str(self.config['downsample_scale']),
            self.vae.get_name()
        )

    def load(self, filename=None):
        def _load(model_filename):
            # load the pool if it exists
            if os.path.isfile(model_filename):
                print("loading existing saccader model from {}".format(
                    model_filename
                ))
                self = torch.load(model_filename)
                return True

            return False

        if filename is None:
            assert os.path.isdir(".models")
            model_filename = os.path.join(".models", self.get_name() + ".th")
            return _load(model_filename)

        return _load(filename)

    def save(self, overwrite=False):
        # save the model if it doesnt exist
        check_or_create_dir(".models")
        model_filename = os.path.join(".models", self.get_name() + ".th")
        if not os.path.isfile(model_filename) or overwrite:
            print("saving existing saccader model")
            torch.save(self, model_filename)

    def loss_function(self, x, labels, output_map):
        ''' loss is: L_{classifier} * L_{VAE} '''
        vae_loss_map = self.vae.loss_function(output_map['decoded'],
                                              #[x] + output_map['crops'],
                                              #[x] + output_map['crops'][0:-1],
                                              [x.clone() for _ in range(len(output_map['decoded']))],
                                              output_map['params'])

        # get classifier loss
        pred_loss = F.cross_entropy(
            input=output_map['preds'],
            target=labels,
            reduce=False
        )
        nan_check_and_break(pred_loss, "pred_loss")

        # TODO: try multi-task loss
        vae_loss_map['loss'] = vae_loss_map['loss'] * pred_loss
        # nan_check_and_break(vae_loss_map['loss'], "full_loss")
        # vae_loss_map['loss'] = self.config['max_time_steps'] * (vae_loss_map['loss'] + pred_loss)
        vae_loss_map['pred_loss_mean'] = torch.mean(pred_loss)
        vae_loss_map['loss_mean'] = torch.mean(vae_loss_map['loss'])
        return vae_loss_map

    def _z_to_image_transformer(self, z, imgs):
        return image_to_window(z[:, 0:3], self.config['window_size'],
                               imgs, max_image_percentage=self.config['max_image_percentage'])

    def _z_to_image_lambda(self, z, imgs):
        crops = torch.cat(self.pool(imgs, z.clone().detach().cpu().numpy()), 0)
        crops = crops.cuda() if z.is_cuda else crops
        return crops

    def _z_to_image(self, z, imgs):
        ''' accepts images (or lambdas to crop) and z (gauss or disc)
            and returns an [N, C, H_trunc, W_trunc] array '''
        if self.config['reparam_type'] == 'discrete':
            raise NotImplementedError

        assert z.size()[-1] >= 3, "z needs to be at least [scale, x, y]"
        assert len(imgs) == z.shape[0], "batch sizes for crop preds vs imgs dont match"
        crop_fn_map = {
            'transformer': self._z_to_image_transformer,
            'lambda': self._z_to_image_lambda
        }
        crop_type = 'transformer' if isinstance(imgs, torch.Tensor) else 'lambda'
        return crop_fn_map[crop_type](z, imgs)

    def generate(self, batch_size):
        self.eval()
        samples = []
        with torch.no_grad():
            for i in range(self.config['max_time_steps']):
                samples.append(
                    self.vae.generate_synthetic_samples(batch_size, reset_state=i==0)
                )

        return samples

    def forward(self, x, x_related):
        ''' encode with VAE, then use posterior sample to
            gather image from original space (eg: 4k image) '''
        batch_size, chans = x_related.size(0), x_related.size(1)

        def _forward_internal(x_related, inference_only=False):
            params, crops, decodes = [], [], []

            # reset the state, output and the truncate window
            self.vae.memory.init_state(batch_size, cuda=x_related.is_cuda)
            self.vae.memory.init_output(batch_size, seqlen=1, cuda=x_related.is_cuda)

            # accumulator for predictions
            x_preds = zeros((batch_size, self.output_size), #self.config['latent_size']),
                            cuda=x_related.is_cuda,
                            dtype=get_dtype(x_related)).requires_grad_()

            for i in range(self.config['max_time_steps']):
                # get posterior and params, expand 0'th dim for seqlen
                z_t, params_t = self.vae.posterior(x_related)

                nan_check_and_break(x_related, "x_related")
                nan_check_and_break(z_t['prior'], "prior")
                nan_check_and_break(z_t['posterior'], "posterior")
                nan_check_and_break(z_t['x_features'], "x_features")

                # extract the required crop from original image
                x_trunc_t = self._z_to_image(z_t['posterior'], x)
                nan_check_and_break(x_trunc_t, "x_trunc_t")

                # do preds and sum
                state = torch.mean(self.vae.memory.get_state()[0], 0)
                x_preds = x_preds + self.latent_projector(x_trunc_t, state)

                # decode the posterior
                decoded_t = self.vae.decode(z_t, produce_output=True)
                nan_check_and_break(decoded_t, "decoded_t")

                # cache for loss function & visualization
                params.append(params_t)
                crops.append(x_trunc_t)
                decodes.append(decoded_t)

            #preds = self.latent_projector.get_output(x_preds / self.config['max_time_steps'])
            preds = x_preds / self.config['max_time_steps']
            return {
                'decoded': decodes,
                'params': params,
                'preds': preds,
                'crops': crops
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
