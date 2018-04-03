import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable

from helpers.utils import expand_dims, check_or_create_dir
from helpers.layers import View

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


# Scaling by `1/scale` here is unsatisfactory, as `scale` could be
# zero.
def z_where_inv(z_where):
    # Take a batch of z_where vectors, and compute their "inverse".
    # That is, for each row compute:
    # [s,x,y] -> [1/s,-x/s,-y/s]
    # These are the parameters required to perform the inverse of the
    # spatial transform performed in the generative model.
    n = z_where.size(0)
    out = torch.cat((ng_ones([1, 1]).type_as(z_where).expand(n, 1), -z_where[:, 1:]), 1)
    # Divide all entries by the scale.
    out = out / z_where[:, 0:1]
    return out


def window_to_image(z_where, window_size, image_size, windows):
    n = windows.size(0)
    assert windows.size(1) == window_size ** 2, 'Size mismatch.'
    theta = expand_z_where(z_where)
    grid = F.affine_grid(theta, torch.Size((n, 1, image_size, image_size)))
    out = F.grid_sample(windows.view(n, 1, window_size, window_size), grid)
    return out.view(n, 1, image_size, image_size)


def image_to_window(z_where, window_size, image_size, images):
    n = images.size(0)
    assert images.size(2) == images.size(3) == image_size[1] == image_size[2], 'Size mismatch.'
    theta_inv = expand_z_where(z_where_inv(z_where))
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
    def __init__(self, vae, **kwargs):
        super(Saccader, self).__init__()
        self.vae = vae
        self.config = kwargs['kwargs']
        self.loss_decoder = self._build_loss_decoder()

    def get_name(self):
        return "{}_win{}_us{}_dscale{}_{}".format(
            str(self.config['uid']),
            str(self.config['window_size']),
            str(self.config['upsample_size']),
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
                self.load_state_dict(torch.load(model_filename))
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
            torch.save(self.state_dict(), model_filename)

    def encode(self, x):
        return self.vae.posterior(x)

    def _build_loss_decoder(self):
        ''' helper function to build convolutional or dense decoder'''
        if self.config['reparam_type'] == 'discrete':
            decoder_input_size = int(np.prod(self.config['img_shp']))*self.vae.chans
        else:
            decoder_input_size = self.config['window_size']**2*self.vae.chans

        decoder = nn.Sequential(
            View([-1, decoder_input_size]),
            nn.Linear(decoder_input_size, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, self.vae.output_size)
        )

        if self.config['cuda']:
            decoder = decoder.cuda()

        return decoder

    def loss_function(self, vae_pred_logits, class_pred_logits, x, labels, params):
        ''' loss is: L_{classifier} * L_{VAE} '''
        vae_loss_map = self.vae.loss_function(vae_pred_logits, x, params)
        pred_loss = F.cross_entropy(input=class_pred_logits, target=labels, reduce=False)
        vae_loss_map['loss'] = vae_loss_map['loss'] * pred_loss
        vae_loss_map['pred_loss_mean'] = torch.mean(pred_loss)
        vae_loss_map['loss_mean'] = torch.mean(vae_loss_map['loss'])
        return vae_loss_map

    def _z_to_image(self, z, imgs):
        ''' accepts images and z (gauss or disc)
            and returns an [N, C, H_trunc, W_trunc] array '''
        if self.config['reparam_type'] == 'discrete':
            raise NotImplementedError
        elif self.config['reparam_type']== 'isotropic_gaussian' or self.config['reparam_type']== 'mixture':
            assert z.size()[-1] >= 3, "z needs to be [scale, x, y]"
            img_size = list(imgs.size())[1:]
            return image_to_window(z[:, 0:3], self.config['window_size'], img_size, imgs)
        else:
            raise Exception("{} reparameterizer not supported".format(
                self.config['reparam_type']
            ))

    def forward(self, x, x_related):
        ''' encode with VAE, then decode by projecting to classes '''
        z, params = self.encode(x_related)

        # extract regions from true image
        params['crop_imgs'] = self._z_to_image(z, x)

        # predict loss and decoded reconstruction
        pred_logits = self.loss_decoder(params['crop_imgs'])
        return pred_logits, self.vae.decode(z), params
