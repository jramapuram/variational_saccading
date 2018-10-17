import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from helpers.utils import nan_check_and_break, zeros_like


class SpatialTransformer(nn.Module):
    def __init__(self, config):
        super(SpatialTransformer, self).__init__()
        self.config = config

    def forward(self, z, imgs):
        assert z.size(1) == 3, "spatial transformer currently only operates over 3-features dims"
        return self.image_to_window(z, self.config['window_size'], imgs,
                                    max_image_percentage=self.config['max_image_percentage'])

    # the folowing few helpers are from pyro example for AIR
    @staticmethod
    def ng_ones(*args, **kwargs):
        """
        :param torch.Tensor type_as: optional argument for tensor type

        A convenience function for Variable(torch.ones(...), requires_grad=False)
        """
        retype = kwargs.pop('type_as', None)
        p_tensor = torch.ones(*args, **kwargs)
        return Variable(p_tensor if retype is None else p_tensor.type_as(retype), requires_grad=False)

    @staticmethod
    def ng_zeros(*args, **kwargs):
        """
        :param torch.Tensor type_as: optional argument for tensor type

        A convenience function for Variable(torch.ones(...), requires_grad=False)
        """
        retype = kwargs.pop('type_as', None)
        p_tensor = torch.zeros(*args, **kwargs)
        return Variable(p_tensor if retype is None else p_tensor.type_as(retype), requires_grad=False)

    @staticmethod
    def expand_z_where(z_where):
        # Take a batch of three-vectors, and massages them into a batch of
        # 2x3 matrices with elements like so:
        # [s,x,y] -> [[s,0,x],
        #             [0,s,y]]
        n = z_where.size(0)
        out = torch.cat((SpatialTransformer.ng_zeros([1, 1]).type_as(z_where).expand(n, 1), z_where), 1)
        expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])
        ix = Variable(expansion_indices)
        if z_where.is_cuda:
            ix = ix.cuda()

        out = torch.index_select(out, 1, ix)
        out = out.view(n, 2, 3)
        return out

    @staticmethod
    def z_where_inv(z_where, clip_scale=5.0):
        # Take a batch of z_where vectors, and compute their "inverse".
        # That is, for each row compute:
        # [s,x,y] -> [1/s,-x/s,-y/s]
        # These are the parameters required to perform the inverse of the
        # spatial transform performed in the generative model.
        n = z_where.size(0)
        out = torch.cat((SpatialTransformer.ng_ones([1, 1]).type_as(z_where).expand(n, 1),
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

    @staticmethod
    def window_to_image(z_where, window_size, image_size, windows):
        n = windows.size(0)
        assert windows.size(1) == window_size ** 2, 'Size mismatch.'
        theta = SpatialTransformer.expand_z_where(z_where)
        grid = F.affine_grid(theta, torch.Size((n, 1, image_size, image_size)))
        out = F.grid_sample(windows.view(n, 1, window_size, window_size), grid)
        return out.view(n, 1, image_size, image_size)

    @staticmethod
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
        theta_inv = SpatialTransformer.expand_z_where(
            SpatialTransformer.z_where_inv(z_where, clip_scale=max_scale)
        )
        grid = F.affine_grid(theta_inv, torch.Size((n, 1, window_size, window_size)))
        out = F.grid_sample(images.view(n, *image_size), grid)
        return out
