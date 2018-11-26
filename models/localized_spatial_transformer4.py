import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.autograd.function import once_differentiable

from helpers.utils import nan_check_and_break, zeros_like, zeros, get_dtype
from datasets.crop_dual_imagefolder import CropLambdaPool, USE_LIB


def scale(val, newmin, newmax, oldmin, oldmax):
    return (((val - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin


def theta_to_grid(theta, window_shape, image_shape, max_image_percentage):
    # convert theta from [N, 3] --> [N, 2, 3] & compute the grid
    assert image_shape[1] == image_shape[2], "only works over square imgs currently"
    max_scale = image_shape[1] / (image_shape[1] * max_image_percentage)
    theta_inv = LocalizedSpatialTransformerFn.expand_z_where(
        LocalizedSpatialTransformerFn.z_where_inv(theta, clip_scale=max_scale)
    )
    grid = F.affine_grid(theta_inv, torch.Size(window_shape))
    return grid


def pool_to_imgs(pool, crop_lambdas, grid, override=False):
    # compute top-left and bottom right corners
    top_left = torch.cat([grid[:, 0, 0, 0].unsqueeze(1),
                          grid[:, 0, 0, 1].unsqueeze(1)], 1)
    bottom_right = torch.cat([grid[:, -1, -1, 0].unsqueeze(1),
                              grid[:, -1, -1, 1].unsqueeze(1)], 1)

    # tabulate the crops using the threadpool and re-stitch together
    top_left_np = top_left.clone().detach().cpu().numpy()
    bottom_right_np = bottom_right.clone().detach().cpu().numpy()
    pool_tabulated = pool(crop_lambdas, top_left_np, bottom_right_np, override=override)
    crops = torch.cat(pool_tabulated, 0)

    # memory cleanups
    del pool_tabulated

    # push to cuda if necessary
    is_cuda = grid.is_cuda if isinstance(grid, torch.Tensor) else False
    return [top_left, bottom_right, crops.cuda() if is_cuda else crops]


class LocalizedSpatialTransformerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, pool, crop_lambdas, override):
        """Localized Spatial Transformer Autograd Function

        This function accepts a threadpool, a list of lambda function and the posterior
        and returns a set of crops. This contrasts traditional ST's by cropping true crops
        instead of just bilinearly transforming them. This is one of the reasons that ST's
        return blurrier and blurrier results.


        Attributes:
            theta ([N, 3] Tensor): A torch tensor that houses [s, x, y]

            pool (ThreadPool): A threadpool (either the RUST one or a joblib one).
                The pool accepts a set of lambdas and co-ordinates and returns crops.

            crop_lambdas (lambda): Lambda functions that accept a co-ordinate and return a crop.

            window_shape (int, int): window size of crop

            override (bool): allows exceeding the max_img_percent flag

        """
        ctx.is_cuda = grid.is_cuda

        # pass the work over to the thread-pool to directly return the crops (and grid)
        top_left, bottom_right, crops = pool_to_imgs(pool, crop_lambdas, grid, override)

        # cache for backward and return TRUE crops
        ctx.save_for_backward(crops, grid, top_left, bottom_right)
        return crops

    @staticmethod
    def straight_through(grad_grid, crops, grid):
        mag = grad_grid * crops
        gx = grid[:, :, :, 0].unsqueeze(1) * mag
        gy = grid[:, :, :, 1].unsqueeze(1) * mag
        gx = gx.squeeze(1)
        gy = gy.squeeze(1)
        return torch.cat([gx.unsqueeze(-1), gy.unsqueeze(-1)], -1), None, None, None

    @staticmethod
    def _crop_window(tensor, top_left, bottom_right, W, H):
        ''' finds the top left and bottom right corners and returns'''
        cropped_matrix_list = []
        for i, (nw, se) in enumerate(zip(top_left, bottom_right)):
            nw_x = int(scale(nw[0].item(), 0, W-1, -1, 1))
            nw_y = int(scale(nw[1].item(), 0, H-1, -1, 1))
            se_x = int(scale(se[0].item(), 0, W-1, -1, 1))
            se_y = int(scale(se[1].item(), 0, H-1, -1, 1))
            # print("nw_x = ", nw_x, " nw_y = ", nw_y, " | se_x = ", se_x, " | se_y = ", se_y)
            # print("tensor[i]= ", tensor[i].shape)
            cropped_matrix_list.append(
                #F.pad(tensor[i], (padding, padding))[nw_x:se_x, nw_y:se_y].unsqueeze(0).unsqueeze(0)
                tensor[i, :, nw_x:se_x, nw_y:se_y]
            )
            # for i, (nw, se) in enumerate(zip(top_left, bottom_right))
            # cropped_matrix_list = [
            #     F.pad(tensor[i], (padding, padding))[nw[0].item():se[0].item(), nw[1].item():se[1].item()].unsqueeze(0).unsqueeze(0)
            #     for i, (nw, se) in enumerate(zip(top_left, bottom_right))]

        # print('cropped = ', [t.shape for t in cropped_matrix_list])
        return cropped_matrix_list # can't concat because sizes are different

    @staticmethod
    def _interspace_base_grid(base_grid, top_left, bottom_right, W, H):
        for i, (nw, se) in enumerate(zip(top_left, bottom_right)):
            nw_x, nw_y = nw[0].item(), nw[1].item()
            se_x, se_y = se[0].item(), se[1].item()
            base_grid[i, :, :, 0] = torch.linspace(nw_x, se_x, W)
            base_grid[i, :, :, 1] = torch.linspace(nw_y, se_y, H)

        return base_grid

    @staticmethod
    def _crop_resize(matrix, top_left, bottom_right, W, H):
        matrix = LocalizedSpatialTransformerFn._crop_window(matrix, top_left, bottom_right, W, H)
        return torch.cat([F.interpolate(mv.unsqueeze(0), size=(W, H), mode='bilinear')
                          for mv in matrix], 0)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grid):
        '''  Example sizing:

              base_grid =  torch.Size([100, 64, 64, 2])
              gray_x_subgradient =  torch.Size([100, 64, 64])
              gray_y_subgradient =  torch.Size([100, 64, 64])
              m_x =  torch.Size([100, 64, 64])  | m_y =  torch.Size([100, 64, 64])
              grad_x_bmm =  torch.Size([100, 1, 64, 64])
              grad_y_bmm =  torch.Size([100, 1, 64, 64])
              grad_y_preconcat =  torch.Size([100, 64, 64])
              grad_x_preconcat =  torch.Size([100, 64, 64])
              final =  torch.Size([100, 64, 64, 2])

        '''
        crops, grid, top_left, bottom_right = ctx.saved_tensors

        # create the base grid which we will need for the gradients
        W, H = crops.shape[-2], crops.shape[-1]
        base_grid = crops.new(crops.shape[0], W, H, 2)
        base_grid = LocalizedSpatialTransformerFn._interspace_base_grid(base_grid, top_left,
                                                                        bottom_right, W, H)
        # base_grid[:, :, :, 0] = torch.linspace(-1, 1, W)
        # base_grid[:, :, :, 1] = torch.linspace(-1, 1, H)

        # compare the base grid to the grid from the affine generator
        grad_x = torch.zeros_like(grid[:, :, :, 0])
        grad_x[base_grid[:, :, :, 0] >= grid[:, :, :, 0]] = 1
        grad_x[base_grid[:, :, :, 0] < grid[:, :, :, 0]] = -1
        grad_y = torch.zeros_like(grid[:, :, :, 1])
        grad_y[base_grid[:, :, :, 1] >= grid[:, :, :, 1]] = 1
        grad_y[base_grid[:, :, :, 1] < grid[:, :, :, 1]] = -1

        # create the max(0, 1 - |y_s - m|) and max(0, 1 - |x_s - m|) terms
        m_x = torch.max(torch.zeros_like(base_grid[:, :, :, 0]),
                        1 - torch.abs(grid[:, :, :, 0] - base_grid[:, :, :, 0]))
        m_y = torch.max(torch.zeros_like(base_grid[:, :, :, 1]),
                        1 - torch.abs(grid[:, :, :, 1] - base_grid[:, :, :, 1]))

        # grad in x leaves m_y as const and vice versa
        # see eqn 7 in https://arxiv.org/abs/1506.02025
        grad_x = torch.bmm(grad_x, m_y).unsqueeze(1)
        grad_y = torch.bmm(m_x, grad_y).unsqueeze(1)
        # grad_y = torch.bmm(grad_y, m_x).unsqueeze(1)

        # crop to gradients to match the crops and interpolate up
        grad_x = LocalizedSpatialTransformerFn._crop_resize(grad_x, top_left, bottom_right, W, H)
        grad_y = LocalizedSpatialTransformerFn._crop_resize(grad_y, top_left, bottom_right, W, H)

        # multiply by the crops and grad-grid from above
        grad_y = grad_grid * (crops * grad_y)
        grad_x = grad_grid * (crops * grad_x)
        grad_x = grad_x.squeeze(1)
        grad_y = grad_y.squeeze(1)

        # concat on the last dimension as expected for affine_grid
        final_grads  = torch.cat([grad_x.unsqueeze(-1), grad_y.unsqueeze(-1)], -1)
        return final_grads, None, None, None



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
    def z_where_inv(z_where, clip_scale=5.0):
        # Take a batch of z_where vectors, and compute their "inverse".
        # That is, for each row compute:
        # [s,x,y] -> [1/s,-x/s,-y/s]
        # These are the parameters required to perform the inverse of the
        # spatial transform performed in the generative model.
        n = z_where.size(0)
        out = torch.cat((LocalizedSpatialTransformerFn.ng_ones([1, 1]).type_as(z_where).expand(n, 1),
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
    def expand_z_where(z_where):
        # Take a batch of three-vectors, and massages them into a batch of
        # 2x3 matrices with elements like so:
        # [s,x,y] -> [[s,0,x],
        #             [0,s,y]]
        n = z_where.size(0)
        out = torch.cat((LocalizedSpatialTransformerFn.ng_zeros([1, 1]).type_as(z_where).expand(n, 1), z_where), 1)
        expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])
        ix = Variable(expansion_indices)
        if z_where.is_cuda:
            ix = ix.cuda()

        out = torch.index_select(out, 1, ix)
        out = out.view(n, 2, 3)
        return out

    @staticmethod
    def _find_nw_se(matrix, padding=3):
        ''' finds north-east and south-west upper bounds and returns'''
        nonzero_elems = matrix.nonzero()
        nw = [nonzero_elems[:, 0].min(), nonzero_elems[:, 1].min()]
        se = [nonzero_elems[:, 0].max(), nonzero_elems[:, 1].max()]

        # if we have dupes expand by padding
        if se[0] == nw[0]:
            se[0] += padding

        if se[1] == nw[1]:
            se[1] += padding

        return [nw, se]

    @staticmethod
    def _crop_zeros(tensor, padding=3):
        ''' finds the top left and bottom right corners and returns'''
        nw_list, se_list = zip(*[LocalizedSpatialTransformerFn._find_nw_se(matrix)
                                 for matrix in tensor])
        assert len(nw_list) == len(se_list) == len(tensor), "# samples mismatch"
        cropped_matrix_list = [
            F.pad(tensor[i], (padding, padding))[nw[0]:se[0], nw[1]:se[1]].unsqueeze(0).unsqueeze(0)
            for i, (nw, se) in enumerate(zip(nw_list, se_list))]
        return cropped_matrix_list, nw_list, se_list # can't concat because sizes are different


class LocalizedSpatialTransformer(nn.Module):
    def __init__(self, chans, config):
        super(LocalizedSpatialTransformer, self).__init__()
        self.chans = chans
        self.config = config

        # build the pool object to process mini-batch images in parallel
        if USE_LIB == 'rust':
            self.pool = CropLambdaPool(window_size=self.config['window_size'],
                                       chans=chans,
                                       max_image_percentage=self.config['max_image_percentage'])
        else:
            self.pool = CropLambdaPool(num_workers=self.config['batch_size'])
            #self.pool = CropLambdaPool(num_workers=1)  # disables
            #self.pool = CropLambdaPool(num_workers=-1) # auto-pool
            #self.pool = CropLambdaPool(num_workers=24) # fixed pool

    def _clamp_sample(self, theta):
        ''' helper to clamp to [-1, 1], TODO: what about just tanh? '''
        clamp_map = {
            'beta': lambda z: (z * 2) - 1,                           # [0, 1] * 2 = [0, 2] - 1 = [-1, 1]
            'isotropic_gaussian': lambda z: (((z + 3) / 6) * 2) -1,  # re-normalize to [0, 1] and same as above
        }
        return torch.clamp(clamp_map[self.config['reparam_type']](theta), -1.0, 1.0)

    def forward(self, theta, crop_lambdas, chans, override=False):
        assert theta.size(1) == 3, "localized spatial transformer currently only operates over 3-features dims"
        assert self.config['img_shp'][1] == self.config['img_shp'][2], "currently only works on square imgs"

        # first compute the grid
        window_shape = (theta.shape[0], chans,
                        self.config['window_size'],
                        self.config['window_size'])
        clamped_theta = self._clamp_sample(theta)
        grid = theta_to_grid(clamped_theta, window_shape,
                             self.config['img_shp'],
                             self.config['max_image_percentage'])

        # run the op for cropping
        return LocalizedSpatialTransformerFn.apply(grid,
                                                   self.pool,
                                                   crop_lambdas,
                                                   override)
