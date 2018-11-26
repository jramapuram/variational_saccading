# here2!
# grad_y =  torch.Size([8, 64])  | grad_x =  torch.Size([8, 64])
# grad_x_small =  torch.Size([8, 64, 64])
# grad_y_small =  torch.Size([8, 64, 64])
# #nw =  8  # se = 8
# #nw =  8  # se = 8
# grad_grid =  torch.Size([8, 3, 32, 32])
# grad_x_upsampled =  torch.Size([8, 1, 32, 32])
# grad_y_upsampled =  torch.Size([8, 1, 32, 32])
# grad_y_final =  torch.Size([8, 1])
# grad_x_final =  torch.Size([8, 1])
# grad_theta =  torch.Size([8, 3])
# here2!
# grad_y =  torch.Size([8, 64])  | grad_x =  torch.Size([8, 64])
# grad_x_small =  torch.Size([8, 64, 64])
# grad_y_small =  torch.Size([8, 64, 64])
# #nw =  8  # se = 8
# #nw =  8  # se = 8
# grad_grid =  torch.Size([8, 3, 32, 32])
# grad_x_upsampled =  torch.Size([8, 1, 32, 32])
# grad_y_upsampled =  torch.Size([8, 1, 32, 32])
# grad_y_final =  torch.Size([8, 1])
# grad_x_final =  torch.Size([8, 1])
# grad_theta =  torch.Size([8, 3])
# here2!
# grad_y =  torch.Size([8, 64])  | grad_x =  torch.Size([8, 64])
# grad_x_small =  torch.Size([8, 64, 64])
# grad_y_small =  torch.Size([8, 64, 64])
# #nw =  8  # se = 8
# #nw =  8  # se = 8
# grad_grid =  torch.Size([8, 3, 32, 32])
# grad_x_upsampled =  torch.Size([8, 1, 32, 32])
# grad_y_upsampled =  torch.Size([8, 1, 32, 32])
# grad_y_final =  torch.Size([8, 1])
# grad_x_final =  torch.Size([8, 1])
# grad_theta =  torch.Size([8, 3])
# nones =  False False False False False False
# nones =  False False False False False False
# nones =  False False False False False False
# here2!
# grad_y =  torch.Size([8, 64])  | grad_x =  torch.Size([8, 64])
# grad_x_small =  torch.Size([8, 64, 64])
# grad_y_small =  torch.Size([8, 64, 64])
# #nw =  8  # se = 8
# #nw =  8  # se = 8
# grad_grid =  torch.Size([8, 3, 32, 32])
# grad_x_upsampled =  torch.Size([8, 1, 32, 32])
# grad_y_upsampled =  torch.Size([8, 1, 32, 32])
# grad_y_final =  torch.Size([8, 1])
# grad_x_final =  torch.Size([8, 1])
# grad_theta =  torch.Size([8, 3])
# here2!
# grad_y =  torch.Size([8, 64])  | grad_x =  torch.Size([8, 64])
# grad_x_small =  torch.Size([8, 64, 64])
# grad_y_small =  torch.Size([8, 64, 64])
# #nw =  8  # se = 8
# #nw =  8  # se = 8
# grad_grid =  torch.Size([8, 3, 32, 32])
# grad_x_upsampled =  torch.Size([8, 1, 32, 32])
# grad_y_upsampled =  torch.Size([8, 1, 32, 32])
# grad_y_final =  torch.Size([8, 1])
# grad_x_final =  torch.Size([8, 1])
# grad_theta =  torch.Size([8, 3])

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd.function import once_differentiable

from helpers.utils import nan_check_and_break, zeros_like, zeros, get_dtype
from datasets.crop_dual_imagefolder import CropLambdaPool, USE_LIB


def pool_to_imgs(pool, crop_lambdas, theta, override=False):
    # tabulate the crops using the threadpool and re-stitch together
    theta_np = theta.clone().detach().cpu().numpy() if not isinstance(theta, np.ndarray) else theta
    pool_tabulated = pool(crop_lambdas, theta_np, override=override)
    crops = torch.cat(pool_tabulated, 0)

    # memory cleanups
    del pool_tabulated

    # push to cuda if necessary
    is_cuda = theta.is_cuda if isinstance(theta, torch.Tensor) else False
    return crops.cuda() if is_cuda else crops


class LocalizedSpatialTransformerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, pool, crop_lambdas, chans, trunc_size, max_scale, override):
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

            chans (int): channels in original full-resolution image

            trunc_size ((int, int)): A tuple containing the size of the grid to generate.
                This grid is then upsampled using F.upsample

            max_scale (float): sets the maximum scale allowable for truncation

           override (bool): allows exceeding the max_img_percent flag

        """
        assert type(trunc_size) == torch.Size # cant save otherwise
        ctx.is_cuda = theta.is_cuda

        N, C, H, W = theta.size(0), chans, *trunc_size
        ctx.size = [N, C, H, W]

        # the grid is created by an inner product across a linspace in x & y
        base_grid = [torch.linspace(-1, 1, W).expand(N, W),
                     torch.linspace(-1, 1, H).expand(N, H)]
        base_grid = [base_grid[0].cuda() if ctx.is_cuda else base_grid[0],
                     base_grid[1].cuda() if ctx.is_cuda else base_grid[1]]

        mat[grid[:]]

        # concat and add a ones-like on the chan dim
        grid_concat = torch.cat([base_grid[0].unsqueeze(2),
                                 base_grid[1].unsqueeze(2),
                                 torch.ones_like(base_grid[1]).unsqueeze(2)], 2)
        theta_inv = LocalizedSpatialTransformerFn.expand_z_where(
            LocalizedSpatialTransformerFn.z_where_inv(theta, clip_scale=max_scale)
        )
        grid = torch.bmm(grid_concat, theta_inv.transpose(1, 2))

        m_x = torch.max(torch.zeros_like(grid[:, :, 0]),
                        1 - torch.abs(theta[:, 1].unsqueeze(1) - grid[:, :, 0])).unsqueeze(2)
        m_y = torch.max(torch.zeros_like(grid[:, :, 1]),
                        1 - torch.abs(theta[:, 2].unsqueeze(1) - grid[:, :, 1])).unsqueeze(1)

        # the final mask is a batch-matrix-multiple of each inner product
        # the dimensions expected here are m_x = [N, W, 1] & m_y = [N, 1, H]
        # M = torch.bmm(m_x, m_y)
        # _, nw_list, se_list = LocalizedSpatialTransformerFn.clip_and_crop(M)
        #print('nw = ', nw_list[0:5], ' | se = ', se_list[0:5])
        #M_clip = torch.clamp(M, M.max()*scale, M.max())


        # pass the work over to the thread-pool to directly return the crops
        theta_pool = theta.clone().detach().cpu()
        #crops = pool_to_imgs(pool, crop_lambdas, nw_list, se_list, override)
        crops = pool_to_imgs(pool, crop_lambdas, theta_pool, override=True)
        crops = crops.cuda() if ctx.is_cuda else crops

        # cache for backward and return TRUE crops
        ctx.save_for_backward(theta, crops, grid,
                              base_grid[0], base_grid[1],
                              m_x, m_y)
        # ctx.M = M # TODO: do we need?
        # crp_out = torch.cat([crops[:, i, :, :] * M for i in range(chans)], 1)
        # return crp_out.unsqueeze(1)

        return crops

    @staticmethod
    def clip_and_crop(matrix, clip_scale=0):
        matrix[matrix < matrix.max() * clip_scale] = 0
        matrix, nw_list, se_list = LocalizedSpatialTransformerFn._crop_zeros(matrix)
        return matrix, nw_list, se_list

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grid):
        theta, crops, grid, base_grid_x, base_grid_y, m_x, m_y = ctx.saved_tensors
        # assert ctx.is_cuda == grad_grpid.is_cuda
        # base_grid, m_x, m_y = [ctx.base_grid_x, ctx.base_grid_y], ctx.m_x, ctx.m_y
        # cached_crops, cached_theta = ctx.crops, ctx.theta
        # N, C, H, W = ctx.size

        # there are three subgradient terms each
        # grad_x = torch.zeros_like(base_grid_x)
        # grad_x[base_grid_x >= theta[:, 1].unsqueeze(1)] = 1
        # grad_x[base_grid_x < theta[:, 1].unsqueeze(1)] = -1
        # grad_y = torch.zeros_like(base_grid_y)
        # grad_y[base_grid_y >= theta[:, 2].unsqueeze(1)] = 1
        # grad_y[base_grid_y < theta[:, 2].unsqueeze(1)] = -1
        grad_x = torch.zeros_like(grid[:, :, 0])
        grad_x[grid[:, :, 0] >= theta[:, 1].unsqueeze(1)] = 1
        grad_x[grid[:, :, 0] < theta[:, 1].unsqueeze(1)] = -1
        grad_y = torch.zeros_like(grid[:, :, 1])
        grad_y[grid[:, :, 1] >= theta[:, 2].unsqueeze(1)] = 1
        grad_y[grid[:, :, 1] < theta[:, 2].unsqueeze(1)] = -1

        # unsqueeze for the BMM, grad_x = [N, W, 1], grad_y = [N, 1, H]
        grad_x = grad_x.unsqueeze(2)
        grad_y = grad_y.unsqueeze(1)

        # grad in x leaves m_y as const and vice versa
        # see eqn 7 in https://arxiv.org/abs/1506.02025
        grad_x_small = torch.bmm(grad_x, m_y).unsqueeze(1)
        grad_y_small = torch.bmm(m_x, grad_y).unsqueeze(1)

        # threshold the low-resolution st gradients
        # clip_scale = 0.1 # XXX, parameterize somewhere
        # grad_x_thresholded, grad_y_thresholded = grad_x_small.clone(), grad_y_small.clone()
        # grad_x_thresholded[grad_x_thresholded < grad_x_thresholded.max() * clip_scale] = 0
        # grad_y_thresholded[grad_y_thresholded < grad_y_thresholded.max() * clip_scale] = 0

        # # crop the gradients to their non-zero values after above threshold
        # grad_x_crops = LocalizedSpatialTransformerFn._crop_zeros(grad_x_thresholded)
        # grad_y_crops = LocalizedSpatialTransformerFn._crop_zeros(grad_y_thresholded)

        # # and then upsample to the size of the true crop size
        # true_crop_size = crops.size()[-2:]
        # grad_y_upsampled = torch.cat([F.interpolate(grad_y_crop, size=true_crop_size, mode='bilinear', align_corners=True)
        #                               for grad_y_crop in grad_y_crops], 0)
        # grad_x_upsampled = torch.cat([F.interpolate(grad_x_crop, size=true_crop_size, mode='bilinear', align_corners=True)
        #                               for grad_x_crop in grad_x_crops], 0)

        # grad_x_upsampled = grad_x_thresholded
        # grad_y_upsampled = grad_y_thresholded
        grad_x_upsampled = grad_x_small.clone()
        grad_y_upsampled = grad_y_small.clone()

        # multiply by the crops
        grad_y_final = grad_grid * (crops * grad_y_upsampled)
        grad_x_final = grad_grid * (crops * grad_x_upsampled)

        # reduce over H, W
        grad_y_final = torch.sum(grad_y_final, (1, 2, 3)).unsqueeze(1)
        grad_x_final = torch.sum(grad_x_final, (1, 2, 3)).unsqueeze(1)
        grad_s_final = torch.ones_like(grad_x_final) # XXX: need grads here

        # only gradient is for posterior
        grad_theta = torch.cat([grad_s_final, grad_x_final, grad_y_final], -1)
        return grad_theta, None, None, None, None, None, None


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
        ''' helper to clamp to [-1, 1], what about just tanh? '''
        clamp_map = {
            'beta': lambda z: (z * 2) - 1,                           # [0, 1] * 2 = [0, 2] - 1 = [-1, 1]
            'isotropic_gaussian': lambda z: (((z + 3) / 6) * 2) -1,  # re-normalize to [0, 1] and same as above
        }
        return torch.clamp(clamp_map[self.config['reparam_type']](theta), -1.0, 1.0)

    def forward(self, theta, crop_lambdas, override=False):
        assert theta.size(1) == 3, "localized spatial transformer currently only operates over 3-features dims"
        #theta_clamped = self._clamp_sample(theta)
        # self.config['window_size'], max_image_percentage=self.config['max_image_percentage']
        # theta_clamped = theta #TODO: clamp outside or inside?
        # def forward(ctx, theta, pool, crop_lambdas, chans, trunc_size, override=False):
        #window_size = [self.config['window_size'], self.config['window_size']]

        # run the autograd fn
        low_res_size  = torch.Size([self.config['window_size'], self.config['window_size']]) # XXX: parameterize
        # low_res_size  = torch.Size([100, 100]) # XXX: parameterize
        assert self.config['img_shp'][1] == self.config['img_shp'][2], "currently only works on square imgs"
        max_scale = self.config['img_shp'][1] / (self.config['img_shp'][1] * self.config['max_image_percentage'])
        return LocalizedSpatialTransformerFn.apply(theta, self.pool, crop_lambdas,
                                                   self.chans, low_res_size,
                                                   max_scale, override)
