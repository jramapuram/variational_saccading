# coding: utf-8

import torch
import torch.nn.functional as F

from helpers.utils import int_type, add_noise_to_imgs


class NumDiffAutoGradFn(torch.autograd.Function):
    """
    A custom backward pass for our [s, x, y] vector when using hard attention

    grid =  torch.Size([16, 32, 32, 2])
    grad_output_shape =  torch.Size([16, 1, 32, 32])
    z_grad =  torch.Size([48, 1, 32, 32])
    expected shape [16, 3] but got [48, 1, 32, 32]

    """

    @staticmethod
    def forward(ctx, z, crops, window_size, delta):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        window_size_tensor = int_type(z.is_cuda)([window_size])
        delta_tensor = int_type(z.is_cuda)([delta])
        ctx.save_for_backward(z, crops, window_size_tensor, delta_tensor) # save the full extra window
        _, _, cw_b, cw_e, ch_b, ch_e = NumDiffAutoGradFn._get_dims(crops, window_size)
        return crops[:, :, cw_b:cw_e, ch_b:ch_e].clone() # return center crop

    @staticmethod
    def _get_dims(crops, window_size):
        W, H = crops.shape[2:]
        assert (W - window_size) % 2 == 0, "width - window_size is not divisible by 2"
        assert (H - window_size) % 2 == 0, "height - window_size is not divisible by 2"
        cw_b, cw_e = [int((W - window_size) / 2.) ,
                      int((W - window_size) / 2. + window_size)]
        ch_b, ch_e = [int((H - window_size) / 2.) ,
                      int((H - window_size) / 2. + window_size)]
        return W, H, cw_b, cw_e, ch_b, ch_e

    @staticmethod
    def _numerical_grads(crops, window_size, delta=1):
        ''' takes an enlarged crop window and returns delta-px perturbed grads
            eg for delta=1:
                full =  34 34  | delta = 1  | center =  1 33 1 33
                xmh =  0 32 1 33
                xph =  2 34 1 33
                ymh =  1 33 0 32
                yph =  1 33 2 34
                smh =  2 32 2 32
                sph =  0 34 0 34

           eg for delta=2:
               full =  36 36  | delta = 1  | center =  2 34 2 34
               xmh =  1 33 2 34
               xph =  3 35 2 34
               ymh =  2 34 1 33
               yph =  2 34 3 35
               smh =  3 33 3 33
               sph =  1 35 1 35
               full =  36 36  | delta = 2  | center =  2 34 2 34
               xmh =  0 32 2 34
               xph =  4 36 2 34
               ymh =  2 34 0 32
               yph =  2 34 4 36
               smh =  4 32 4 32
               sph =  0 36 0 36
        '''
        assert len(crops.shape) == 4, "num-grad needs 4d inputs"
        assert window_size > 1, "window size needs to be larger than 1"

        # get dims and sanity check
        W, H, cw_b, cw_e, ch_b, ch_e = NumDiffAutoGradFn._get_dims(crops, window_size)
        # print("full = ", W, H, " | delta =", delta, " | center = ", cw_b, cw_e, ch_b, ch_e)
        assert cw_b - delta >= 0
        assert cw_e + delta < W + 1
        assert ch_b - delta >= 0
        assert ch_e + delta < H + 1

        # [f(x+h, y) - f(x-h, y)] / delta
        fx_m_h = crops[:, :, cw_b-delta:cw_e-delta, ch_b:ch_e]
        fx_p_h = crops[:, :, cw_b+delta:cw_e+delta, ch_b:ch_e]
        # print('xmh = ',cw_b-delta,cw_e-delta, ch_b,ch_e)
        # print('xph = ', cw_b+delta,cw_e+delta, ch_b,ch_e)
        dfx = (fx_p_h - fx_m_h) / delta

        # [f(x, y+h) - f(x, y-h)] / delta
        fy_m_h = crops[:, :, cw_b:cw_e, ch_b-delta:ch_e-delta]
        fy_p_h = crops[:, :, cw_b:cw_e, ch_b+delta:ch_e+delta]
        # print('ymh = ', cw_b,cw_e, ch_b-delta,ch_e-delta)
        # print('yph = ', cw_b,cw_e, ch_b+delta,ch_e+delta)
        dfy = (fy_p_h - fy_m_h) / delta

        # approximately this: [f(x, y, s*1.01) - f(x, y, s*0.99)] / delta
        fs_m_h = crops[:, :, cw_b+delta:cw_e-delta, ch_b+delta:ch_e-delta]
        fs_p_h = crops[:, :, cw_b-delta:cw_e+delta, ch_b-delta:ch_e+delta]
        # print("fs_m_h = ", fs_m_h.shape, " | fs_p_h = ", fs_p_h.shape)
        fs_m_h = F.interpolate(fs_m_h, size=(window_size, window_size), mode='bilinear')
        fs_p_h = F.interpolate(fs_p_h, size=(window_size, window_size), mode='bilinear')
        # print('smh = ', cw_b+delta,cw_e-delta, ch_b+delta,ch_e-delta)
        # print('sph = ', cw_b-delta,cw_e+delta, ch_b-delta,ch_e+delta)
        dfs = (fs_p_h - fs_m_h) / delta # TODO: is this delta right?

        # expand 1'st dim and concat, returns [B, 3, C, W, H]
        grads = torch.cat(
            [dfs.unsqueeze(1), dfx.unsqueeze(1), dfy.unsqueeze(1)], 1
        )

        # memory cleanups
        del dfx; del fx_m_h; del fx_p_h
        del dfy; del fy_m_h; del fy_p_h
        del dfs; del fs_m_h; del fs_p_h

        return grads

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        z, crops, window_size, delta = ctx.saved_tensors
        window_size, delta = window_size.item(), delta.item()
        crops_perturbed = add_noise_to_imgs(crops) # add noise

        # get the mean of the gradients over all perturbations
        z_grad = torch.cat([NumDiffAutoGradFn._numerical_grads(
            crops_perturbed, window_size, k+1).unsqueeze(0) for k in range(0, delta)], 0)
        z_grad = torch.mean(z_grad, 0) # MC estimate over all possible perturbations
        #z_grad = torch.sum(z_grad, 0) # MC estimate over all possible perturbations, TODO: try mean
        z_grad = torch.matmul(grad_output.unsqueeze(1), z_grad) # connect the grads
        z_grad = torch.mean(torch.mean(torch.mean(z_grad, -1), -1), -1) # reduce over y, x, chans
        #z_grad = torch.sum(torch.sum(torch.sum(z_grad, -1), -1), -1) # reduce over y, x, chans TODO: try mean

        del crops; del crops_perturbed # mem cleanups
        return z_grad, None, None, None # no need for grads for crops, window_size and delta
