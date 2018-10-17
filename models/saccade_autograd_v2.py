import torch
import torch.nn.functional as F

class SaccadeAutoGradFn(torch.autograd.Function):
    """
    A custom backward pass for our [s, x, y] vector when using hard attention

    grid =  torch.Size([16, 32, 32, 2])
    grad_output_shape =  torch.Size([16, 1, 32, 32])
    z_grad =  torch.Size([48, 1, 32, 32])
    expected shape [16, 3] but got [48, 1, 32, 32]

    """

    @staticmethod
    def forward(ctx, crops):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(crops) # save the full 1px extra window
        W, H = crops.shape[2:] # the true crop is in the center
        return crops[:, :, 1:W-1, 1:H-1].clone()

    @staticmethod
    def _numerical_grads(crops):
        ''' takes an enlarged crop window and returns 1px perturbed grads '''
        assert len(crops.shape) == 4, "num-grad needs 4d inputs"
        W, H = crops.shape[2:]

        # f(x+h, y) - f(x-h, y), eps=1
        fx_m_h = crops[:, :, 0:W-2, 1:H-1].clone()
        fx_p_h = crops[:, :, 2:W, 1:H-1].clone()
        dfx = fx_p_h - fx_m_h

        # f(x, y+h) - f(x, y-h), eps=1
        fy_m_h = crops[:, :, 1:W-1, 0:H-2].clone()
        fy_p_h = crops[:, :, 1:W-1, 2:H].clone()
        dfy = fy_p_h - fy_m_h

        # f(x, y, s*1.01) - f(x, y, s*0.99), eps=?
        fs_m_h = F.interpolate(crops[:, :, 2:W-2, 2:H-2].clone(), size=(W-2, H-2), mode='bilinear')
        fs_p_h = F.interpolate(crops.clone(), size=(W-2, H-2), mode='bilinear')
        dfs = fs_p_h - fs_m_h

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
        crops, = ctx.saved_tensors # intentional comma
        z_grad = SaccadeAutoGradFn._numerical_grads(crops)
        z_grad = torch.matmul(grad_output.unsqueeze(1), z_grad) # connect the grads
        z_grad = torch.sum(torch.sum(torch.sum(z_grad, -1), -1), -1) # reduce over y, x, chans
        del crops # mem cleanups
        return z_grad, None, None, None # we don't need hard-crop grads, etc
