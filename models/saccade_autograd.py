import torch

class SaccadeAutoGradFn(torch.autograd.Function):
    """
    A custom backward pass for our [s, x, y] vector when using hard attention

grid =  torch.Size([16, 32, 32, 2])
grad_output_shape =  torch.Size([16, 1, 32, 32])
z_grad =  torch.Size([48, 1, 32, 32])
Traceback (most recent call last):
  File "main.py", line 552, in <module>
    run(args)
  File "main.py", line 528, in run
    train(epoch, model, optimizer, loader.train_loader, grapher)
  File "main.py", line 416, in train
    plot_mem=True)
  File "main.py", line 356, in execute_graph
    loss_t['loss_mean'].backward()
  File "/home/jramapuram/.venv3/lib/python3.7/site-packages/torch/tensor.py", line 93, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/home/jramapuram/.venv3/lib/python3.7/site-packages/torch/autograd/__init__.py", line 90, in backward
    allow_unreachable=True)  # allow_unreachable flag
RuntimeError: Function SaccadeAutoGradFnBackward returned an invalid gradient at index 0 - expected shape [16, 3] but got [48, 1, 32, 32]

    """
    # @staticmethod
    # def forward(ctx, z, crops, crops_num_grad):
    #     """
    #     In the forward pass we receive a Tensor containing the input and return
    #     a Tensor containing the output. ctx is a context object that can be used
    #     to stash information for backward computation. You can cache arbitrary
    #     objects for use in the backward pass using the ctx.save_for_backward method.
    #     """
    #     assert z.size(1) == 3, \
    #         "numerical spatial transformer currently only operates over 3-features dims"
    #     ctx.save_for_backward(crops_fx_p_h, crops_fx_m_h)
    #     return crops.clone()

    # @staticmethod
    # def backward(ctx, grad_output):
    #     """
    #     In the backward pass we receive a Tensor containing the gradient of the loss
    #     with respect to the output, and we need to compute the gradient of the loss
    #     with respect to the input.
    #     """
    #     eps = 1
    #     fx_p_h, fx_m_h = ctx.saved_tensors
    #     z_grad = (fx_p_h - fx_m_h) / (2 * eps) # [batch, 3, chans, x, y]
    #     z_grad = torch.matmul(grad_output.unsqueeze(1), z_grad) # connect the grads
    #     z_grad = torch.sum(torch.sum(torch.sum(z_grad, -1), -1), -1) # reduce over y, x, chans

    #     print("grad_output = ", type(grad_output))
    #     print("grad_output_shape = ", grad_output.shape)
    #     print("z_grad = ", z_grad.shape)
    #     return z_grad, None, None, None # we don't need hard-crop grads


    @staticmethod
    def forward(ctx, z, crops, numerical_grad):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        assert z.size(1) == 3, \
            "numerical spatial transformer currently only operates over 3-features dims"
        ctx.save_for_backward(numerical_grad)
        return crops.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        z_grad, = ctx.saved_tensors # intentional comma
        z_grad = torch.matmul(grad_output.unsqueeze(1), z_grad) # connect the grads
        z_grad = torch.sum(torch.sum(torch.sum(z_grad, -1), -1), -1) # reduce over y, x, chans
        return z_grad, None, None, None # we don't need hard-crop grads, etc
