import torch

TINFO = torch.finfo(torch.float)

class SignActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (torch.sign(x) + 1) / 2  # Forward pass

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # return grad_output 
        return grad_output / (x.abs() + 1e-6)

    # @staticmethod
    # def backward(ctx, grad_output):
    #     x, = ctx.saved_tensors
    #     grad_input = grad_output * (torch.exp(-x) / (1 + torch.exp(-x)**2))
    #     return grad_input

