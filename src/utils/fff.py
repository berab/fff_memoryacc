import torch

TINFO = torch.finfo(torch.float)

class ASigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        result = 1 / (1 + torch.exp(-a * x))  # Steep sigmoid
        ctx.save_for_backward(x, a)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_tensors
        term1, term2 = (a * torch.exp(-a * x)), (1 + torch.exp(-a * x)**2)
        term = (term1.clamp(min=TINFO.min, max=TINFO.max) /
                term2.clamp(min=TINFO.min, max=TINFO.max))
        grad_input = grad_output * term
        grad_a = None
        return grad_input, grad_a
    
class SignActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (torch.sign(x) + 1) / 2  # Forward pass

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # return grad_output 
        return grad_output / x.abs()

    # @staticmethod
    # def backward(ctx, grad_output):
    #     x, = ctx.saved_tensors
    #     grad_input = grad_output * (torch.exp(-x) / (1 + torch.exp(-x)**2))
    #     return grad_input

