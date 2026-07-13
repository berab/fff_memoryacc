import torch
from torch.distributions.exponential import Exponential
from torch.distributions import HalfNormal

def get_exp(n_mem1: int, n: int) -> torch.Tensor:
    # mem1: more efficient memory

    # 1. Define your inputs (X must be >= 0)
    n_mem2 = n - n_mem1
    sigma = 1/(n**0.5)

    # x = torch.arange(0, n+1/n, 1/n) * sigma
    x1 = torch.linspace(0, n/2, steps=n_mem1)
    x2 = torch.linspace(n/2*(1+1/n_mem2), n, steps=n_mem2)
    x = torch.cat((x1, x2))

    # 2. Initialize the HalfNormal distribution with your sigma (scale)
    dist = Exponential(rate=sigma)

    # 3. Get the probability density (PDF)
    raw_px = torch.exp(dist.log_prob(x))
    px = raw_px / raw_px.sum()
    return px


def get_halfnormal(n_mem1: int, n: int) -> torch.Tensor:
    n_mem2 = n - n_mem1
    sigma = n**0.5

    x1 = torch.linspace(0, n/2, steps=n_mem1)
    x2 = torch.linspace(n/2*(1+1/n_mem2), n, steps=n_mem2)
    x = torch.cat((x1, x2))

    dist = HalfNormal(scale=sigma)

    raw_px = torch.exp(dist.log_prob(x))
    px = raw_px / raw_px.sum()
    return px
