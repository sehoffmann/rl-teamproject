import torch
import numpy as np
import torch.nn.functional as F

SQRT_PI = torch.as_tensor(np.pi).sqrt()
SQRT_TWO = torch.as_tensor(2.).sqrt()


def positive_std(x, eps=1e-2):
    return F.softplus(x) + eps


def reduce(x, reduction=None):
    reduction = reduction or 'mean'
    if reduction == 'mean':
        return x.mean()
    elif reduction == 'sum':
        return x.sum()
    elif reduction == 'none':
        return x
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def normal_kl_div(mu_1, sigma_1, mu_2, sigma_2, reduction=None):
    """
    Unreduced KL-Divergence between two normal distributions
    """
    l2 = (mu_1 - mu_2).pow(2)
    loss = torch.log(sigma_2/sigma_1) + (sigma_1.pow(2) + l2) / (2 * sigma_2.pow(2)) - 0.5
    i = loss.argmax(dim=0)
    max_loss = loss[i]
    if max_loss.item() > 10:
        print(f"KL: {max_loss.item()}, mu_1: {mu_1[i].item()}, mu_2: {mu_2[i].item()}, sigma_1: {sigma_1[i].item()}, sigma_2: {sigma_2[i].item()}")
    loss = reduce(loss, reduction)
    return loss


def crps_loss(target, mean, sigma, reduction=None):
    z = (target - mean) / sigma # z transform
    phi = torch.exp(-z ** 2 / 2).div(SQRT_TWO * SQRT_PI) # standard normal pdf
    loss = sigma * (z * torch.erf(z / SQRT_TWO) + 2 * phi - 1 / SQRT_PI) # crps as per Gneiting et al 2005
    loss = reduce(loss, reduction)
    return loss