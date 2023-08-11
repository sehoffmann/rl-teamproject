import torch
import numpy as np

def normal_kullback_div(mean1, std1_log, mean2, std2_log):
    """
    Unreduced KL-Divergence between two normal distributions
    """
    std1 = std1_log.exp()
    std2 = std2_log.exp()
    return (std2_log - std1_log) + (std1.pow(2) + (mean1 - mean2).pow(2)) / (2 * std2.pow(2)) - 0.5

def crps_loss(target, mean, std_log, reduction = 'mean'):
    sqrtPi = torch.as_tensor(np.pi).sqrt()
    sqrtTwo = torch.as_tensor(2.).sqrt()

    sigma = std_log.exp() # ensures positivity
    z = (target - mean) / sigma # z transform
    phi = torch.exp(-z ** 2 / 2).div(sqrtTwo * sqrtPi) # standard normal pdf
    loss = sigma * (z * torch.erf(z / sqrtTwo) + 2 * phi - 1 / sqrtPi) # crps as per Gneiting et al 2005
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss