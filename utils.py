import torch
import torch.nn.functional as F

def reparameterize(mu, sigma):
    """Samples from a Gaussian distribution using the reparameterization trick."""
    eps = torch.randn_like(sigma)
    return mu + eps * sigma

def kl_divergence(mu_post, sigma_post, mu_prior, sigma_prior):
    # ADDED EPSILON (1e-8) TO PREVENT DIVISION BY ZERO AND LOG(0)
    var_post = (sigma_post ** 2) + 1e-8
    var_prior = (sigma_prior ** 2) + 1e-8
    
    kl = 0.5 * (torch.log(var_prior / var_post) + (var_post + (mu_post - mu_prior)**2) / var_prior - 1)
    return kl.sum(dim=-1).mean()

def negative_log_likelihood(y_true, mu_y, sigma_y):
    # ADDED EPSILON (1e-8) TO PREVENT DIVISION BY ZERO AND LOG(0)
    var_y = (sigma_y ** 2) + 1e-8
    
    nll = 0.5 * torch.log(2 * torch.pi * var_y) + ((y_true - mu_y)**2) / (2 * var_y)
    return nll.mean()