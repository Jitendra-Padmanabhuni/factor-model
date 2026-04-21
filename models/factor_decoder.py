import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorDecoder(nn.Module):
    def __init__(self, hidden_size, num_factors):
        super(FactorDecoder, self).__init__()
        # Alpha layer
        self.alpha_hidden = nn.Linear(hidden_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.alpha_mu = nn.Linear(hidden_size, 1)
        self.alpha_sigma = nn.Linear(hidden_size, 1)
        
        # Beta layer
        self.beta_layer = nn.Linear(hidden_size, num_factors)

    def forward(self, e, mu_z, sigma_z=None):
        # e shape: (N, H), mu_z shape: (K,), sigma_z shape: (K,)
        N = e.shape[0]
        
        # --- Alpha Layer (Idiosyncratic) ---
        h_alpha = self.leaky_relu(self.alpha_hidden(e)) # (N, H)
        mu_alpha = self.alpha_mu(h_alpha).squeeze(-1)   # (N,)
        sigma_alpha = F.softplus(self.alpha_sigma(h_alpha)).squeeze(-1) # (N,)
        
        # --- Beta Layer (Exposure) ---
        beta = self.beta_layer(e) # (N, K)
        
        # --- Output Distribution ---
        # Eq 12
        mu_y = mu_alpha + torch.matmul(beta, mu_z) # (N,)
        
        if sigma_z is not None:
            # Broadcast sigma_z to match beta shape, elementwise multiply, square, sum over factors
            beta_var = torch.sum((beta * sigma_z.unsqueeze(0))**2, dim=1) # (N,)
            sigma_y = torch.sqrt(sigma_alpha**2 + beta_var)               # (N,)
            return mu_y, sigma_y
        
        return mu_y