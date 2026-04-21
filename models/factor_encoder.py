import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorEncoder(nn.Module):
    def __init__(self, hidden_size, num_portfolios, num_factors):
        super(FactorEncoder, self).__init__()
        # Portfolio weights layer
        self.portfolio_layer = nn.Linear(hidden_size, num_portfolios)
        
        # Mapping layer for posterior distribution
        self.map_mu = nn.Linear(num_portfolios, num_factors)
        self.map_sigma = nn.Linear(num_portfolios, num_factors)

    def forward(self, y, e):
        # y shape: (N,), e shape: (N, H)
        
        # 1. Calculate portfolio weights (softmax over N stocks)
        # Eq 6: a_p^{(i,j)} = exp(w_p e^{(i)} + b_p) / sum(...)
        logits = self.portfolio_layer(e) # (N, M)
        a_p = F.softmax(logits, dim=0)   # (N, M)
        
        # 2. Calculate portfolio returns
        # Eq 7: y_p^{(j)} = sum(y^{(i)} a_p^{(i,j)})
        y_p = torch.matmul(y.unsqueeze(0), a_p).squeeze(0) # (M,)
        
        # 3. Mapping layer to posterior factor distribution
        # Eq 8
        mu_post = self.map_mu(y_p)                  # (K,)
        sigma_post = F.softplus(self.map_sigma(y_p)) # (K,)
        
        return mu_post, sigma_post