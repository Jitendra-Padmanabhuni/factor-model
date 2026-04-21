import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorPredictor(nn.Module):
    def __init__(self, hidden_size, num_factors):
        super(FactorPredictor, self).__init__()
        self.num_factors = num_factors
        self.hidden_size = hidden_size
        
        # Multi-head attention components
        self.w_key = nn.Linear(hidden_size, hidden_size)
        self.w_value = nn.Linear(hidden_size, hidden_size)
        # Global learnable query vector for each head (K heads, H size)
        self.q = nn.Parameter(torch.randn(num_factors, hidden_size))
        
        # Distribution network
        self.dist_mu = nn.Linear(num_factors * hidden_size, num_factors)
        self.dist_sigma = nn.Linear(num_factors * hidden_size, num_factors)

    def forward(self, e):
        # e shape: (N, H)
        k = self.w_key(e)   # (N, H)
        v = self.w_value(e) # (N, H)
        
        h_multi = []
        for i in range(self.num_factors):
            q_i = self.q[i] # (H,)
            
            # Attention scores: q_i dot k^T / (||q|| * ||k||)
            # Cosine similarity-like formulation
            q_norm = torch.norm(q_i)
            k_norm = torch.norm(k, dim=1)
            scores = torch.matmul(k, q_i) / (q_norm * k_norm + 1e-8)
            
            # Max(0, x) acting as a relu prior to normalization
            scores = F.relu(scores)
            
            # Eq 14: Normalize over N stocks
            a_att = scores / (scores.sum() + 1e-8) # (N,)
            
            # Global representation for this head
            h_att = torch.matmul(a_att.unsqueeze(0), v).squeeze(0) # (H,)
            h_multi.append(h_att)
            
        # Eq 15: Concatenate multi-head representations
        h_multi = torch.cat(h_multi, dim=0) # (K * H)
        
        # Eq 16: Distribution network
        mu_prior = self.dist_mu(h_multi)
        sigma_prior = F.softplus(self.dist_sigma(h_multi))
        
        return mu_prior, sigma_prior