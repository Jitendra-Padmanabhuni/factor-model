import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(FeatureExtractor, self).__init__()
        self.proj = nn.Linear(num_features, hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        # GRU for temporal dependence
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        # x shape: (N, T, C)
        N, T, C = x.shape
        
        # Project and apply LeakyReLU
        h_proj = self.leaky_relu(self.proj(x)) # (N, T, H)
        
        # Pass through GRU
        _, h_n = self.gru(h_proj) # h_n shape: (1, N, H)
        
        # Return the last hidden state as stock latent features
        e = h_n.squeeze(0) # (N, H)
        return e