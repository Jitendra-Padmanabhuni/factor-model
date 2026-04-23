import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers=2, nhead=4):
        super(FeatureExtractor, self).__init__()
        self.proj = nn.Linear(num_features, hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        
        # Add a LayerNorm right after the projection to stabilize inputs
        self.norm = nn.LayerNorm(hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size * 2, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (N, T, C)
        h_proj = self.leaky_relu(self.proj(x))
        h_proj = self.norm(h_proj) # <--- Stabilize before attention
        
        h_trans = self.transformer(h_proj)
        
        # We take the mean across the 20 days instead of just the last day.
        # This is more robust to noise than taking [:, -1, :]
        e = h_trans.mean(dim=1) 
        return e