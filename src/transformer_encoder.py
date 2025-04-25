import torch.nn as nn
from transformer_encoder_block import TransformerEncoderBlock

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
