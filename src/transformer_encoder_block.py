import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        
        attention_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout(attention_output)
        
        x_norm = self.norm2(x)
        ffn_output = self.feed_forward(x_norm)
        x = x + self.dropout(ffn_output)
        
        return x