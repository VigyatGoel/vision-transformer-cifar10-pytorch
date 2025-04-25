import torch.nn as nn
from vit_embeddings import ViTEmbeddings
from transformer_encoder import TransformerEncoder

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.embeddings = ViTEmbeddings(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        cls_token_output = x[:, 0]
        logits = self.cls_head(cls_token_output)
        return logits