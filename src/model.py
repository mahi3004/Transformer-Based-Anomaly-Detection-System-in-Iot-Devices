import torch
import torch.nn as nn


class IoTTransformer(nn.Module):
    def __init__(self, num_features=115, embed_dim=32, num_heads=4,
                 num_layers=2, ff_dim=64, dropout=0.1, num_classes=3):
        super().__init__()
        # each single feature value (1 number) -> embed_dim numbers
        self.embed = nn.Linear(1, embed_dim)
        # learned "which feature am I" tag, one per token
        self.pos = nn.Parameter(torch.randn(1, num_features, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, num_classes))

    def forward(self, x):
        # x: (batch, 115)
        x = x.unsqueeze(-1)        # (batch, 115, 1)   each feature = one token
        x = self.embed(x)          # (batch, 115, 32)
        x = x + self.pos           # add positional embedding
        x = self.encoder(x)        # (batch, 115, 32)
        x = x.mean(dim=1)          # (batch, 32)       mean pooling
        return self.head(x)        # (batch, 3)        raw scores (logits)


if __name__ == "__main__":
    model = IoTTransformer()
    out = model(torch.randn(64, 115))
    print("output shape:", out.shape)
    print("trainable parameters:", sum(p.numel() for p in model.parameters()))
