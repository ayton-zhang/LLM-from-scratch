import torch
import torch.nn as nn


VOCAB_SIZE = 256
EMBED_DIM = 64
N_HEADS = 4
FF_DIM = 256


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu")
        return checkpoint
    except Exception as e:
        print(f"Failed to load checkpoint from {path}: {e}")
        return None


if __name__ == "__main__":
    embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
    model = TransformerBlock(EMBED_DIM, N_HEADS, FF_DIM)
    token_ids = torch.randint(0, VOCAB_SIZE, (2, 16))
    x = embedding(token_ids)
    out = model(x)
    print(f"Output shape: {out.shape}")
