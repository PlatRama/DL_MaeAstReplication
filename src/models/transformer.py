import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger('mae_ast')

class TransformerEncoder(nn.Module):
    """
    Stack of Transformer encoder layers.

    Default configuration (from paper):
    - Encoder: 6 layers, 768 dim, 12 heads, 3072 FFN
    - Decoder: 2 layers, 768 dim, 12 heads, 3072 FFN

    Args:
        num_layers: Number of encoder layers
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ffn_dim: FFN hidden dimension
        dropout: Dropout probability
        layer_norm_first: Pre-LN vs Post-LN
    """

    def __init__(
            self,
            num_layers: int = 6,
            embed_dim: int = 768,
            num_heads: int = 12,
            ffn_dim: int = 3072,
            dropout: float=0.1,
            layer_norm_first: bool = True
    ):
        super().__init__()

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.layer_norm_first = layer_norm_first

        layers = []
        for _ in range(num_layers):
            layers.append(nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,  # Fondamentale per il nostro input [B, N, D]
                norm_first=layer_norm_first
            ))
        self.layers = nn.ModuleList(layers)

        if layer_norm_first:
            self.final_norm = nn.LayerNorm(embed_dim)
        else:
            self.final_norm = None

        logger.info(f"TransformerEncoder: layers={num_layers}, dim={embed_dim}, "
                    f"heads={num_heads}, ffn_dim={ffn_dim}")

    def forward(self,
                x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
               Forward pass through all encoder layers.

               Args:
                   x: Input [batch, seq_len, embed_dim]
                   padding_mask: Padding mask [batch, seq_len]

               Returns:
                   output: Final output [batch, seq_len, embed_dim]
               """
        for layer in self.layers:
            # La libreria chiama il parametro src_key_padding_mask
            x = layer(x, src_key_padding_mask=padding_mask)

        if self.final_norm is not None:
            x = self.final_norm(x)

        return x