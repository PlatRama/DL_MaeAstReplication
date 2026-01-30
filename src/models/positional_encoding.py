import logging
import math
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger('mae_ast')

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Where:
        pos = position in sequence
        i = dimension index
        d_model = embedding dimension

    Key properties:
    - Fixed (non-learned) encoding
    - Supports any sequence length up to max_len
    - Model can extrapolate to longer sequences
    - Smooth: nearby positions have similar encodings


    Args:
        d_model: Embedding dimension (default: 768 as in paper)
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(self,
                 d_model: int = 768,
                 max_len: int = 5000,
                 dropout: float = 0.0,):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

        # Create positional encoding matrix
        # Shape: [1, max_len, d_model]
        pe = torch.zeros(1, max_len, d_model)

        # Position indices: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]

        # Division term: 10000^(2i/d_model)
        # Equivalent to: exp(2i * -log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even dimensions
        pe[0, :, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd dimensions
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)


        logger.info(f"SinusoidalPositionalEncoding: d_model={d_model}, "
                    f"max_len={max_len}, dropout={dropout}")

    def forward(self,
                x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            padding_mask: Optional padding mask [batch, seq_len]
                         True = padding, False = valid

        Returns:
            Positional encoding [batch, seq_len, d_model]
            (To be added to x in the model)
        """
        batch_size, seq_len, _ = x.shape

        # Get positional encoding for this sequence length
        # pe: [1, max_len, d_model] -> [1, seq_len, d_model]
        pos_enc = self.pe[:, :seq_len, :]

        # Expand to batch size
        pos_enc = pos_enc.expand(batch_size, -1, -1)

        # Zero out padding positions
        if padding_mask is not None:
            # padding_mask: [batch, seq_len] -> [batch, seq_len, 1]
            pos_enc = pos_enc.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Apply dropout
        if self.dropout is not None:
            pos_enc = self.dropout(pos_enc)
        return pos_enc

    def extra_repr(self):
        return f"d_model={self.d_model}, max_len={self.max_len}"
