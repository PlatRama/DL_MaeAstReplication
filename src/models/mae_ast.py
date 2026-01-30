from dataclasses import dataclass
import logging
from typing import Tuple, Optional, Dict

import torch
from torch import nn

from .masking import RandomMasking, ChunkedMasking
from .positional_encoding import SinusoidalPositionalEncoding
from .transformer import TransformerEncoder

logger = logging.getLogger('mae_ast')

@dataclass
class MAEASTConfig:
    n_mels: int = 128
    input_type: str = "patch"  # "patch" or "frame"

    # Patch-based tokenization (16×16 patches)
    # Used for: AudioSet, ESC-50
    patch_size_time: int = 16  # Patch size in time dimension
    patch_size_freq: int = 16  # Patch size in frequency dimension

    # Frame-based tokenization (128×2 frames)
    frame_size_time: int = 2  # Frame size in time (typically 2)
    # frame_size_freq is always n_mels (128) for frame-based

    # Encoder architecture
    encoder_embed_dim: int = 768
    encoder_depth: int = 6
    encoder_num_heads: int = 12
    encoder_ffn_dim: int = 3072  # 4× embed_dim

    # Decoder architecture
    decoder_embed_dim: int = 768
    decoder_depth: int = 2
    decoder_num_heads: int = 12
    decoder_ffn_dim: int = 3072

    # Masking
    mask_ratio: float = 0.75
    mask_type: str = 'random'  # 'random' or 'chunk'
    mask_batched: bool = False
    chunk_size_range: Tuple[int, int] = (3, 5)

    # Positional encoding
    use_sinusoidal_pos: bool = True
    use_conv_pos: bool = False
    max_position: int = 5000

    # Dropout
    dropout: float = 0.1

    # Training
    layer_norm_first: bool = True

    @property
    def token_size_time(self) -> int:
        """Get token size in time dimension based on input type."""
        if self.input_type == "patch":
            return self.patch_size_time
        else:  # frame
            return self.frame_size_time

    @property
    def token_size_freq(self) -> int:
        """Get token size in frequency dimension based on input type."""
        if self.input_type == "patch":
            return self.patch_size_freq
        else:  # frame
            return self.n_mels  # Full frequency for frames

    @property
    def token_dim(self) -> int:
        """Get total token dimension (flattened)."""
        return self.token_size_time * self.token_size_freq

    @property
    def num_freq_tokens(self) -> int:
        """Get number of tokens in frequency dimension."""
        if self.input_type == "patch":
            return self.n_mels // self.patch_size_freq
        else:  # frame
            return 1  # Only one token covers full frequency range



class MAEAST(nn.Module):
    def __init__(self, config: MAEASTConfig):
        super().__init__()

        self.config = config

        # Use input_type to determine tokenization
        self.num_freq_tokens = config.num_freq_tokens
        self.token_dim = config.token_dim

        logger.info(f"Initializing MAE-AST model")
        logger.info(f"  Input type: {config.input_type}")
        logger.info(f"  Input: {config.n_mels} mel bins")

        if config.input_type == "patch":
            logger.info(f"  Patch size: {config.patch_size_freq}×{config.patch_size_time}")
            logger.info(f"  Token dimension: {self.token_dim}")
        else:  # frame
            logger.info(f"  Frame size: {config.n_mels}×{config.frame_size_time}")
            logger.info(f"  Token dimension: {self.token_dim}")

        logger.info(f"  Encoder: {config.encoder_depth} layers, {config.encoder_embed_dim} dim")
        logger.info(f"  Decoder: {config.decoder_depth} layers, {config.decoder_embed_dim} dim")
        logger.info(f"  Mask ratio: {config.mask_ratio}")

        # ==================== Input Processing ====================

        # Patchification/Framification using Unfold
        # split(depends) into 16 filter × 16 frame tokens
        # Questo frammento di codice trasforma il tuo spettrogramma audio in una "lista" di piccoli pezzi (patch) pronti per essere mascherati e poi elaborati dal Transformer.
        self.unfold = nn.Unfold(
            kernel_size=(config.token_size_freq, config.token_size_time),
            stride=(config.token_size_freq, config.token_size_time)
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm2d(num_features=1, affine=False)

        # Linear projection: token_dim → encoder_embed_dim
        self.patch_embed = nn.Linear(self.token_dim, config.encoder_embed_dim)

        # ==================== Masking ====================
        # Initialize mask generator based on config
        if config.mask_type == 'random':
            self.mask_generator = RandomMasking(
                mask_ratio=config.mask_ratio,
                batched=config.mask_batched,
            )
        elif config.mask_type == 'chunk':
            self.mask_generator = ChunkedMasking(
                mask_ratio=config.mask_ratio,
                chunk_size_range=config.chunk_size_range,
                num_freq_patches=self.num_freq_tokens,
                batched=config.mask_batched,
            )
        else:
            raise ValueError(f"Unknown mask type: {config.mask_type}")

        # Learnable mask token embeddings
        self.encoder_mask_token = nn.Parameter(torch.zeros(1, 1, config.encoder_embed_dim))
        self.decoder_mask_token = nn.Parameter(torch.zeros(1, 1, config.encoder_embed_dim))

        # Initialize mask tokens
        nn.init.normal_(self.encoder_mask_token, std=0.02)
        nn.init.normal_(self.decoder_mask_token, std=0.02)

        # ==================== Positional Encoding ====================
        # Encoder positional encoding
        self.encoder_pos_embed = SinusoidalPositionalEncoding(
            d_model=config.encoder_embed_dim,
            max_len=config.max_position,
            dropout=0.0
        )
        # Decoder positional encoding
        self.decoder_pos_embed = SinusoidalPositionalEncoding(
            d_model=config.decoder_embed_dim,
            max_len=config.max_position,
            dropout=0.0
        )

        # ==================== Transformer Encoder ====================
        self.encoder = TransformerEncoder(
            num_layers=config.encoder_depth,
            embed_dim=config.encoder_embed_dim,
            num_heads=config.encoder_num_heads,
            ffn_dim=config.encoder_ffn_dim,
            dropout=config.dropout,
            layer_norm_first=config.layer_norm_first,
        )

        # ==================== Transformer Decoder ====================
        self.decoder = TransformerEncoder(
            num_layers=config.decoder_depth,
            embed_dim=config.decoder_embed_dim,
            num_heads=config.decoder_num_heads,
            ffn_dim=config.decoder_ffn_dim,
            dropout=config.dropout,
            layer_norm_first=config.layer_norm_first,
        )

        # ==================== Prediction Heads ====================

        # Reconstruction head: decoder_dim → patch_dim
        # Paper: "single-layer linear projections"
        self.reconstruction_head = nn.Linear(
            config.decoder_embed_dim,
            self.token_dim
        )

        # Contrastive head: decoder_dim → patch_dim
        # Used for InfoNCE loss
        self.contrastive_head = nn.Linear(
            config.decoder_embed_dim,
            self.token_dim
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Log total parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert spectrogram to tokens (patches or frames).

        Paper:
        - Patch-based: "16 filter × 16 frame tokens"
        - Frame-based: "128 filter × 2 frame tokens"

        Args:
            x: Spectrogram [batch, 1, n_mels, time]

        Returns:
            tokens: [batch, num_tokens, token_dim]
        """
        # Apply batch normalization (mean=0, std=0.5)
        x = self.batch_norm(x) * 0.5

        # Unfold: [batch, 1, n_mels, time] → [batch, token_dim, num_tokens]
        tokens = self.unfold(x)

        # Transpose: [batch, token_dim, num_tokens] → [batch, num_tokens, token_dim]
        tokens = tokens.transpose(1, 2)

        return tokens

    def forward_encoder(self,
                        x: torch.Tensor,
                        padding_mask: Optional[torch.Tensor] = None,
                        apply_mask: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through encoder only.

        Used during fine-tuning (no masking, no decoder).

        Args:
            x: Input spectrogram [batch, n_mels, time]
            padding_mask: Padding mask [batch, time]
            apply_mask: Whether to apply masking (False during fine-tuning)

        Returns:
            encoder_output: [batch, num_tokens, embed_dim]
            info: Dict with intermediate values
        """
        batch_size = x.size(0)

        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [batch, 1, n_mels, time]

        # Tokenize (works for both patch and frame)
        tokens = self.tokenize(x)  # [batch, num_tokens, token_dim]
        num_tokens = tokens.size(1)

        # Project to embedding space
        x = self.patch_embed(tokens)  # [batch, num_tokens, embed_dim]

        # Apply positional encoding BEFORE masking
        if self.encoder_pos_embed is not None:
            if padding_mask is not None:
                # Convert time-based padding to token-based
                token_size_time = self.config.token_size_time
                num_time_tokens = num_tokens // self.num_freq_tokens

                # Sample padding mask at token boundaries
                token_padding_mask = padding_mask[:, ::token_size_time][:, :num_time_tokens]

                # Expand to all frequency tokens
                token_padding_mask = token_padding_mask.repeat_interleave(
                    self.num_freq_tokens, dim=1
                )
            else:
                token_padding_mask = None

            pos_embed = self.encoder_pos_embed(x, token_padding_mask)
            x = x + pos_embed

        # Masking (only during pretraining)
        retained_indices = None
        masked_indices = None

        if apply_mask:
            # Generate mask
            retained_indices, masked_indices, num_masked = self.mask_generator.generate_mask(
                batch_size, num_tokens, x.device
            )

            # Extract only unmasked tokens
            if self.config.mask_batched:
                x = x[:, retained_indices[0]]
                if token_padding_mask is not None:
                    token_padding_mask = token_padding_mask[:, retained_indices[0]]
            else:
                x_list = []
                mask_list = []
                for i in range(batch_size):
                    x_list.append(x[i, retained_indices[i]])
                    if token_padding_mask is not None:
                        mask_list.append(token_padding_mask[i, retained_indices[i]])
                x = torch.stack(x_list, dim=0)
                if token_padding_mask is not None:
                    token_padding_mask = torch.stack(mask_list, dim=0)

        # Encoder forward pass
        encoder_output = self.encoder(x, token_padding_mask)

        info = {
            'num_tokens': num_tokens,
            'retained_indices': retained_indices,
            'masked_indices': masked_indices,
            'tokens': tokens,
        }

        return encoder_output, info



    def forward_decoder(
            self,
            encoder_output: torch.Tensor,
            info: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through decoder.

        Paper: "We add mask tokens with encoder output embeddings
        before feeding them to a shallow decoder"

        Args:
            encoder_output: Encoder output [batch, num_unmasked, embed_dim]
            info: Dict from forward_encoder with masking info

        Returns:
            reconstruction_pred: [batch, num_masked, patch_dim]
            contrastive_pred: [batch, num_masked, patch_dim]
        """
        batch_size = encoder_output.size(0)
        num_tokens = info['num_tokens']
        retained_indices = info['retained_indices']
        masked_indices = info['masked_indices']

        # Reconstruct full sequence with mask tokens
        # concatenate mask tokens with encoder output embeddings
        full_sequence = torch.zeros(
            batch_size, num_tokens, self.config.decoder_embed_dim,
            device=encoder_output.device, dtype=encoder_output.dtype
        )

        mask_indicators = torch.zeros(batch_size, num_tokens, device=encoder_output.device, dtype=torch.bool)

        if self.config.mask_batched:
            # Same mask for entire batch
            full_sequence[:, retained_indices[0]] = encoder_output
            full_sequence[:, masked_indices[0]] = self.decoder_mask_token
            mask_indicators[:, masked_indices[0]] = True
        else:
            # Different mask per sample
            for i in range(batch_size):
                full_sequence[i, retained_indices[i]] = encoder_output[i]
                full_sequence[i, masked_indices[i]] = self.decoder_mask_token
                mask_indicators[i, masked_indices[i]] = True

        # Add positional encoding to full sequence
        # We add sinusoidal positional embeddings to all tokens
        # before inputting to the decoder
        if self.decoder_pos_embed is not None:
            pos_embed = self.decoder_pos_embed(full_sequence)
            full_sequence = full_sequence + pos_embed


        # decoder forward pass
        decoder_output = self.decoder(full_sequence)

        # Extract predictions for masked tokens only
        # The output embeddings of masked tokens from the decoder
        # are then sent into two single-layer linear projections
        masked_output = decoder_output[mask_indicators]  # [batch * num_masked, embed_dim]
        num_masked = masked_indices[0].shape[0] if self.config.mask_batched else masked_indices[0].shape[0]
        masked_output = masked_output.view(batch_size, num_masked, -1)

        # Prediction heads
        reconstruction_pred = self.reconstruction_head(masked_output)
        contrastive_pred = self.contrastive_head(masked_output)

        return reconstruction_pred, contrastive_pred


    def forward(self,
               x: torch.Tensor,
               padding_mask: Optional[torch.Tensor] = None,
               return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pretraining.

        Args:
            x: Input spectrogram [batch, n_mels, time]
            padding_mask: Padding mask [batch, time]
            return_features: If True, return encoder features (for fine-tuning)

        Returns:
            Dict with:
                - reconstruction_pred: [batch, num_masked, patch_dim]
                - contrastive_pred: [batch, num_masked, patch_dim]
                - targets: [batch, num_masked, patch_dim]
                - (optional) features: encoder output
        """
        # Encoder forward
        encoder_output, info = self.forward_encoder(x, padding_mask, apply_mask=True)

        if return_features:
            return {'features': encoder_output}

        # Decoder forward
        reconstruction_pred, contrastive_pred = self.forward_decoder(encoder_output, info)

        # Get targets (original patches at masked positions)
        tokens = info['tokens']
        masked_indices = info['masked_indices']

        if self.config.mask_batched:
            targets = tokens[:, masked_indices[0]]
        else:
            targets = torch.stack([
                tokens[i, masked_indices[i]]
                for i in range(tokens.size(0))
            ], dim=0)

        return {'reconstruction_pred': reconstruction_pred,
                'contrastive_pred': contrastive_pred,
                'targets': targets}

    def get_num_params(self, only_trainable: bool = False) -> int:
        """Get number of parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
