import logging

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger('mae_ast')

class ReconstructionLoss(nn.Module):
    """
    Mean Squared Error (MSE) reconstruction loss.

    Implementation:
    - Prediction: unnormalized output from reconstruction head
    - Target: normalized input patches
    - Loss: MSE between prediction and target

    Why MSE for spectrograms:
    MSE is natural for continuous values like log mel spectrograms.
    It encourages pixel-perfect reconstruction of audio features.

    Args:
        reduction: 'mean', 'sum', or 'none'
        normalize_target: Whether target is already normalized
    """

    def __init__(
            self,
            reduction: str = 'mean',
            normalize_target: bool = True
    ):
        super().__init__()

        self.reduction = reduction
        self.normalize_target = normalize_target

        logger.info(f"ReconstructionLoss: reduction={reduction}, "
                    f"normalize_target={normalize_target}")

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Args:
            predictions: Model predictions [batch, num_masked, patch_dim]
            targets: Ground truth patches [batch, num_masked, patch_dim]
            mask: Optional mask [batch, num_masked] to ignore certain positions

        Returns:
            loss: Scalar loss value
        """
        # mean-squared error between the unnormalized output
        # of the linear reconstruction head and the normalized input

        # Compute MSE
        loss = F.mse_loss(predictions, targets, reduction='none')
        # loss: [batch, num_masked, patch_dim]

        # Apply mask if provided (useful for variable-length sequences)
        if mask is not None:
            # mask: [batch, num_masked] -> [batch, num_masked, 1]
            loss = loss * mask.unsqueeze(-1)

            # Normalize by number of valid positions
            if self.reduction == 'mean':
                loss = loss.sum() / mask.sum().clamp(min=1)
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            # Standard reduction
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        return loss