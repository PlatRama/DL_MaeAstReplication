import logging
from typing import Optional, Dict

import torch
import torch.nn as nn

from .contrastive_loss import InfoNCELoss
from .reconstruction_loss import ReconstructionLoss

logger = logging.getLogger('mae_ast')


class MAEASTLoss(nn.Module):
    """
    Combined loss for MAE-AST pretraining.
    Combines two objectives:
    1. Generative (Reconstruction): Pixel-level MSE loss
    2. Discriminative (Contrastive): Embedding-level InfoNCE loss

    The reconstruction loss is weighted by λ = 10 to balance the scales.

    Formula:
        loss = λ * MSE(reconstruction_pred, targets)
             + InfoNCE(contrastive_pred, targets)

    Args:
        reconstruction_weight: λ parameter
        contrastive_weight: Weight for contrastive loss
        temperature: Temperature for InfoNCE (default: 0.07)
        use_reconstruction: Whether to use reconstruction loss
        use_contrastive: Whether to use contrastive loss
    """

    def __init__(
            self,
            reconstruction_weight: float = 10.0,
            contrastive_weight: float = 1.0,
            temperature: float = 0.07,
            use_reconstruction: bool = True,
            use_contrastive: bool = True
    ):
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight
        self.use_reconstruction = use_reconstruction
        self.use_contrastive = use_contrastive

        # Initialize loss functions
        if use_reconstruction:
            self.reconstruction_loss = ReconstructionLoss(reduction='mean')

        if use_contrastive:
            self.contrastive_loss = InfoNCELoss(
                temperature=temperature,
                reduction='mean'
            )

        logger.info(f"MAEASTLoss initialized:")
        logger.info(f"  Reconstruction weight (lambda): {reconstruction_weight}")
        logger.info(f"  Contrastive weight: {contrastive_weight}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Use reconstruction: {use_reconstruction}")
        logger.info(f"  Use contrastive: {use_contrastive}")

        # Sanity check
        if not use_reconstruction and not use_contrastive:
            raise ValueError("At least one loss must be enabled!")

    def forward(
            self,
            reconstruction_pred: torch.Tensor,
            contrastive_pred: torch.Tensor,
            targets: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            reconstruction_pred: Predictions for reconstruction [batch, num_masked, patch_dim]
            contrastive_pred: Predictions for contrastive [batch, num_masked, patch_dim]
            targets: Ground truth patches [batch, num_masked, patch_dim]
            mask: Optional mask for valid positions [batch, num_masked]

        Returns:
            Dict with:
                - 'loss': Total loss
                - 'reconstruction_loss': Reconstruction component
                - 'contrastive_loss': Contrastive component
        """
        losses = {}
        total_loss = 0.0

        # Reconstruction loss (MSE)
        if self.use_reconstruction:
            recon_loss = self.reconstruction_loss(
                reconstruction_pred,
                targets,
                mask
            )
            losses['reconstruction_loss'] = recon_loss
            total_loss += self.reconstruction_weight * recon_loss

        # Contrastive loss (InfoNCE)
        if self.use_contrastive:
            contrast_loss = self.contrastive_loss(
                contrastive_pred,
                targets
            )
            losses['contrastive_loss'] = contrast_loss
            total_loss += self.contrastive_weight * contrast_loss

        losses['loss'] = total_loss

        return losses

    def forward_from_model_output(
            self,
            model_output: Dict[str, torch.Tensor],
            mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss directly from model output dict.

        Convenience method for training loop.

        Args:
            model_output: Dict from model.forward() with keys:
                - 'reconstruction_pred'
                - 'contrastive_pred'
                - 'targets'
            mask: Optional mask

        Returns:
            Dict with loss components
        """
        return self.forward(
            reconstruction_pred=model_output['reconstruction_pred'],
            contrastive_pred=model_output['contrastive_pred'],
            targets=model_output['targets'],
            mask=mask
        )