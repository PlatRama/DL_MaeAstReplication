import logging

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger('mae_ast')

class InfoNCELoss(nn.Module):
    """
    InfoNCE (Information Noise Contrastive Estimation) loss.

    Formula:
        For each masked token i with prediction p_i and target t_i:

        similarity(i, j) = dot(p_i, t_j)
        logits = [similarity(i, 0), similarity(i, 1), ..., similarity(i, N-1)]
        loss = -log(softmax(logits)[i])

    Where:
    - i is the index of current masked token
    - j ranges over all masked tokens in same sample
    - Positive: j = i (same token)
    - Negatives: j ≠ i (other masked tokens)

    Args:
        temperature: Temperature parameter for softmax (default: 0.07)
        reduction: 'mean' or 'sum'
    """

    def __init__(self,
                 temperature=0.07,
                 reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

        logger.info(f"InfoNCELoss: temperature={temperature}, reduction={reduction}")

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Paper Algorithm:
        1. Compute similarity matrix: predictions @ targets^T
        2. Apply temperature scaling
        3. Compute log softmax
        4. Extract diagonal (positive pairs)
        5. Return negative mean

        Args:
            predictions: Model predictions [batch, num_masked, patch_dim]
            targets: Ground truth patches [batch, num_masked, patch_dim]

        Returns:
            loss: Scalar loss value
        """

        batch_size, num_masked, patch_dim = predictions.shape

        # Normalize embeddings
        predictions = F.normalize(predictions, dim=-1, p=2)
        targets = F.normalize(targets, dim=-1, p=2)

        # Compute similarity matrix for each sample in batch
        # [batch, num_masked, patch_dim] @ [batch, patch_dim, num_masked]
        # -> [batch, num_masked, num_masked]
        similarity_matrix = torch.matmul(predictions, targets.transpose(-1, -2)) / self.temperature

        # we obtain negative samples from other
        # masked inputs within the same audio segment
        # This means negatives are other masked tokens in SAME sample,
        # not across batch (avoids batch size dependency)

        # Apply log softmax along last dimension
        # For each query (row), compute softmax over all keys (columns)
        log_probs = F.log_softmax(similarity_matrix, dim=-1)

        # Extract diagonal elements (positive pairs)
        # These are the log probabilities of matching each prediction
        # with its corresponding target
        positive_log_probs = torch.diagonal(log_probs, dim1=-2, dim2=-1)

        # Loss is negative log likelihood
        loss = -positive_log_probs

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class ContrastiveLoss(nn.Module):
    """
    Alternative contrastive loss with hard negative mining.

    This is NOT used in the paper but is a more advanced variant
    that focuses on hard negatives (most confusing examples).

    Args:
        temperature: Temperature for softmax
        negative_mode: 'all' or 'hard' (top-k hardest negatives)
        top_k: Number of hard negatives to use
    """

    def __init__(self,
                 temperature: float=0.07,
                 negative_mode: str='all',
                 top_k: int =10):
        super().__init__()
        self.temperature = temperature
        self.negative_mode = negative_mode
        self.top_k = top_k

        logger.info(f"ContrastiveLoss: temperature={temperature}, "
                    f"negative_mode={negative_mode}, top_k={top_k}")


    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss with optional hard negative mining.

        Args:
            predictions: [batch, num_masked, patch_dim]
            targets: [batch, num_masked, patch_dim]

        Returns:
            loss: Scalar loss
        """

        batch_size, num_masked, patch_dim = predictions.shape

        # Normalize
        predictions = F.normalize(predictions, dim=-1, p=2)
        targets = F.normalize(targets, dim=-1, p=2)

        # Similarity matrix
        similarity = torch.matmul(predictions, targets.transpose(-1, -2)) / self.temperature

        if self.negative_mode == 'all':
            # Use all negatives (same as InfoNCE)
            log_probs = F.log_softmax(similarity, dim=-1)
            positive_log_probs = torch.diagonal(log_probs, dim1=-2, dim2=-1)
            loss = -positive_log_probs.mean()
        elif self.negative_mode == 'hard':
            # Hard negative mining
            # For each query, select top-k hardest negatives (highest similarity)

            # mask out positive pairs
            mask = torch.eye(num_masked, device=similarity.device).bool()
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            similarity_negatives = similarity.masked_fill(mask, float('-inf'))

            # Get top-k hardest negatives
            hard_negatives, _ = torch.topk(
                similarity_negatives, k=min(self.top_k, num_masked - 1), dim=-1
            )

            # Positive similarities
            positive_sim = torch.diagonal(similarity, dim1=-2, dim2=-1).unsqueeze(-1)

            # Combine positive and hard negatives
            logits = torch.cat([positive_sim, hard_negatives], dim=-1)

            # Loss: positive should have highest similarity
            labels = torch.zeros(batch_size, num_masked, dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        else:
            raise ValueError(f"Unknown negative_mode: {self.negative_mode}")

        return loss


class SimCLRLoss(nn.Module):
    """
    SimCLR-style contrastive loss (cross-batch negatives).

    NOT used in paper. This variant uses negatives across the batch,
    making it more similar to standard SimCLR.

    Args:
        temperature: Temperature parameter
    """

    def __init__(self, temperature: float=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute SimCLR loss with cross-batch negatives.

        Args:
            predictions: [batch, num_masked, patch_dim]
            targets: [batch, num_masked, patch_dim]

        Returns:
            loss: Scalar loss
        """
        batch_size, num_masked, patch_dim = predictions.shape

        # Flatten batch and masked dimensions
        predictions = predictions.view(-1, patch_dim)
        targets = targets.view(-1, patch_dim)
        N = predictions.size(0)

        # normalize
        predictions = F.normalize(predictions, dim=-1, p=2)
        targets = F.normalize(targets, dim=-1, p=2)

        # Full similarity matrix (including cross-batch)
        similarity = torch.matmul(predictions, targets.T) / self.temperature

        # Labels: diagonal elements are positives
        labels = torch.arange(N, device=similarity.device)

        # cross-entropy loss
        loss = F.cross_entropy(similarity, labels)

        return loss




