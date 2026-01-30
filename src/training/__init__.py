"""
Training components for MAE-AST.

This module provides:
- Trainer: Main training loop with validation
- Optimizer builders: Adam with weight decay
- Learning rate schedulers: Polynomial decay, cosine, etc.
- Training utilities

Paper Training Setup (Section 2.1):
- Optimizer: Adam with weight_decay=0.01
- Learning rate: 0.0001 (1e-4)
- Scheduler: Polynomial decay
- Batch size: ~32 (depends on GPU memory)
- Training steps: 600,000 iterations (~2 days on RTX-8000)
"""

from .trainer import MAEASTTrainer
from .optimizer import build_optimizer
from .scheduler import build_scheduler, PolynomialDecayLR, CosineAnnealingWarmupLR

__all__ = [
    'MAEASTTrainer',
    'build_optimizer',
    'build_scheduler',
    'PolynomialDecayLR',
    'CosineAnnealingWarmupLR'
]