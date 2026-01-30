import logging
from typing import Tuple

import torch
from torch import nn
from torch.optim import Optimizer

logger = logging.getLogger('mae_ast')

def build_optimizer(model: nn.Module,
                    optimizer_name: str = 'adam',
                    learning_rate: float = 1e-4,
                    weight_decay: float = 0.01,
                    betas: Tuple[float, float] = (0.9, 0.999),
                    eps: float = 1e-8,
                    **kwargs
                    ) -> Optimizer:
    """
    Build optimizer for MAE-AST training.
    Args:
        model: Model to optimize
        optimizer_name: 'adam', 'adamw', or 'sgd'
        learning_rate: Initial learning rate (default: 1e-4 as in paper)
        weight_decay: Weight decay (default: 0.01 as in paper)
        betas: Adam betas (default: (0.9, 0.999))
        eps: Adam epsilon (default: 1e-8)
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Optimizer instance
    """
    parameters = model.parameters()

    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(parameters,
                                     lr = learning_rate,
                                     weight_decay = weight_decay,
                                     betas = betas,
                                     eps = eps)
        logger.info(f"Adam optimizer: lr={learning_rate}, weight_decay={weight_decay}")
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(parameters,
                                      lr=learning_rate,
                                      weight_decay=weight_decay,
                                      betas=betas,
                                      eps=eps)
        logger.info(f"AdamW optimizer: lr={learning_rate}, weight_decay={weight_decay}")
    elif optimizer_name.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        optimizer = torch.optim.SGD(parameters,
                                    lr=learning_rate,
                                    weight_decay=weight_decay,
                                    momentum=momentum)
        logger.info(f"SGD optimizer: lr={learning_rate}, momentum={momentum}")

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer