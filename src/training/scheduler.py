import logging
import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger('mae_ast')

class PolynomialDecayLR(_LRScheduler):
    """
       Polynomial decay learning rate scheduler.

       Formula:
           lr = (base_lr - end_lr) * (1 - step/total_steps)^power + end_lr

       With warmup:
           if step < warmup_steps:
               lr = base_lr * (step / warmup_steps)
           else:
               lr = polynomial_decay(step - warmup_steps)

       Args:
           optimizer: Optimizer
           total_steps: Total training steps (default: 600,000 as in paper)
           warmup_steps: Warmup steps (default: 10,000)
           end_lr: Final learning rate (default: 0.0)
           power: Polynomial power (default: 1.0 for linear decay)
           last_epoch: Last epoch for resuming
       """
    def __init__(self,
                 optimizer: Optimizer,
                 total_steps: int = 600000,
                 warmup_steps: int = 10000,
                 end_lr: float = 0.0,
                 power: float = 1.0,
                 last_epoch: int = -1):

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.end_lr = end_lr
        self.power = power

        super().__init__(optimizer, last_epoch)

        logger.info(f"PolynomialDecayLR scheduler:")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  End LR: {end_lr}")
        logger.info(f"  Power: {power}")

    def get_lr(self):
        """Compute learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Warmup: linear increase
            warmup_factor = step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            decay_step = step - self.warmup_steps
            decay_total = self.total_steps - self.warmup_steps

            if decay_step >= decay_total:
                return [self.end_lr for _ in self.base_lrs]
            decay_factor = (1.0 - decay_step / decay_total) ** self.power
            return [
                (base_lr - self.end_lr) * decay_factor + self.end_lr
                for base_lr in self.base_lrs
            ]

class CosineAnnealingWarmupLR(_LRScheduler):
    """
    Cosine annealing with warmup (alternative scheduler).

    Formula:
        Warmup: lr = base_lr * (step / warmup_steps)
        Cosine: lr = end_lr + (base_lr - end_lr) *
                     (1 + cos(π * (step - warmup) / total)) / 2

    Args:
        optimizer: Optimizer
        total_steps: Total training steps
        warmup_steps: Warmup steps
        end_lr: Minimum learning rate
        last_epoch: Last epoch for resuming
    """

    def __init__(
            self,
            optimizer: Optimizer,
            total_steps: int,
            warmup_steps: int = 0,
            end_lr: float = 0.0,
            last_epoch: int = -1
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.end_lr = end_lr

        super().__init__(optimizer, last_epoch)

        logger.info(f"CosineAnnealingWarmupLR scheduler:")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  End LR: {end_lr}")

    def get_lr(self):
        """Compute learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Warmup
            warmup_factor = step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)

            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

            return [
                self.end_lr + (base_lr - self.end_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]



def build_scheduler(optimizer: Optimizer,
                    scheduler_name: str = 'polynomial',
                    total_steps: int = 60000,
                    warmup_steps: int = 10000,
                    **kwargs) -> _LRScheduler:
    """
    Build learning rate scheduler.

    Paper uses polynomial decay scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_name: 'polynomial', 'cosine', 'step', or 'constant'
        total_steps: Total training steps
        warmup_steps: Warmup steps
        **kwargs: Additional scheduler-specific arguments

    Returns:
        Scheduler instance
    """
    if scheduler_name == 'polynomial':
        scheduler = PolynomialDecayLR(optimizer,
                                      total_steps=total_steps,
                                      warmup_steps=warmup_steps,
                                      end_lr=kwargs.get('end_lr', 0.0),
                                      power=kwargs.get('power', 1.0))

    elif scheduler_name.lower() == 'cosine':
        scheduler = CosineAnnealingWarmupLR(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            end_lr=kwargs.get('end_lr', 0.0)
        )

    elif scheduler_name == 'step':
        step_size = kwargs.get('step_size', 100000)
        gamma = kwargs.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        logger.info(f"StepLR scheduler: step_size={step_size}, gamma={gamma}")

    elif scheduler_name.lower() == 'constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=total_steps
        )
        logger.info(f"Constant LR scheduler")

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler