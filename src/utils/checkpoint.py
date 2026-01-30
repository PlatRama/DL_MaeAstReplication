import logging
import shutil
from pathlib import Path

from typing import Dict, Any, Optional

import torch
from torch import nn

logger = logging.getLogger('mae_ast')

def save_checkpoint(
        state: Dict[str, Any],
        filepath: str,
        is_best: bool = False,
        max_keep: int = 5
):
    """
    Save checkpoint to disk.

    Args:
        state: Dict containing model state, optimizer state, etc.
        filepath: Path to save checkpoint
        is_best: If True, also save as 'best.pt'
        max_keep: Maximum number of checkpoints to keep
    """

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    #save checkpoint
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

    if is_best:
        best_path = filepath.parent/"best.pt"
        shutil.copyfile(filepath, best_path)
        logger.info(f"Best checkpoint saved to {best_path}")

    # Remove old checkpoints (keep only max_keep most recent)
    if max_keep > 0:
        checkpoints = sorted(filepath.parent.glob("checkpoint_*.pt"), key= lambda x: x.stat().st_mtime)

        for old_ckpt in checkpoints[:-max_keep]:
            old_ckpt.unlink()
            logger.info(f"Removing old checkpoint {old_ckpt}")


def load_checkpoint(
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        strict: bool = True,
) -> Dict[str, Any]:
    """
    Load checkpoint from disk.

    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load tensors to
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Dict with metadata (epoch, step, metrics, etc.)
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    logger.info(f"Loading checkpoint from {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        logger.info("Model state loaded")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Optimizer state loaded")

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Scheduler state loaded")

    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'best_metric': checkpoint.get('best_metric', float('inf')),
    }
    logger.info(f"Checkpoint loaded: epoch={metadata['epoch']}, step={metadata['step']}")
    return metadata

class CheckpointManager:
    """
    Manager for saving and loading checkpoints.

    Features:
    - Auto-save at regular intervals
    - Keep best N checkpoints based on metric
    - Resume from latest checkpoint
    - Clean up old checkpoints
    """

    def __init__(self,
                 checkpoint_dir: str,
                 model: nn.Module,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[Any] = None,
                 max_keep: int = 5,
                 save_every: int = 10000,
                 metric_name: str = 'loss',
                 mode: str = 'min'):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            scheduler: Scheduler to checkpoint
            max_keep: Maximum number of checkpoints to keep
            save_every: Save checkpoint every N steps
            metric_name: Metric to track for best checkpoint
            mode: 'min' or 'max' for metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_keep = max_keep
        self.save_every = save_every
        self.metric_name = metric_name
        self.mode = mode

        # Track best metric
        self.best_metric = float('inf') if mode == 'min' else float('-inf')

        logger.info(f"CheckpointManager initialized: {checkpoint_dir}")
        logger.info(f"Save every: {save_every} steps, Keep: {max_keep} checkpoints")


    def save(self,
             step: int,
             epoch: int,
             metrics: Dict[str, float],
             is_scheduled: bool = False):

        if is_scheduled and step % self.save_every != 0:
            return

        state = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
        }

        if self.optimizer is not None:
            state['optimizer_state_dict'] = self.optimizer.state_dict()

        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()


        current_metric = metrics.get(self.metric_name, None)
        is_best = False

        if current_metric is not None:
            if self.mode == 'min':
                is_best = current_metric < self.best_metric
            else:
                is_best = current_metric > self.best_metric

            if is_best:
                self.best_metric = current_metric
                state['best_metric'] = self.best_metric

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:08d}.pt"
        save_checkpoint(state, str(checkpoint_path), is_best, self.max_keep)

        # Also save as latest
        latest_path = self.checkpoint_dir / "latest.pt"
        shutil.copyfile(checkpoint_path, latest_path)


    def load_latest(self, device: str = 'cuda') -> Optional[Dict[str, Any]]:
        """
        Load latest checkpoint.

        Returns:
            Metadata dict if checkpoint exists, None otherwise
        """

        latest_path = self.checkpoint_dir / "latest.pt"

        if not latest_path.exists():
            logger.info("No checkpoint found, starting from scratch")
            return None

        metadata = load_checkpoint(str(latest_path),
                                   self.model,
                                   self.optimizer,
                                   self.scheduler,
                                   device)
        self.best_metric = metadata.get('best_metric', self.best_metric)
        return metadata

    def load_best(self, device: str = 'cuda') -> Optional[Dict[str, Any]]:
        """
        Load best checkpoint.

        Returns:
            Metadata dict if checkpoint exists, None otherwise
        """
        best_path = self.checkpoint_dir / "best.pt"

        if not best_path.exists():
            logger.warning("No best checkpoint found")
            return None

        metadata = load_checkpoint(
            str(best_path),
            self.model,
            self.optimizer,
            self.scheduler,
            device
        )

        return metadata

    def load_specific(self, step: int, device: str = 'cuda') -> Dict[str, Any]:
        """Load checkpoint from specific step."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:08d}.pt"

        return load_checkpoint(
            str(checkpoint_path),
            self.model,
            self.optimizer,
            self.scheduler,
            device
        )