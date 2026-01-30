"""
Logging utilities for training.

Provides:
1. Setup for Python logging
2. MetricLogger for tracking metrics during training
3. TensorBoard integration
4. WandB integration (optional)

Paper Context:
The paper trains for 600,000 iterations with logging every N steps.
These utilities help track loss, learning rate, and other metrics.
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from collections import defaultdict, deque

import torch
from torch.utils.tensorboard import SummaryWriter

import wandb
import time


def setup_logger(
        name: str = "mae_ast",
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logger with console and file handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_str: Custom format string

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    logger.handlers.clear()

    if format_str is None:
        format_str = '[%(asctime)s] [%(levelname)s] %(message)s'
    formatter = logging.Formatter(format_str)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_path}")

    return logger

class MetricLogger:
    """
    Logger for tracking multiple metrics during training.

    Features:
    - Track multiple metrics with moving averages
    - Smoothing with deque window
    - Synchronized across distributed training
    """

    def __init__(self, delimiter: str = " ", window_size: int = 20):
        """
        Args:
            delimiter: String to join metric strings
            window_size: Size of moving average window
        """
        self.delimiter = delimiter
        self.meters = defaultdict(lambda : SmoothedValue(window_size))

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def get_global_avg(self) -> Dict[str, float]:
        return {name: meter.global_avg for name, meter in self.meters.items()}

    def get_current(self) -> Dict[str, float]:
        """Get current value of all metrics."""
        return {name: meter.value for name, meter in self.meters.items()}

class SmoothedValue:
    """
    Track a series of values and provide smoothed average.

    Uses deque for efficient sliding window computation.
    """

    def __init__(self, window_size: int = 20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def value(self) -> float:
        return self.deque[-1] if len(self.deque) > 0 else 0.0

    def __str__(self):
        return f"{self.median:.4f} ({self.global_avg:.4f})"


class TensorBoardLogger:
    """
    TensorBoard logging wrapper.

    Paper Context:
    Logs training metrics for visualization:
    - Loss curves (total, reconstruction, contrastive)
    - Learning rate schedule
    - Gradient norms
    - Validation metrics
    """

    def __init__(self, log_dir: str, enabled: bool = True):
        self.writer = None
        self.enabled = enabled

        if enabled:
            try:
                log_path  = Path(log_dir)
                log_path.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(log_path))
                logging.info(f"TensorBoard logging to: {log_dir}")
            except ImportError:
                logging.warning("TensorBoard not available. Install with: pip install tensorboard")
                self.enabled = False


    def add_scalar(self, tag: str, value: float, step: int):
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars under same tag."""
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def add_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram of values."""
        if self.enabled and self.writer is not None:
            self.writer.add_histogram(tag, values, step)

    def add_image(self, tag: str, img: torch.Tensor, step: int):
        """Log image."""
        if self.enabled and self.writer is not None:
            self.writer.add_image(tag, img, step)

    def add_figure(self, tag: str, figure, step: int):
        """Log matplotlib figure."""
        if self.enabled and self.writer is not None:
            self.writer.add_figure(tag, figure, step)

    def flush(self):
        """Flush pending logs."""
        if self.enabled and self.writer is not None:
            self.writer.flush()

    def close(self):
        """Close writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()

class WandBLogger:
    """
    Weights & Biases logging wrapper (optional).

    Provides more advanced experiment tracking than TensorBoard:
    - Hyperparameter sweeps
    - Model versioning
    - Collaborative features
    """

    def __init__(self,
                 project: str,
                 name: Optional[str] = None,
                 config: Optional[Dict] = None,
                 enabled: bool = False):
        """
        Args:
            project: WandB project name
            name: Run name
            config: Configuration dict to log
            enabled: Whether WandB is enabled
        """
        self.enabled = enabled
        self.run = None

        if enabled:
            try:
                self.run = wandb.init(project=project, name=name, config=config)
                logging.info(f"WandB logging enabled for project: {project}")
            except ImportError:
                logging.warning("WandB not available. Install with: pip install wandb")
                self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        if self.enabled and self.run is not None:
            wandb.log(metrics, step=step)

    def watch(self, model: torch.nn.Module):
        """Watch model gradients."""
        if self.enabled and self.run is not None:
            wandb.watch(model)

    def finish(self):
        """Finish logging with wandb."""
        if self.enabled and self.run is not None:
            wandb.finish()

class ProgressLogger:
    """
    Progress logger for training/validation loops.

    Displays progress with ETA and metrics.
    """

    def __init__(
            self,
            total_steps: int,
            desc: str = "Training",
            log_every: int = 100
    ):
        self.total_steps = total_steps
        self.desc = desc
        self.log_every = log_every
        self.start_time = time.time()
        self.logger = logging.getLogger("mae_ast")

    def log(self, step: int, metrics: Dict[str, float]):
        """
        Log progress.

        Args:
            step: Current step
            metrics: Dict of metrics to log
        """
        if step % self.log_every != 0 and step != self.total_steps:
            return

        # Calculate progress
        progress = step / self.total_steps * 100
        elapsed = time.time() - self.start_time

        # Estimate time remaining
        if step > 0:
            eta = elapsed / step * (self.total_steps - step)
            eta_str = self._format_time(eta)
        else:
            eta_str = "N/A"

        # Format metrics
        metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        metric_str = ", ".join(metric_strs)

        # Log
        self.logger.info(
            f"{self.desc} [{step}/{self.total_steps}] ({progress:.1f}%) "
            f"| {metric_str} | ETA: {eta_str}"
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

