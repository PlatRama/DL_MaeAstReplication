import logging
from typing import Dict, Optional, List

import numpy as np
import torch

logger = logging.getLogger('mae_ast')

class AverageMeter:
    """
    Compute and store average and current value.

    Useful for tracking metrics over batches.
    """
    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self):
        return f"{self.name}: {self.val:.4f} ({self.avg:.4f})"

class MetricsTracker:
    """
    Track multiple metrics during training/evaluation.

    Automatically handles multiple AverageMeters.
    """

    def __init__(self):
        self.metrics = {}

    def update(self, metrics_dict: Dict[str, float], n: int=1):
        """
        Update metrics.

        Args:
            metrics_dict: Dict of metric_name -> value
            n: Batch size
        """
        for name, value in metrics_dict.items():
            if name not in self.metrics:
                self.metrics[name] = AverageMeter(name)

            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[name].update(value, n)


    def get(self, name: str) -> Optional[AverageMeter]:
        return self.metrics.get(name, None)

    def get_averages(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.metrics.items()}

    def get_current(self) -> Dict[str, float]:
        """Get current value of all metrics."""
        return {name: meter.val for name, meter in self.metrics.items()}

    def reset(self):
        """Reset all metrics."""
        for meter in self.metrics.values():
            meter.reset()

    def __str__(self):
        return " | ".join(str(meter) for meter in self.metrics.values())

def compute_accuracy(
        output: torch.Tensor,
        target: torch.Tensor,
        topk: tuple = (1,5)
) -> List[float]:
    """
    Compute top-k accuracy.

    Args:
        output: Model predictions [batch, num_classes]
        target: Ground truth labels [batch]
        topk: Tuple of k values

    Returns:
        List of top-k accuracies
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions
        _, pred = output.topk(maxk, dim= 1, largest=True, sorted=True)
        pred = pred.t() # [maxk, batch]
        # Check if predictions match target
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # Compute accuracy for each k
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def compute_mAP(
        predictions: np.ndarray,
        targets: np.ndarray,
        top_k: Optional[int] = None
) -> float:
    """
    Compute mean Average Precision (mAP).

    Used for multi-label classification (AudioSet).

    Args:
        predictions: [num_samples, num_classes] prediction scores
        targets: [num_samples, num_classes] binary labels
        top_k: If specified, only consider top-k predictions

    Returns:
        mAP score
    """

    num_samples, num_classes = predictions.shape

    aps = []
    for i in range(num_samples):
        pred = predictions[i]
        target = targets[i]

        if target.sum() == 0:
            continue

        sorted_indices = np.argsort(-pred)

        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]

        tp = 0
        fp = 0
        precision_at_k = []

        for k, idx in enumerate(sorted_indices, 1):
            if target[idx] == 1:
                tp += 1
                precision_at_k.append(tp / k)
            else:
                fp += 1

        if len(precision_at_k) > 0:
            ap = np.mean(precision_at_k)
            aps.append(ap)

    return np.mean(aps) if len(aps) > 0 else 0.0

def compute_auc(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Compute Area Under ROC Curve (AUC).

    Args:
        predictions: [num_samples, num_classes] prediction scores
        targets: [num_samples, num_classes] binary labels

    Returns:
        AUC score
    """
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(targets, predictions, average='macro')
    except ImportError:
        logger.warning("sklearn not available, cannot compute AUC")
        return 0.0

class ConfusionMatrix:
    """
    Confusion matrix for multi-class classification.

    Useful for analyzing per-class performance.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix.

        Args:
            predictions: [batch] predicted class indices
            targets: [batch] ground truth class indices
        """
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        for pred, target in zip(predictions, targets):
            self.matrix[target, pred] += 1

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute metrics from confusion matrix.

        Returns:
            Dict with accuracy, precision, recall, f1
        """
        # Accuracy
        accuracy = np.diag(self.matrix).sum() / self.matrix.sum()

        # Per-class metrics
        precision = np.diag(self.matrix) / (self.matrix.sum(axis=0) + 1e-10)
        recall = np.diag(self.matrix) / (self.matrix.sum(axis=1) + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        return {
            'accuracy': accuracy,
            'precision': precision.mean(),
            'recall': recall.mean(),
            'f1': f1.mean()
        }

    def reset(self):
        """Reset confusion matrix."""
        self.matrix.fill(0)

    def __str__(self):
        return f"ConfusionMatrix:\n{self.matrix}"