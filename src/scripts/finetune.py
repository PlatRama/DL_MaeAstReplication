"""
Fine-tune pretrained MAE-AST on downstream tasks.

Supports all tasks from paper:
- AudioSet (AS): Multi-label audio classification
- ESC-50 (ESC): Environmental sounds
- Speech Commands v1/v2 (KS1/KS2): Keyword spotting
- VoxCeleb (SID): Speaker identification
- IEMOCAP (ER): Emotion recognition

Usage:
    python scripts/finetune.py \
        --config configs/finetune_config.yaml \
        --pretrained_checkpoint checkpoints/pretrain_best.pt \
        --task speechcommands_v2
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from ..data import AudioDataset, AudioCollator
from ..models import MAEAST, MAEASTConfig
from ..training import build_optimizer, build_scheduler
from ..utils import (
    setup_logger, set_seed, get_device, print_model_info,
    save_config, MetricLogger, TensorBoardLogger,
    CheckpointManager, compute_accuracy, get_lr, load_config
)

logger = logging.getLogger('mae_ast')


class ClassificationHead(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 768,
            num_classes: int = 527,
            dropout: float = 0.1,
            use_layernorm: bool = True,
            task_type: str = "single-label"
    ):
        super().__init__()

        self.task_type = task_type
        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.norm = nn.LayerNorm(encoder_dim)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder_dim, num_classes)

        # Initialize
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, encoder_output: torch.Tensor):
        """
        Forward pass.

        Args:
            encoder_output: [batch, seq_len, encoder_dim]

        Returns:
            logits: [batch, num_classes]
        """
        # Mean pooling over sequence dimension
        # Durante fine-tuning non c'è masking, quindi facciamo mean pooling semplice
        pooled = encoder_output.mean(dim=1)  # [batch, encoder_dim]

        # Classification
        if self.use_layernorm:
            pooled = self.norm(pooled)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits


class FineTuneModel(nn.Module):
    def __init__(
            self,
            encoder: MAEAST,
            num_classes: int,
            dropout: float = 0.1,
            freeze_encoder: bool = False,
            task_type: str = "single-label"
    ):
        super().__init__()

        self.encoder = encoder
        self.task_type = task_type

        self.classification_head = ClassificationHead(
            encoder_dim=encoder.config.encoder_embed_dim,
            num_classes=num_classes,
            dropout=dropout,
            task_type=task_type
        )

        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen (only training classification head)")
        else:
            logger.info("Encoder unfrozen (training full model)")

    def forward(self, audio: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        Forward pass for classification.

        Args:
            audio: [batch, time, n_mels]
            padding_mask: [batch, time] (optional)

        Returns:
            logits: [batch, num_classes]
        """
        # Extract encoder features (no masking during fine-tuning)
        encoder_output, _ = self.encoder.forward_encoder(
            audio, padding_mask, apply_mask=False
        )

        # Classification
        logits = self.classification_head(encoder_output)

        return logits


def load_pretrained_encoder(checkpoint_path: str, config: dict, task_config: dict, device: torch.device):
    """Load pretrained encoder from checkpoint."""
    logger.info(f"Loading pretrained encoder from: {checkpoint_path}")

    # Get input_type from task config
    input_type = task_config.get('input_type', 'patch')
    logger.info(f"Using input type: {input_type}")

    # Create model config
    model_config = MAEASTConfig(
        n_mels=task_config.get('n_mels',128),
        input_type=input_type,
        patch_size_time=config['model']['patch_size_time'],
        patch_size_freq=config['model']['patch_size_freq'],
        frame_size_time=config['model'].get('frame_size_time', 2),

        encoder_embed_dim=config['model']['encoder_embed_dim'],
        encoder_depth=config['model']['encoder_depth'],
        encoder_num_heads=config['model']['encoder_num_heads'],
        encoder_ffn_dim=config['model']['encoder_ffn_dim'],

        decoder_embed_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_num_heads=config['model']['decoder_num_heads'],
        decoder_ffn_dim=config['model']['decoder_ffn_dim'],

        mask_ratio=config['masking']['mask_ratio'],
        mask_type=config['masking']['mask_type'],
        mask_batched=config['masking'].get('mask_batched', True),
        chunk_size_range=tuple(config['masking'].get('chunk_size_range', [3, 5])),

        use_sinusoidal_pos=config['model']['use_sinusoidal_pos'],

        dropout=config['model']['dropout'],
        layer_norm_first = config['model'].get('layer_norm_first', True)
    )

    # Create model
    model = MAEAST(model_config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f"Loaded pretrained weights from step {checkpoint.get('step', 'unknown')}")
    else:
        model.load_state_dict(checkpoint, strict=True)
        logger.info("Loaded pretrained weights")

    return model


def create_dataloaders(config: dict, task: str):
    """Create dataloaders for fine-tuning task."""

    # Task-specific configuration
    if task not in config['tasks']:
        raise ValueError(f"Task '{task}' not found in config. Available: {list(config['tasks'].keys())}")

    task_config = config['tasks'][task]

    logger.info(f"Task configuration:")
    logger.info(f"  Task type: {task_config.get('task_type', 'single-label')}")
    logger.info(f"  Num classes: {task_config['num_classes']}")
    logger.info(f"  Max audio length: {task_config.get('max_duration', 10.0)}s")
    logger.info(f"  Input type: {task_config.get('input_type', 'patch')}")

    # Training dataset
    logger.info(f"Creating training dataset for {task}...")
    train_dataset = AudioDataset(
        manifest_path=task_config['train_manifest'],
        sample_rate=task_config.get('sample_rate', 16000),
        max_duration=task_config.get('max_duration', 10.0),
        feature_type=task_config.get('feature_type', 'fbank'),
        n_mels=task_config.get('n_mels', 128),
        normalize=task_config.get('normalize', True),
        random_crop=True,
        return_labels=True
    )
    logger.info(f"Training dataset: {len(train_dataset)} samples")

    # Validation dataset
    val_dataset = AudioDataset(
        manifest_path=task_config['val_manifest'],
        sample_rate=task_config.get('sample_rate', 16000),
        max_duration=task_config.get('max_duration', 10.0),
        feature_type=task_config.get('feature_type', 'fbank'),
        n_mels=task_config.get('n_mels', 128),
        normalize=task_config.get('normalize', True),
        random_crop=False,
        return_labels=True
    )
    logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # Collator
    collator = AudioCollator(sample_rate=task_config.get('sample_rate', 16000))

    # Data loaders
    batch_size = task_config.get('batch_size', config['training'].get('batch_size', 32))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        collate_fn=collator,
        pin_memory=config['hardware']['pin_memory'],
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        collate_fn=collator,
        pin_memory=config['hardware']['pin_memory'],
    )

    num_classes = task_config['num_classes']
    task_type = task_config.get('task_type', 'single-label')
    metric = task_config.get('metric', 'accuracy')

    return train_loader, val_loader, num_classes, task_type, metric


def train_epoch(
        model: FineTuneModel,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        epoch: int,
        config: dict,
        task_type: str = "single-label"
):
    """Train for one epoch."""
    model.train()
    metrics = MetricLogger(delimiter="  ")

    log_every = config.get('log_every', 100)

    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        audio = batch['audio'].to(device)
        labels = batch.get('labels', None)
        padding_mask = batch.get('padding_mask', None)

        if labels is None:
            logger.warning("No labels found in batch, skipping...")
            continue

        labels = labels.to(device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(device)

        # Forward pass
        logits = model(audio, padding_mask)

        # Compute loss
        if task_type == "multi-label":
            # AudioSet: multi-label classification with BCE loss
            # Labels should be one-hot encoded
            loss = criterion(logits, labels.float())
        else:
            # Single-label classification with CrossEntropy
            loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.get('max_grad_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['max_grad_norm']
            )

        optimizer.step()

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Compute accuracy (for single-label tasks)
        if task_type == "single-label":
            #acc1, acc5 = compute_accuracy(logits, labels, topk=(1, 5))
            acc1, acc_k = compute_accuracy(logits, labels, topk=(1, min(5, logits.size(-1))))
            metrics.update(loss=loss.item(), acc1=acc1, acc_k=acc_k)
        else:
            metrics.update(loss=loss.item())

        # Logging
        if batch_idx % log_every == 0:
            if task_type == "single-label":
                logger.info(
                    f"Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc@1: {acc1:.2f}% Acc@5: {acc_k:.2f}% "
                    f"LR: {get_lr(optimizer):.2e}"
                )
            else:
                logger.info(
                    f"Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"LR: {get_lr(optimizer):.2e}"
                )

    return metrics.get_global_avg()


@torch.no_grad()
def validate(
        model: FineTuneModel,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        task_type: str = "single-label",
        metric_name: str = "accuracy"
):
    """Validate model."""
    model.eval()
    metrics = MetricLogger(delimiter="  ")

    # For mAP calculation, accumulate all predictions and labels
    if metric_name == "mAP":
        all_predictions = []
        all_labels = []

    for batch in val_loader:
        # Move to device
        audio = batch['audio'].to(device)
        labels = batch.get('labels', None)
        padding_mask = batch.get('padding_mask', None)

        if labels is None:
            continue

        labels = labels.to(device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(device)

        # Forward pass
        logits = model(audio, padding_mask)

        # Compute loss
        if task_type == "multi-label":
            loss = criterion(logits, labels.float())
        else:
            loss = criterion(logits, labels)

        # Compute metrics based on task type and metric name
        if task_type == "single-label":
            #acc1, acc5 = compute_accuracy(logits, labels, topk=(1, 5))
            acc1, acc_k = compute_accuracy(logits, labels, topk=(1, min(5, logits.size(-1))))
            metrics.update(loss=loss.item(), acc1=acc1, acc_k=acc_k)
        else:
            # Multi-label: accumulate for mAP
            metrics.update(loss=loss.item())
            if metric_name == "mAP":
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                all_predictions.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    # Calculate mAP if needed
    result_metrics = metrics.get_global_avg()

    if metric_name == "mAP" and len(all_predictions) > 0:
        import numpy as np
        from ..utils.metrics import compute_mAP

        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_labels)

        mAP = compute_mAP(predictions, targets)
        result_metrics['mAP'] = mAP * 100  # Convert to percentage

    return result_metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MAE-AST on downstream tasks")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to fine-tuning configuration YAML')
    parser.add_argument('--pretrained_checkpoint', type=str, required=True,
                        help='Path to pretrained MAE-AST checkpoint')
    parser.add_argument('--task', type=str, required=True,
                        choices=['audioset', 'esc50', 'speechcommands_v1',
                                 'speechcommands_v2', 'voxceleb', 'iemocap'],
                        help='Downstream task name')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder weights (only train classification head)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    base_log_dir = config['logging'].get('log_dir', 'logs/finetune')
    log_dir = Path(base_log_dir) / f"{args.task}"
    #log_dir = Path(config['logging'].get('log_dir', f'logs/finetune_{args.task}'))
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(
        name='mae_ast',
        log_file=str(log_dir / 'finetune.log'),
        level=logging.INFO
    )

    logger.info("=" * 60)
    logger.info(f"MAE-AST Fine-tuning on {args.task.upper()}")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Pretrained checkpoint: {args.pretrained_checkpoint}")
    logger.info(f"Freeze encoder: {args.freeze_encoder}")

    # Set seed
    set_seed(config.get('seed', 42))

    # Get device
    device = get_device(config['hardware'].get('device', None))

    # Create dataloaders
    logger.info("\n" + "=" * 60)
    logger.info("Creating Data Loaders")
    logger.info("=" * 60)
    train_loader, val_loader, num_classes, task_type, metric = create_dataloaders(
        config, args.task
    )
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Task type: {task_type}")
    logger.info(f"Primary metric: {metric}")

    # Load pretrained encoder
    logger.info("\n" + "=" * 60)
    logger.info("Loading Pretrained Encoder")
    logger.info("=" * 60)
    task_config = config['tasks'][args.task]
    pretrained_model = load_pretrained_encoder(
        args.pretrained_checkpoint, config, task_config, device
    )

    # Create fine-tuning model
    logger.info("\n" + "=" * 60)
    logger.info("Creating Fine-tuning Model")
    logger.info("=" * 60)
    model = FineTuneModel(
        encoder=pretrained_model,
        num_classes=num_classes,
        dropout=config['model'].get('dropout', 0.1),
        freeze_encoder=args.freeze_encoder,
        task_type=task_type
    )
    model = model.to(device)
    print_model_info(model)

    # Create criterion
    if task_type == "multi-label":
        criterion = nn.BCEWithLogitsLoss()
        logger.info("Using BCEWithLogitsLoss for multi-label classification")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using CrossEntropyLoss for single-label classification")

    # Get task-specific training config
    task_config = config['tasks'][args.task]

    # Create optimizer
    logger.info("\n" + "=" * 60)
    logger.info("Creating Optimizer")
    logger.info("=" * 60)
    optimizer = build_optimizer(
        model=model,
        optimizer_name=config['training']['optimizer'],
        learning_rate=task_config.get('learning_rate', 5e-5),
        weight_decay=task_config.get('weight_decay', 0.01),
    )

    # Create scheduler
    logger.info("\n" + "=" * 60)
    logger.info("Creating Scheduler")
    logger.info("=" * 60)
    num_epochs = task_config.get('num_epochs', 50)
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_epochs = task_config.get('warmup_epochs', 5)

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=config['training'].get('scheduler', 'cosine'),
        total_steps=total_steps,
        warmup_steps=warmup_epochs * steps_per_epoch,
    )

    # Checkpointing
    base_checkpoint_dir = config['training'].get('checkpoint_dir', 'checkpoints/finetune')
    checkpoint_dir = Path(base_checkpoint_dir) / f"{args.task}"
    #checkpoint_dir = Path(config['training'].get('checkpoint_dir', f'checkpoints/finetune_{args.task}'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Metric for checkpointing based on config
    if metric == "mAP":
        checkpoint_metric = 'mAP'
        metric_mode = 'max'  # Higher mAP is better
    elif metric == "accuracy":
        checkpoint_metric = 'acc1'
        metric_mode = 'max'  # Higher accuracy is better
    else:
        checkpoint_metric = 'loss'
        metric_mode = 'min'  # Lower loss is better

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        max_keep=config['training'].get('keep_last_n', 3),
        metric_name=checkpoint_metric,
        mode=metric_mode
    )

    # TensorBoard logging
    tb_logger = TensorBoardLogger(
        str(log_dir),
        enabled=config['logging'].get('use_tensorboard', True)
    )

    # Resume if specified
    start_epoch = 0
    best_metric = 0.0 if metric_mode == 'max' else float('inf')
    best_metric_epoch = 0

    if args.resume is not None:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        from src.utils.checkpoint import load_checkpoint
        metadata = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = metadata.get('epoch', 0) + 1
        best_metric = metadata.get('best_metric', best_metric)
        logger.info(f"Resuming from epoch {start_epoch}, best {checkpoint_metric}: {best_metric}")

    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("Starting Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"Total epochs: {num_epochs}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 60)

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch + 1, config['training'], task_type
        )

        if task_type == "single-label":
            logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f} "
                f"Acc@1: {train_metrics['acc1']:.2f}% "
                f"Acc@5: {train_metrics['acc_k']:.2f}%"
            )
        else:
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, task_type, metric)

        if task_type == "single-label":
            logger.info(
                f"Val   - Loss: {val_metrics['loss']:.4f} "
                f"Acc@1: {val_metrics['acc1']:.2f}% "
                f"Acc@5: {val_metrics['acc_k']:.2f}%"
            )
        else:
            # Multi-label (AudioSet)
            if metric == "mAP":
                logger.info(
                    f"Val   - Loss: {val_metrics['loss']:.4f} "
                    f"mAP: {val_metrics.get('mAP', 0.0):.2f}"
                )
            else:
                logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}")

        # TensorBoard logging
        tb_logger.add_scalar('train/loss', train_metrics['loss'], epoch)
        tb_logger.add_scalar('val/loss', val_metrics['loss'], epoch)

        if task_type == "single-label":
            tb_logger.add_scalar('train/acc1', train_metrics['acc1'], epoch)
            tb_logger.add_scalar('train/acc_k', train_metrics['acc_k'], epoch)
            tb_logger.add_scalar('val/acc1', val_metrics['acc1'], epoch)
            tb_logger.add_scalar('val/acc_k', val_metrics['acc_k'], epoch)
        elif metric == "mAP":
            tb_logger.add_scalar('val/mAP', val_metrics.get('mAP', 0.0), epoch)

        tb_logger.add_scalar('lr', get_lr(optimizer), epoch)

        # Save checkpoint
        current_metric = val_metrics.get(checkpoint_metric, val_metrics['loss'])

        if metric_mode == 'max':
            is_best = current_metric > best_metric
        else:
            is_best = current_metric < best_metric

        if is_best:
            best_metric = current_metric
            best_metric_epoch = epoch
            logger.info(f"New best {checkpoint_metric}: {best_metric:.4f}")

        checkpoint_manager.save(
            step=epoch,
            epoch=epoch,
            metrics={checkpoint_metric: current_metric, 'loss': val_metrics['loss']},
            is_scheduled=False
        )

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Fine-tuning Complete!")
    logger.info("=" * 60)
    logger.info(f"Best {checkpoint_metric}: {best_metric:.4f}, in epoch {best_metric_epoch}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    logger.info("=" * 60)

    tb_logger.close()


if __name__ == "__main__":
    main()