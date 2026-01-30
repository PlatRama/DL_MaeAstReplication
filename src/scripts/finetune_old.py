"""
Fine-tune pretrained MAE-AST on downstream tasks.

Paper Context:
After pretraining, the model is fine-tuned on downstream tasks by:
1. Loading pretrained encoder weights
2. Removing decoder (only encoder is used)
3. Adding task-specific classification head
4. Training on labeled data

Downstream tasks evaluated in paper (Section 3.1):
- AudioSet (AS): Audio event classification
- ESC-50 (ESC): Environmental sound classification
- Speech Commands 1/2 (KS1/KS2): Keyword spotting
- VoxCeleb (SID): Speaker identification
- IEMOCAP (ER): Emotion recognition

Usage:
    python scripts/finetune.py \
        --config configs/finetuning_config.yaml \
        --pretrained_checkpoint checkpoints/pretrain_best.pt \
        --task audioset
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
    setup_logger,
    set_seed,
    get_device,
    print_model_info,
    save_config,
    MetricLogger,
    TensorBoardLogger,
    CheckpointManager,
    compute_accuracy,
    get_lr,
)

logger = logging.getLogger('mae_ast')


class ClassificationHead(nn.Module):
    """
    Classification head for downstream tasks.
    Paper: Uses mean pooling + linear classifier for fine-tuning.
    """

    def __init__(
            self,
            encoder_dim: int = 768,
            num_classes: int = 527,
            dropout: float = 0.1,
            use_layernorm: bool = True
    ):
        super().__init__()
        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.norm = nn.LayerNorm(encoder_dim)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder_dim, num_classes)

        # Initialize
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, encoder_output: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        Forward pass.

        Args:
            encoder_output: [batch, seq_len, encoder_dim] - output patches dall'encoder
            padding_mask: [batch, time] - maschera originale sui frame temporali

        Returns:
            logits: [batch, num_classes]
        """
        # ✅ CORREZIONE: Mean pooling semplice senza padding mask
        # Durante il fine-tuning NON si applica masking, quindi possiamo
        # semplicemente fare mean pooling su tutte le patch

        # Il padding_mask originale è sui frame audio [batch, time_frames]
        # ma encoder_output è sulle patch [batch, num_patches, dim]
        # Siccome le patch vengono create da regioni contigue di frame,
        # è difficile mappare correttamente la maschera.

        # Soluzione 1: Mean pooling semplice (funziona bene per audio di lunghezza fissa)
        pooled = encoder_output.mean(dim=1)  # [batch, encoder_dim]

        # Soluzione 2 (se vuoi usare padding mask - più complessa):
        # if padding_mask is not None:
        #     # Converti padding mask da frame a patch
        #     # Assumi che ogni patch copre patch_size_time frames
        #     patch_size_time = 16  # Dal config
        #     num_patches = encoder_output.size(1)
        #
        #     # Prendi un frame ogni patch_size_time per rappresentare ogni patch
        #     patch_mask = padding_mask[:, ::patch_size_time][:, :num_patches]
        #
        #     # Mean pooling mascherato
        #     mask_expanded = (~patch_mask).unsqueeze(-1).float()
        #     masked_output = encoder_output * mask_expanded
        #     pooled = masked_output.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        # else:
        #     pooled = encoder_output.mean(dim=1)

        # Classification
        if self.use_layernorm:
            pooled = self.norm(pooled)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits


class FineTuneModel(nn.Module):
    """
    Wrapper model for fine-tuning.

    Combines pretrained MAE-AST encoder with classification head.
    """

    def __init__(
            self,
            encoder: MAEAST,
            num_classes: int,
            dropout: float = 0.1,
            freeze_encoder: bool = False
    ):
        super().__init__()

        self.encoder = encoder
        self.classification_head = ClassificationHead(
            encoder_dim=encoder.config.encoder_embed_dim,
            num_classes=num_classes,
            dropout=dropout
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
            audio: [batch, n_mels, time]
            padding_mask: [batch, time]

        Returns:
            logits: [batch, num_classes]
        """
        # Extract encoder features (no masking during fine-tuning)
        encoder_output, _ = self.encoder.forward_encoder(
            audio, padding_mask, apply_mask=False
        )

        # Classification
        logits = self.classification_head(encoder_output, padding_mask)

        return logits


def load_pretrained_encoder(checkpoint_path: str, config: dict, device: torch.device):
    """
    Load pretrained encoder from checkpoint.

    Args:
        checkpoint_path: Path to pretrained checkpoint
        config: Model configuration
        device: Device

    Returns:
        MAE-AST model with pretrained weights
    """
    logger.info(f"Loading pretrained encoder from: {checkpoint_path}")

    # Create model config
    model_config = MAEASTConfig(
        n_mels=config['data'].get('n_mels', 128),#todo impostare input_type
        patch_size_time=config['model']['patch_size_time'],
        patch_size_freq=config['model']['patch_size_freq'],
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
        use_sinusoidal_pos=config['model']['use_sinusoidal_pos'],
        dropout=config['model']['dropout'],
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


def create_finetuning_dataloaders(config: dict, task: str):
    """
    Create dataloaders for fine-tuning task.

    Args:
        config: Configuration dict
        task: Task name (audioset, esc50, etc.)

    Returns:
        train_loader, val_loader, num_classes
    """
    # Task-specific configuration
    task_config = config['tasks'][task]

    # Training dataset
    logger.info(f"Creating training dataset for {task}...")
    train_dataset = AudioDataset(
        manifest_path=task_config['train_manifest'],
        sample_rate=config['data']['sample_rate'],
        max_duration=config['data'].get('max_duration', 10.0),
        feature_type=config['data'].get('feature_type', 'fbank'),
        n_mels=config['data'].get('n_mels', 128),
        normalize=config['data'].get('normalize', True),
        random_crop=True,
        return_labels=True
    )
    logger.info(f"Training dataset: {len(train_dataset)} samples")

    # Validation dataset
    val_dataset = AudioDataset(
        manifest_path=task_config['val_manifest'],
        sample_rate=config['data']['sample_rate'],
        max_duration=config['data'].get('max_duration', 10.0),
        feature_type=config['data'].get('feature_type', 'fbank'),
        n_mels=config['data'].get('n_mels', 128),
        normalize=config['data'].get('normalize', True),
        random_crop=False,
        return_labels=True
    )
    logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # Collator
    collator = AudioCollator(sample_rate=config['data']['sample_rate'])

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        collate_fn=collator,
        pin_memory=config['hardware']['pin_memory'],
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        collate_fn=collator,
        pin_memory=config['hardware']['pin_memory'],
    )

    num_classes = task_config['num_classes']

    return train_loader, val_loader, num_classes


def train_epoch(
        model: FineTuneModel,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        epoch: int,
        config: dict
):
    """
    Train for one epoch.

    Args:
        model: Fine-tuning model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device
        epoch: Current epoch number
        config: Training configuration

    Returns:
        Dict with training metrics
    """
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

        # Compute accuracy
        acc1, acc5 = compute_accuracy(logits, labels, topk=(1, 5))

        # Update metrics
        metrics.update(
            loss=loss.item(),
            acc1=acc1,
            acc5=acc5,
        )

        # Logging
        if batch_idx % log_every == 0:
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Acc@1: {acc1:.2f}% Acc@5: {acc5:.2f}% "
                f"LR: {get_lr(optimizer):.2e}"
            )

    return metrics.get_global_avg()


@torch.no_grad()
def validate(
        model: FineTuneModel,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
):
    """
    Validate model.

    Args:
        model: Fine-tuning model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device

    Returns:
        Dict with validation metrics
    """
    model.eval()
    metrics = MetricLogger(delimiter="  ")

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
        loss = criterion(logits, labels)

        # Compute accuracy
        acc1, acc5 = compute_accuracy(logits, labels, topk=(1, 5))

        # Update metrics
        metrics.update(
            loss=loss.item(),
            acc1=acc1,
            acc5=acc5,
        )

    return metrics.get_global_avg()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MAE-AST on downstream tasks")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to fine-tuning configuration YAML'
    )
    parser.add_argument(
        '--pretrained_checkpoint',
        type=str,
        required=True,
        help='Path to pretrained MAE-AST checkpoint'
    )
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['audioset', 'esc50', 'speechcommands', 'voxceleb', 'iemocap'],
        help='Downstream task name'
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        help='Freeze encoder weights (only train classification head)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    log_dir = Path(config['logging'].get('log_dir', f'logs/finetune_{args.task}'))
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
    train_loader, val_loader, num_classes = create_finetuning_dataloaders(
        config, args.task
    )
    logger.info(f"Number of classes: {num_classes}")

    # Load pretrained encoder
    logger.info("\n" + "=" * 60)
    logger.info("Loading Pretrained Encoder")
    logger.info("=" * 60)
    pretrained_model = load_pretrained_encoder(
        args.pretrained_checkpoint, config, device
    )

    # Create fine-tuning model
    logger.info("\n" + "=" * 60)
    logger.info("Creating Fine-tuning Model")
    logger.info("=" * 60)
    model = FineTuneModel(
        encoder=pretrained_model,
        num_classes=num_classes,
        dropout=config['model'].get('dropout', 0.1),
        freeze_encoder=args.freeze_encoder
    )
    model = model.to(device)
    print_model_info(model)

    # Create criterion
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    logger.info("\n" + "=" * 60)
    logger.info("Creating Optimizer")
    logger.info("=" * 60)
    optimizer = build_optimizer(
        model=model,
        optimizer_name=config['training']['optimizer'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    # Create scheduler
    logger.info("\n" + "=" * 60)
    logger.info("Creating Scheduler")
    logger.info("=" * 60)
    num_epochs = config['training']['num_epochs']
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=config['training'].get('scheduler', 'cosine'),
        total_steps=total_steps,
        warmup_steps=config['training'].get('warmup_epochs', 5) * steps_per_epoch,
    )

    # Checkpointing
    checkpoint_dir = Path(config['training'].get('checkpoint_dir', f'checkpoints/finetune_{args.task}'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        max_keep=config['training'].get('keep_last_n', 3),
        metric_name='acc1',
        mode='max'  # Higher accuracy is better
    )

    # TensorBoard logging
    tb_logger = TensorBoardLogger(
        str(log_dir),
        enabled=config['logging'].get('use_tensorboard', True)
    )

    # Resume if specified
    start_epoch = 0
    best_acc = 0.0

    if args.resume is not None:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        from src.utils.checkpoint import load_checkpoint
        metadata = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        start_epoch = metadata.get('epoch', 0) + 1
        best_acc = metadata.get('best_metric', 0.0)
        logger.info(f"Resuming from epoch {start_epoch}, best acc: {best_acc:.2f}%")

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
            device, epoch + 1, config['training']
        )

        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f} "
            f"Acc@1: {train_metrics['acc1']:.2f}% "
            f"Acc@5: {train_metrics['acc5']:.2f}%"
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        logger.info(
            f"Val   - Loss: {val_metrics['loss']:.4f} "
            f"Acc@1: {val_metrics['acc1']:.2f}% "
            f"Acc@5: {val_metrics['acc5']:.2f}%"
        )

        # TensorBoard logging
        tb_logger.add_scalar('train/loss', train_metrics['loss'], epoch)
        tb_logger.add_scalar('train/acc1', train_metrics['acc1'], epoch)
        tb_logger.add_scalar('train/acc5', train_metrics['acc5'], epoch)
        tb_logger.add_scalar('val/loss', val_metrics['loss'], epoch)
        tb_logger.add_scalar('val/acc1', val_metrics['acc1'], epoch)
        tb_logger.add_scalar('val/acc5', val_metrics['acc5'], epoch)
        tb_logger.add_scalar('lr', get_lr(optimizer), epoch)

        # Save checkpoint
        is_best = val_metrics['acc1'] > best_acc
        if is_best:
            best_acc = val_metrics['acc1']
            logger.info(f"New best accuracy: {best_acc:.2f}%")

        checkpoint_manager.save(
            step=epoch,
            epoch=epoch,
            metrics={'acc1': val_metrics['acc1'], 'loss': val_metrics['loss']},
            is_scheduled=False
        )

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Fine-tuning Complete!")
    logger.info("=" * 60)
    logger.info(f"Best validation accuracy: {best_acc:.2f}%")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    logger.info("=" * 60)

    tb_logger.close()


if __name__ == "__main__":
    main()