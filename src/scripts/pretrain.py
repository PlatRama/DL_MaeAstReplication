"""
Pretrain MAE-AST model.

Paper Setup (Section 2.1):
- Dataset: AudioSet (2M clips) + LibriSpeech (960h)
- Training steps: 600,000 (~2 days on RTX-8000)
- Batch size: ~32
- Optimizer: Adam (lr=1e-4, weight_decay=0.01)
- Scheduler: Polynomial decay
- Masking: 75% random or chunked

Usage:
    python scripts/pretrain_first.py --config configs/pretraining_config.yaml
"""
import argparse
import logging
from pathlib import Path

import torch

from ..data import AudioDataset, AudioCollator
from ..losses import MAEASTLoss
from ..models import MAEASTConfig, MAEAST
from ..training import build_optimizer, build_scheduler, MAEASTTrainer
from ..utils import setup_logger, set_seed, get_device, save_config, print_model_info, load_config

#logger = logging.getLogger(__name__)
logger = logging.getLogger('mae_ast')

def create_dataloaders(config: dict):
    """Create train and validation data loaders."""
    # Training dataset
    logger.info("Creating training dataset...")
    train_dataset = AudioDataset(
        manifest_path=config['data']['manifest_path'],
        sample_rate=config['data']['sample_rate'],
        max_duration=config['data']['max_duration'],
        min_duration=config['data']['min_duration'],
        feature_type=config['data']['feature_type'],
        n_mels=config['data']['n_mels'],
        normalize=config['data']['normalize'],
        random_crop=True
    )
    logger.info(f"Training dataset: {len(train_dataset)} samples")

    # Validation dataset
    val_dataset = None
    if 'valid_manifest_path' in config['data']:
        logger.info("Creating validation dataset...")
        val_dataset = AudioDataset(
            manifest_path=config['data']['valid_manifest_path'],
            sample_rate=config['data']['sample_rate'],
            max_duration=config['data']['max_duration'],
            min_duration=config['data']['min_duration'],
            feature_type=config['data']['feature_type'],
            n_mels=config['data']['n_mels'],
            normalize=config['data']['normalize'],
            random_crop=False
        )
        logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # collator
    collator = AudioCollator(
        sample_rate=config['data']['sample_rate'],
        max_length=None,
        pad_to_multiple=1
    )

    # data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        collate_fn=collator,
        pin_memory=config['hardware']['pin_memory'],
        drop_last=True
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['hardware']['num_workers'],
            collate_fn=collator,
            pin_memory=config['hardware']['pin_memory']
        )

    return train_loader, val_loader


def create_model(config: dict):
    """Create MAE-AST model from config."""
    model_config = MAEASTConfig(
        # input
        input_type=config.get('input_type', 'patch'),
        n_mels=config['data']['n_mels'],
        patch_size_time=config['model']['patch_size_time'],
        patch_size_freq=config['model']['patch_size_freq'],
        frame_size_time=config['model'].get('frame_size_time', 2),

        # encoder
        encoder_embed_dim=config['model']['encoder_embed_dim'],
        encoder_depth=config['model']['encoder_depth'],
        encoder_num_heads=config['model']['encoder_num_heads'],
        encoder_ffn_dim=config['model']['encoder_ffn_dim'],

        # decoder
        decoder_embed_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_num_heads=config['model']['decoder_num_heads'],
        decoder_ffn_dim=config['model']['decoder_ffn_dim'],

        # masking
        mask_ratio=config['masking']['mask_ratio'],
        mask_type=config['masking']['mask_type'],
        mask_batched=config['masking'].get('mask_batched', True),
        chunk_size_range=tuple(config['masking'].get('chunk_size_range', [3, 5])),

        # positional encoding
        use_sinusoidal_pos=config['model']['use_sinusoidal_pos'],
        max_position=config['model'].get('max_position', 5000),

        # dropout
        dropout=config['model']['dropout'],

        # training
        layer_norm_first=config['model'].get('layer_norm_first', True),
    )

    model = MAEAST(model_config)
    return model


def main():
    parser = argparse.ArgumentParser(description='Pretrain MAE-AST')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()

    # load config
    config = load_config(args.config)

    # Setup logging
    log_dir = Path(config['logging'].get('log_dir', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(
        name='mae_ast',
        log_file=str(log_dir / 'pretrain.log'),
        level=logging.INFO
    )

    logger.info("=" * 60)
    logger.info("MAE-AST Pretraining")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")

    # Set random seed
    set_seed(config.get('seed', 42))

    device = get_device(config['hardware'].get('device', None))

    checkpoint_dir = Path(config['logging'].get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(checkpoint_dir / 'config.yaml'))

    # Create data loaders
    logger.info("\n" + "=" * 60)
    logger.info("Creating Data Loaders")
    logger.info("=" * 60)
    train_loader, val_loader = create_dataloaders(config)

    #create model
    logger.info("\n" + "=" * 60)
    logger.info("Creating Model")
    logger.info("=" * 60)
    model = create_model(config)
    print_model_info(model)

    # Create criterion
    logger.info("\n" + "=" * 60)
    logger.info("Creating Loss Function")
    logger.info("=" * 60)
    criterion = MAEASTLoss(
        reconstruction_weight=config['loss']['reconstruction_weight'],
        contrastive_weight=config['loss']['contrastive_weight'],
        temperature=config['loss'].get('temperature', 0.07),
        use_reconstruction=True,
        use_contrastive=True,
    )
    logger.info(f"Reconstruction weight (lambda): {config['loss']['reconstruction_weight']}")
    logger.info(f"Contrastive weight: {config['loss']['contrastive_weight']}")
    logger.info(f"Temperature: {config['loss'].get('temperature', 0.07)}")

    # Create optimizer
    logger.info("\n" + "=" * 60)
    logger.info("Creating Optimizer")
    logger.info("=" * 60)
    optimizer = build_optimizer(
        model=model,
        optimizer_name=config['training']['optimizer'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', [0.9, 0.999])),
        eps=float(config['training'].get('eps', 1e-8)),
    )

    # Create scheduler
    logger.info("\n" + "=" * 60)
    logger.info("Creating Learning Rate Scheduler")
    logger.info("=" * 60)
    print(config['training'].get('warmup_steps', 10000))

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=config['training']['scheduler'],
        total_steps=config['training']['total_steps'],
        warmup_steps=config['training'].get('warmup_steps', 10000),
        end_lr=config['training'].get('end_lr', 0.0),
        power=config['training'].get('power', 1.0),
    )

    # Create trainer
    logger.info("\n" + "=" * 60)
    logger.info("Creating Trainer")
    logger.info("=" * 60)
    trainer_config = {
        'max_steps': config['training']['total_steps'],
        'log_every': config['logging']['log_every'],
        'eval_every': config['logging']['eval_every'],
        'save_every': config['training']['save_every'],
        'max_grad_norm': config['training']['max_grad_norm'],
        'gradient_accumulation_steps': config['training'].get('gradient_accumulation_steps', 1),
        'use_amp': config['training'].get('use_amp', True),
        'checkpoint_dir': str(checkpoint_dir),
        'log_dir': str(log_dir),
        'use_tensorboard': config['logging']['use_tensorboard'],
        'max_keep_checkpoints': config['training'].get('keep_last_n', 3),
    }

    trainer = MAEASTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=trainer_config,
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        trainer.resume_from_checkpoint(args.resume)

    # Start training
    logger.info("\n" + "=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    trainer.train()

    logger.info("\n" + "=" * 60)
    logger.info("Pretraining Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()



