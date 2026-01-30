import logging
import time
from typing import Optional, Dict, Any

import torch
from torch import GradScaler
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from ..models import MAEAST
from ..losses import MAEASTLoss
from ..utils import (
    MetricLogger,
    TensorBoardLogger,
    CheckpointManager,
    get_lr,
    format_time)

logger = logging.getLogger('mae_ast')
class MAEASTTrainer:
    """
    Trainer for MAE-AST pretraining.

    Args:
        model: MAE-AST model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device for training
        config: Training configuration dict
    """
    def __init__(self,
                 model: MAEAST,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 criterion: MAEASTLoss,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 device: torch.device,
                 config: Dict[str, Any]):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        # move model to device
        self.model.to(device)

        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # Configuration
        self.max_steps = config.get('max_steps', 600000)
        self.log_every = config.get('log_every', 100)
        self.eval_every = config.get('eval_every', 5000)
        self.save_every = config.get('save_every', 10000)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir, model=model,
            optimizer=optimizer, scheduler=scheduler,
            max_keep=config.get('max_keep_checkpoints', 5),
            save_every=self.save_every,
            metric_name='val_loss',
            mode='min'
        )

        #logging
        log_dir = config.get('log_dir', 'logs')
        use_tensorboard = config.get('use_tensorboard', True)
        self.tb_logger = TensorBoardLogger(log_dir, enabled=use_tensorboard)

        self.train_metrics = MetricLogger(delimiter= "  ")

        logger.info("=" * 60)
        logger.info("MAEASTTrainer Initialized")
        logger.info("=" * 60)
        logger.info(f"Max steps: {self.max_steps:,}")
        logger.info(f"Batch size: {train_loader.batch_size}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {train_loader.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"Log every: {self.log_every} steps")
        logger.info(f"Eval every: {self.eval_every} steps")
        logger.info(f"Save every: {self.save_every} steps")
        logger.info(f"Max grad norm: {self.max_grad_norm}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Device: {device}")
        logger.info("=" * 60)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch from data loader with keys:
                - 'audio': [batch, time, n_mels]
                - 'padding_mask': [batch, time]

        Returns:
            Dict with loss values
        """
        # Move batch to device
        audio = batch['audio'].to(self.device)
        padding_mask = batch.get('padding_mask', None)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)

        # forward pass with mixed precision
        if self.use_amp:
            with torch.amp.autocast(device_type='cuda'):#TODO: mi da errore in fase di test
                model_output = self.model(audio, padding_mask)

                losses = self.criterion.forward_from_model_output(model_output)
                loss = losses['loss']

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
        else:
            model_output = self.model(audio, padding_mask)

            losses = self.criterion.forward_from_model_output(model_output)
            loss = losses['loss']

            loss = loss / self.gradient_accumulation_steps


        #backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Return unscaled losses for logging
        return {
            'loss': losses['loss'].item(),
            'reconstruction_loss': losses.get('reconstruction_loss', torch.tensor(0.0)).item(),
            'contrastive_loss': losses.get('contrastive_loss', torch.tensor(0.0)).item(),
        }

    def optimizer_step(self):
        """
        Optimizer step with gradient clipping.
        """
        if self.use_amp:
            # unscale gradients
            self.scaler.unscale_(self.optimizer)

            # clip gradients
            if self.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            else:
                grad_norm = 0.0

            # opt step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            else:
                grad_norm = 0.0

            self.optimizer.step()

        self.optimizer.zero_grad()
        return grad_norm

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch (or until max_steps reached).

        Returns:
            Dict with average metrics
        """
        self.model.train()
        #todo: implementare metodo
        #self.train_metrics.reset()

        epoch_start_time = time.time()
        for batch_idx, batch in enumerate(self.train_loader):
            step_start_time = time.time()

            # training step
            step_losses = self.train_step(batch)

            self.train_metrics.update(**step_losses)

            # Optimizer step (every gradient_accumulation_steps)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                grad_norm = self.optimizer_step()
                # Learning rate step
                self.scheduler.step()
                # Increment global step
                self.current_step += 1

                # Logging
                if self.current_step % self.log_every == 0:
                    step_time = time.time() - step_start_time
                    current_lr = get_lr(self.optimizer)

                    # Console logging
                    logger.info(
                        f"Step [{self.current_step}/{self.max_steps}] "
                        f"Loss: {step_losses['loss']:.4f} "
                        f"(Recon: {step_losses['reconstruction_loss']:.4f}, "
                        f"Contrast: {step_losses['contrastive_loss']:.4f}) "
                        f"LR: {current_lr:.2e} "
                        f"Time: {step_time:.3f}s"
                    )

                    # TensorBoard logging
                    self.tb_logger.add_scalar('train/loss', step_losses['loss'], self.current_step)
                    self.tb_logger.add_scalar('train/reconstruction_loss',
                                              step_losses['reconstruction_loss'], self.current_step)
                    self.tb_logger.add_scalar('train/contrastive_loss',
                                              step_losses['contrastive_loss'], self.current_step)
                    self.tb_logger.add_scalar('train/learning_rate', current_lr, self.current_step)
                    if grad_norm > 0:
                        self.tb_logger.add_scalar('train/grad_norm', grad_norm, self.current_step)


                # validation
                if self.val_loader is not None and self.current_step % self.eval_every == 0:
                    val_metrics = self.validate()

                    logger.info(
                        f"Validation [{self.current_step}] "
                        f"Loss: {val_metrics['val_loss']:.4f} "
                        f"(Recon: {val_metrics['val_reconstruction_loss']:.4f}, "
                        f"Contrast: {val_metrics['val_contrastive_loss']:.4f})"
                    )

                    # TensorBoard logging
                    self.tb_logger.add_scalar('val/loss', val_metrics['val_loss'], self.current_step)
                    self.tb_logger.add_scalar('val/reconstruction_loss',
                                              val_metrics['val_reconstruction_loss'], self.current_step)
                    self.tb_logger.add_scalar('val/contrastive_loss',
                                              val_metrics['val_contrastive_loss'], self.current_step)

                    # Check if best model
                    is_best = val_metrics['val_loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['val_loss']
                        logger.info(f"New best validation loss: {self.best_val_loss:.4f}")

                    #save checkpoint
                    self.checkpoint_manager.save(
                        step=self.current_step,
                        epoch=self.current_epoch,
                        metrics=val_metrics,
                        is_scheduled=False
                    )

                elif self.current_step % self.save_every == 0:
                    self.checkpoint_manager.save(
                        step=self.current_step,
                        epoch=self.current_epoch,
                        metrics=self.train_metrics.get_global_avg(),
                        is_scheduled=True
                    )

                if self.current_step >= self.max_steps:
                    logger.info(f"Reached max steps: {self.max_steps}")
                    break


        epoch_time = time.time() - epoch_start_time
        avg_metrics = self.train_metrics.get_global_avg()

        logger.info(
            f"Epoch {self.current_epoch} completed in {format_time(epoch_time)} | "
            f"Avg Loss: {avg_metrics['loss']:.4f}"
        )

        return avg_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_metrics = MetricLogger(delimiter=" ")

        for batch in self.val_loader:
            audio = batch['audio'].to(self.device)
            padding_mask = batch.get('padding_mask', None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device)

            if self.use_amp:
                with autocast():
                    model_output = self.model(audio, padding_mask)
                    losses = self.criterion.forward_from_model_output(model_output)
            else:
                model_output = self.model(audio, padding_mask)
                losses = self.criterion.forward_from_model_output(model_output)

            # Update metrics
            val_metrics.update(
                loss=losses['loss'].item(),
                reconstruction_loss=losses.get('reconstruction_loss', torch.tensor(0.0)).item(),
                contrastive_loss=losses.get('contrastive_loss', torch.tensor(0.0)).item(),
            )

        avg_metrics = val_metrics.get_global_avg()

        # Rename keys for clarity
        avg_metrics = {
            'val_loss': avg_metrics['loss'],
            'val_reconstruction_loss': avg_metrics['reconstruction_loss'],
            'val_contrastive_loss': avg_metrics['contrastive_loss'],
        }

        self.model.train()
        return avg_metrics

    def train(self):
        logger.info("Starting training...")
        logger.info(f"Training from step {self.current_step} to {self.max_steps}")

        training_start_time = time.time()
        while self.current_step < self.max_steps:
            self.train_epoch()

            self.current_epoch += 1

            if self.current_step >= self.max_steps:
                break

        total_time = time.time() - training_start_time
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Total time: {format_time(total_time)}")
        logger.info(f"Total steps: {self.current_step}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)

        final_metrics = {"val_loss": self.best_val_loss}
        self.checkpoint_manager.save(
            step=self.current_step,
            epoch=self.current_epoch,
            metrics=final_metrics,
            is_scheduled=False
        )

        self.tb_logger.close()

    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None):
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
                            If None, loads latest checkpoint
        """
        if checkpoint_path is None:
            metadata = self.checkpoint_manager.load_latest(device=self.device)
        else:
            from ..utils.checkpoint import load_checkpoint
            metadata = load_checkpoint(
                checkpoint_path,
                self.model,
                self.optimizer,
                self.scheduler,
                device=self.device
            )

        if metadata is not None:
            self.current_step = metadata.get('step', 0)
            self.current_epoch = metadata.get('epoch', 0)
            self.best_val_loss = metadata.get('best_metric', float('inf'))

            logger.info(f"Resumed from checkpoint:")
            logger.info(f"  Step: {self.current_step}")
            logger.info(f"  Epoch: {self.current_epoch}")
            logger.info(f"  Best val loss: {self.best_val_loss:.4f}")
        else:
            logger.info("No checkpoint found, starting from scratch")