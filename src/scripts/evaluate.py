"""
Evaluate pretrained MAE-AST model.

This script evaluates a pretrained model by:
1. Computing reconstruction quality
2. Visualizing reconstructed spectrograms
3. Computing embedding quality metrics

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/best.pt \
        --config configs/pretraining_config.yaml \
        --data_manifest data/manifests/test.tsv \
        --output_dir evaluation_results
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import AudioDataset, AudioCollator
from ..losses import MAEASTLoss
from ..models import MAEASTConfig, MAEAST
#from .evaluate import visualize_reconstructions
from src.utils import setup_logger, set_seed, get_device

#logger = logging.getLogger(__name__)
logger = logging.getLogger('mae_ast')


def load_model_from_checkpoint(checkpoint_path: str, config: dict, device: torch.device):
    """
    Load pretrained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dict
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model from: {checkpoint_path}")

    # Create model config
    model_config = MAEASTConfig(
        # Input
        n_mels=config['data'].get('n_mels', 128),
        patch_size_time=config['model']['patch_size_time'],
        patch_size_freq=config['model']['patch_size_freq'],

        # Encoder
        encoder_embed_dim=config['model']['encoder_embed_dim'],
        encoder_depth=config['model']['encoder_depth'],
        encoder_num_heads=config['model']['encoder_num_heads'],
        encoder_ffn_dim=config['model']['encoder_ffn_dim'],

        # Decoder
        decoder_embed_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_num_heads=config['model']['decoder_num_heads'],
        decoder_ffn_dim=config['model']['decoder_ffn_dim'],

        # Masking
        mask_ratio=config['masking']['mask_ratio'],
        mask_type=config['masking']['mask_type'],
        mask_batched=config['masking'].get('mask_batched', True),

        # Positional encoding
        use_sinusoidal_pos=config['model']['use_sinusoidal_pos'],
        use_conv_pos=config['model'].get('use_conv_pos', False),

        # Dropout
        dropout=config['model']['dropout'],
    )

    # Create model
    model = MAEAST(model_config)

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f"Loaded model from step {checkpoint.get('step', 'unknown')}")
    else:
        model.load_state_dict(checkpoint, strict=True)
        logger.info("Loaded model weights")

    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully")

    return model


@torch.no_grad()
def evaluate_reconstruction(
        model: MAEAST,
        data_loader: DataLoader,
        criterion: MAEASTLoss,
        device: torch.device,
        num_samples: int = 100
):
    """
    Evaluate reconstruction quality on test set.

    Args:
        model: MAE-AST model
        data_loader: Test data loader
        criterion: Loss function
        device: Device
        num_samples: Number of samples to evaluate

    Returns:
        Dict with evaluation metrics
    """
    model.eval()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_contrast_loss = 0.0
    num_batches = 0

    # For computing additional metrics
    all_recon_errors = []
    all_contrast_similarities = []

    logger.info(f"Evaluating on {num_samples} samples...")

    max_batches = (num_samples + data_loader.batch_size - 1) // data_loader.batch_size

    for batch_idx, batch in enumerate(tqdm(data_loader, total=max_batches, desc="Evaluating")):
        if batch_idx >= max_batches:
            break

        audio = batch['audio'].to(device)
        padding_mask = batch.get('padding_mask', None)
        if padding_mask is not None:
            padding_mask = padding_mask.to(device)

        model_output = model(audio, padding_mask=padding_mask)

        losses = criterion.forward_from_model_output(model_output)

        # Accumulate losses
        total_loss += losses['loss'].item()
        total_recon_loss += losses.get('reconstruction_loss', torch.tensor(0.0)).item()
        total_contrast_loss += losses.get('contrastive_loss', torch.tensor(0.0)).item()
        num_batches += 1

        # Compute per-sample reconstruction error
        recon_pred = model_output['reconstruction_pred']  # [batch, num_masked, patch_dim]
        targets = model_output['targets']  # [batch, num_masked, patch_dim]

        # MSE per sample
        recon_error = torch.mean((recon_pred - targets) ** 2, dim=[1, 2])
        all_recon_errors.extend(recon_error.cpu().numpy().tolist())

        # Cosine similarity for contrastive
        recon_pred_norm = torch.nn.functional.normalize(recon_pred, dim=-1)
        targets_norm = torch.nn.functional.normalize(targets, dim=-1)
        cos_sim = torch.sum(recon_pred_norm * targets_norm, dim=-1)  # [batch, num_masked]
        all_contrast_similarities.extend(cos_sim.mean(dim=1).cpu().numpy().tolist())

    # Compute average metrics
    metrics = {
        'loss': total_loss / num_batches,
        'reconstruction_loss': total_recon_loss / num_batches,
        'contrastive_loss': total_contrast_loss / num_batches,
        'mean_recon_error': np.mean(all_recon_errors),
        'std_recon_error': np.std(all_recon_errors),
        'mean_cosine_similarity': np.mean(all_contrast_similarities),
        'std_cosine_similarity': np.std(all_contrast_similarities),
    }
    return metrics


@torch.no_grad()
def visualize_reconstructions(
        model: MAEAST,
        data_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
        num_examples: int = 5
):
    """
    Visualize original vs reconstructed spectrograms.

    Args:
        model: MAE-AST model
        data_loader: Data loader
        device: Device
        output_dir: Directory to save visualizations
        num_examples: Number of examples to visualize
    """
    model.eval()

    logger.info(f"Generating {num_examples} reconstruction visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get one batch
    batch = next(iter(data_loader))
    audio = batch['audio'][:num_examples].to(device)

    # Forward pass
    model_output = model(audio)

    # Get predictions and targets
    recon_pred = model_output['reconstruction_pred']  # [batch, num_masked, patch_dim]
    targets = model_output['targets']  # [batch, num_masked, patch_dim]

    # Reshape patches to 2D for visualization
    patch_size = int(np.sqrt(recon_pred.size(-1)))  # Assuming square patches

    # Visualize each example
    for i in range(num_examples):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Original spectrogram
        spec_orig = audio[i].cpu().numpy()
        im0 = axes[0, 0].imshow(spec_orig.T, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title('Original Spectrogram', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time Frame')
        axes[0, 0].set_ylabel('Mel Frequency')
        plt.colorbar(im0, ax=axes[0, 0])

        # Masked patches (ground truth)
        target_patches = targets[i].cpu().numpy()  # [num_masked, patch_dim]
        im1 = axes[0, 1].imshow(target_patches.T, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title(f'Ground Truth Masked Patches\n({target_patches.shape[0]} patches)',
                             fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Patch Index')
        axes[0, 1].set_ylabel('Patch Dimension')
        plt.colorbar(im1, ax=axes[0, 1])

        # Reconstructed patches
        recon_patches = recon_pred[i].cpu().numpy()  # [num_masked, patch_dim]
        im2 = axes[0, 2].imshow(recon_patches.T, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 2].set_title(f'Reconstructed Masked Patches\n({recon_patches.shape[0]} patches)',
                             fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Patch Index')
        axes[0, 2].set_ylabel('Patch Dimension')
        plt.colorbar(im2, ax=axes[0, 2])

        # Reconstruction error
        error = np.abs(target_patches - recon_patches)
        im3 = axes[1, 0].imshow(error.T, aspect='auto', origin='lower', cmap='hot')
        axes[1, 0].set_title(f'Absolute Error\nMean: {error.mean():.4f}',
                             fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Patch Index')
        axes[1, 0].set_ylabel('Patch Dimension')
        plt.colorbar(im3, ax=axes[1, 0])

        # Per-patch MSE
        per_patch_mse = np.mean((target_patches - recon_patches) ** 2, axis=1)
        axes[1, 1].bar(range(len(per_patch_mse)), per_patch_mse, color='steelblue', alpha=0.7)
        axes[1, 1].set_title(f'Per-Patch MSE\nMean: {per_patch_mse.mean():.4f}',
                             fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Patch Index')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].grid(True, alpha=0.3)

        # Cosine similarity per patch
        target_norm = target_patches / (np.linalg.norm(target_patches, axis=1, keepdims=True) + 1e-8)
        recon_norm = recon_patches / (np.linalg.norm(recon_patches, axis=1, keepdims=True) + 1e-8)
        cos_sim = np.sum(target_norm * recon_norm, axis=1)
        axes[1, 2].bar(range(len(cos_sim)), cos_sim, color='forestgreen', alpha=0.7)
        axes[1, 2].set_title(f'Per-Patch Cosine Similarity\nMean: {cos_sim.mean():.4f}',
                             fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Patch Index')
        axes[1, 2].set_ylabel('Cosine Similarity')
        axes[1, 2].set_ylim([0, 1])
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'reconstruction_{i:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


@torch.no_grad()
def extract_embeddings(model: MAEAST,
                       data_loader: DataLoader,
                       device: torch.device,
                       num_samples: int = 100):
    """
    Extract encoder embeddings for visualization/analysis.

    Args:
        model: MAE-AST model
        data_loader: Data loader
        device: Device
        num_samples: Number of samples to process

    Returns:
        embeddings: [num_samples, embed_dim] numpy array
        ids: [num_samples] sample IDs
    """
    model.eval()

    all_embeddings = []
    all_ids = []

    logger.info(f"Extracting embeddings from {num_samples} samples...")

    max_batches = (num_samples + data_loader.batch_size - 1) // data_loader.batch_size

    for batch_idx, batch in enumerate(tqdm(data_loader, total=max_batches, desc="Extracting")):
        if batch_idx >= max_batches:
            break

        # Move to device
        audio = batch['audio'].to(device)
        padding_mask = batch.get('padding_mask', None)
        if padding_mask is not None:
            padding_mask = padding_mask.to(device)

        # Extract encoder features (no masking)
        encoder_output, _ = model.forward_encoder(audio, padding_mask, apply_mask=False)

        # Mean pool over sequence dimension
        if padding_mask is not None:
            # Mask out padding before pooling
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            masked_output = encoder_output * mask_expanded
            embeddings = masked_output.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            embeddings = encoder_output.mean(dim=1)

        all_embeddings.append(embeddings.cpu().numpy())
        all_ids.extend(batch['ids'].cpu().numpy().tolist())

    embeddings = np.concatenate(all_embeddings, axis=0)

    logger.info(f"Extracted embeddings: {embeddings.shape}")

    return embeddings, all_ids


def visualize_embedding_distribution(
        embeddings: np.ndarray,
        output_dir: Path
):
    """
    Visualize embedding distribution using t-SNE or PCA.

    Args:
        embeddings: [num_samples, embed_dim] embeddings
        output_dir: Output directory
    """
    logger.info("Visualizing embedding distribution...")

    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        # PCA
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
                    alpha=0.5, s=20, c=range(len(embeddings)), cmap='viridis')
        plt.colorbar(label='Sample Index')
        plt.title('Embedding Distribution (PCA)', fontsize=14, fontweight='bold')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'embeddings_pca.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

        # t-SNE (only if not too many samples)
        if len(embeddings) <= 1000:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_tsne = tsne.fit_transform(embeddings)

            plt.figure(figsize=(10, 8))
            plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                        alpha=0.5, s=20, c=range(len(embeddings)), cmap='viridis')
            plt.colorbar(label='Sample Index')
            plt.title('Embedding Distribution (t-SNE)', fontsize=14, fontweight='bold')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'embeddings_tsne.png', dpi=150, bbox_inches='tight')
            plt.close()

            logger.info("t-SNE visualization saved")
        else:
            logger.info("Skipping t-SNE (too many samples, use PCA instead)")

    except ImportError:
        logger.warning("scikit-learn not available, skipping embedding visualization")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAE-AST Model")
    parser.add_argument("--checkpoint", type=str, default=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", type=str, default=True, help="Path to training configuration YAML file")
    parser.add_argument("--data_manifest", type=str, default=None, help="Path to test data manifests TSV file")
    parser.add_argument('--output_dir', type=str, default='evaluation_results',help='Output directory for evaluation results')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--num_visualizations', type=int, default=5,help='Number of reconstruction visualizations to generate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--extract_embeddings', action='store_true', help='Extract and visualize embeddings')

    args = parser.parse_args()
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logger('mae_ast', str(output_dir / 'evaluate.log'))

    logger.info("=" * 60)
    logger.info("MAE-AST Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data manifests: {args.data_manifest}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of samples: {args.num_samples}")

    # Load configuration
    logger.info("\nLoading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    device = get_device()

    logger.info("\n" + "=" * 60)
    logger.info("Loading Model")
    logger.info("=" * 60)
    model = load_model_from_checkpoint(args.checkpoint, config, device)

    # Create dataset
    logger.info("\n" + "=" * 60)
    logger.info("Creating Dataset")
    logger.info("=" * 60)
    dataset = AudioDataset(
        manifest_path=args.data_manifest,
        sample_rate=config['data']['sample_rate'],
        max_duration=config['data']['max_duration'],
        feature_type=config['data']['feature_type'],
        n_mels=config['data']['n_mels'],
        normalize=config['data']['normalize'],
        random_crop=False
    )
    logger.info(f"Dataset size: {len(dataset)} samples")


    collator = AudioCollator(sample_rate=config['data']['sample_rate'])
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True
    )

    criterion = MAEASTLoss(
        reconstruction_weight=config['loss']['reconstruction_weight'],
        contrastive_weight=config['loss']['contrastive_weight'],
        temperature=config['loss']['temperature'],
    )

    # Evaluate reconstruction quality
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating Reconstruction Quality")
    logger.info("=" * 60)
    metrics = evaluate_reconstruction(model, data_loader, criterion, device, num_samples=args.num_samples)

    logger.info("\nEvaluation Metrics:")
    logger.info(f"  Total Loss: {metrics['loss']:.4f}")
    logger.info(f"  Reconstruction Loss: {metrics['reconstruction_loss']:.4f}")
    logger.info(f"  Contrastive Loss: {metrics['contrastive_loss']:.4f}")
    logger.info(f"  Mean Reconstruction Error: {metrics['mean_recon_error']:.4f} ± {metrics['std_recon_error']:.4f}")
    logger.info(f"  Mean Cosine Similarity: {metrics['mean_cosine_similarity']:.4f} ± {metrics['std_cosine_similarity']:.4f}")

    # Save metrics
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nMetrics saved to: {metrics_file}")

    # Generate visualizations
    logger.info("\n" + "=" * 60)
    logger.info("Generating Reconstruction Visualizations")
    logger.info("=" * 60)
    vis_dir = output_dir / 'visualizations'
    visualize_reconstructions(
        model, data_loader, device, vis_dir, args.num_visualizations
    )

    # Extract and visualize embeddings (optional)
    if args.extract_embeddings:
        logger.info("\n" + "=" * 60)
        logger.info("Extracting Embeddings")
        logger.info("=" * 60)
        embeddings, ids = extract_embeddings(
            model, data_loader, device, args.num_samples
        )

        # Save embeddings
        embeddings_file = output_dir / 'embeddings.npz'
        np.savez(embeddings_file, embeddings=embeddings, ids=ids)
        logger.info(f"Embeddings saved to: {embeddings_file}")

        # Visualize embedding distribution
        visualize_embedding_distribution(embeddings, output_dir)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - Metrics: {metrics_file}")
    logger.info(f"  - Visualizations: {vis_dir}")
    if args.extract_embeddings:
        logger.info(f"  - Embeddings: {embeddings_file}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

