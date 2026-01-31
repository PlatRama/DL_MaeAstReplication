# DL_MaeAstReplication

This repository contains a replication of a **MAE‑AST (Masked Autoencoding Audio Spectrogram Transformer)** model, supporting both **self‑supervised pretraining** and **supervised fine‑tuning** for downstream audio classification tasks.

---

## Project Goals

- Faithfully replicate the MAE‑AST architecture and training pipeline proposed in the original paper
- Clearly separate **model**, **data**, **losses**, **training loop**, and **execution scripts**
- Enable easy experimentation with:
  - different masking strategies (patch / frame, random / chunk)
  - fine‑tuning with or without the decoder
  - extension to new audio datasets

---

## Repository Structure

<details>
<summary><strong>Click to expand the code structure</strong></summary>

```
mae_ast/
│
├── configs/                       # YAML configuration files
│   ├── pretrain_*_config.yaml     # Pretraining configurations
│   └── finetune_*_config.yaml     # Fine‑tuning configurations
│
├── src/
│   ├── data/                      # Dataset, collator, and transforms
│   │   ├── dataset.py             # TSV‑based audio dataset
│   │   ├── collator.py            # Batch collation (masking, padding)
│   │   └── transforms.py          # Audio feature extraction
│   │
│   ├── models/                    # Model architecture
│   │   ├── mae_ast.py             # Main MAE‑AST model
│   │   ├── transformer.py         # Transformer encoder / decoder
│   │   ├── masking.py             # Masking strategies
│   │   └── positional_encoding.py # Positional encoding
│   │
│   ├── losses/                    # Loss functions
│   │   ├── reconstruction_loss.py # MAE reconstruction loss
│   │   ├── contrastive_loss.py    # Contrastive loss
│   │   └── combined_loss.py       # Combined loss definition
│   │
│   ├── training/                  # Training logic
│   │   ├── trainer.py             # Training / validation loop
│   │   ├── optimizer.py           # Optimizer setup
│   │   └── scheduler.py           # Learning rate scheduler
│   │
│   ├── utils/                     # General utilities
│   │   ├── checkpoint.py          # Model checkpointing
│   │   ├── logger.py              # Logging utilities
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── misc.py                # Miscellaneous helpers
│   │
│   └── scripts/                   # Executable entry points
│       ├── pretrain.py            # MAE pretraining
│       ├── finetune.py            # Supervised fine‑tuning
│       └── evaluate.py            # Model evaluation
│
├── requirements.txt
└── README.md
```

</details>

---

## Non‑versioned Directories

The following directories are **not included** in the repository and are created dynamically based on the configuration files:

- `data/manifests/`  
  Contains dataset **TSV manifest files** (train/val/test).

- `checkpoints/`  
  Stores pretrained and fine‑tuned model checkpoints.

- `logs/`  
  Stores training logs, validation outputs, and metrics.

Paths and filenames are configurable via YAML.

---

## Configuration System

All runtime behavior is controlled through files in `configs/`:

- **Pretraining**
  - `pretrain_patch_random_config.yaml`
  - `pretrain_patch_chunk_config.yaml`
  - `pretrain_frame_random_config.yaml`
  - `pretrain_frame_chunk_config.yaml`

- **Fine‑tuning**
  - `finetune_patch_config.yaml`
  - `finetune_frame_config.yaml`

Configurations define:

- model architecture (dimensions, depth, masking ratio)
- dataset paths and manifest locations
- optimizer and scheduler parameters
- logging and checkpoint behavior

---

## Usage

### To install run
```bash
pip install -r requirements.txt
```

### Pretraining

```bash
python -m src.scripts.pretrain.py \
  --config configs/pretrain_patch_random_config.yaml
```

### Fine‑tuning

```bash
python -m src.scripts.finetune.py \
  --config configs/finetune_patch_config.yaml \
  --task esc50 
```

Other option for fine-tuning:
- possible task: audioset, esc50, speechcommands_v1, speechcommands_v2, voxceleb, iemocap
- it is possible to use the encoder only: --freeze_encoder

### Evaluation

```bash
python -m src.scripts.evaluate.py \
  --config configs/finetune_patch_config.yaml \
  --checkpoint path/to/checkpoint.pt
```

---

## Experiments

### Pretraining Setup

Due to hardware limitations, pretraining was performed on a **reduced subset** of the original data and number of **iterations**. Specifically:

- **Datasets used**:
  - A subset of **AudioSet** consisting of **50,000 audio samples**
  - **LibriSpeech**

- **Training details**:
  - Number of training iterations: **30,000**
  - Patch size: **24 × 24**
  - Input representation: audio spectrogram patches / frames

- **Masking configurations**:
  
  Pretraining was executed independently for **all combinations** of:

  - Patch-based masking vs. frame-based masking
  - Random masking vs. chunk-based masking

  resulting in **four distinct pretrained models**, each trained under the same iteration budget and patch size.

These experiments were intended to validate the correctness and stability of the implementation rather than to fully match the scale of the original paper.

---

### Fine-tuning Setup

Fine-tuning experiments were conducted using the pretrained checkpoints obtained from all four pretraining configurations.

- **Downstream datasets**:
  - **ESC-50**
  - **SpeechCommands v1**
  - **SpeechCommands v2**
  - **IEMOCAP**

- **Fine-tuning strategies**:
  - Encoder **frozen** (linear / shallow head training)
  - Encoder **unfrozen** (full model fine-tuning)

Each downstream dataset was fine-tuned using **all four pretrained models**, resulting in a comprehensive comparison across:

- masking strategy (patch vs. frame)
- masking pattern (random vs. chunk)
- encoder freezing strategy

---

