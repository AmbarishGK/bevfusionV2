# MoRAL Pipeline — UV Environment Setup & Run Guide

## Prerequisites

- Python 3.10 or 3.11
- NVIDIA GPU with CUDA 12.x drivers
- `uv` package manager

---

## Install `uv`

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Verify
uv --version
```

---

## Create Environment

```bash
cd /path/to/bevfusionV2/moral_pipeline

# Create venv with Python 3.11
uv venv .venv --python 3.11

# Activate
source .venv/bin/activate

# Install all dependencies
uv pip install -r requirements.txt

# Optional: FlashAttention-2 (H100/A100 only)
uv pip install flash-attn --no-build-isolation

# Optional: W&B logging
uv pip install wandb

# Optional: Unsloth (faster inference)
uv pip install unsloth
```

---

## HuggingFace Authentication

```bash
# Get your token: https://huggingface.co/settings/tokens
# Needs READ access for downloading models
# Needs WRITE access for uploading fine-tuned models

# Login (stores token in ~/.cache/huggingface/)
huggingface-cli login

# Or set environment variable
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Enable fast downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## Running Each Script

### 1. Training (2B — for testing on consumer GPU)
```bash
source .venv/bin/activate

# Dry run first (2 steps, ~2 min)
python train_cosmos2b.py \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos2b_condD \
    --dry_run

# Full training
python train_cosmos2b.py \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos2b_condD \
    --epochs 3
```

### 2. Training (8B — cloud GPU)
```bash
source .venv/bin/activate

# Dry run
python train_cosmos8b.py \
    --model nvidia/Cosmos-Reason2-8B \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos8b_condD \
    --tf32 --grad_ckpt \
    --dry_run

# Full training on H100
python train_cosmos8b.py \
    --model nvidia/Cosmos-Reason2-8B \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos8b_condD \
    --tf32 --torch_compile --grad_ckpt \
    --batch_size 2 --grad_accum 8 \
    --epochs 3

# Resume from checkpoint
python train_cosmos8b.py \
    --model nvidia/Cosmos-Reason2-8B \
    --resume \
    --output_dir saves/cosmos8b_condD \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --tf32 --torch_compile --grad_ckpt \
    --batch_size 2 --grad_accum 8
```

### 3. Training (Unified Script — recommended)
```bash
source .venv/bin/activate

# Auto-detects GPU and applies optimal settings
python antigravity_suggestions/train_cosmos_unified.py \
    --model nvidia/Cosmos-Reason2-8B \
    --profile auto \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos8b_condD \
    --dry_run

# With HF upload after training
python antigravity_suggestions/train_cosmos_unified.py \
    --model nvidia/Cosmos-Reason2-8B \
    --profile h100 \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos8b_condD \
    --hf_upload_repo AmbarishGK/moral-cosmos8b-condD
```

### 4. Evaluation (Zero-shot)
```bash
source .venv/bin/activate

# Evaluate base model (zero-shot)
python evaluate_zeroshot.py \
    --model nvidia/Cosmos-Reason2-8B \
    --val_file 02_cosmos_integration/hf_data/local_conditionB_val.jsonl \
    --input_level img \
    --condition B \
    --max_samples 200

# Evaluate fine-tuned model
python evaluate_zeroshot.py \
    --model saves/cosmos8b_condD/best_model \
    --val_file 02_cosmos_integration/hf_data/local_conditionB_val.jsonl \
    --input_level img \
    --condition B

# Camera-only baseline (no BEV)
python evaluate_zeroshot.py \
    --model nvidia/Cosmos-Reason2-8B \
    --val_file 02_cosmos_integration/hf_data/local_conditionB_val.jsonl \
    --input_level cam_only \
    --condition B
```

---

## Checkpoint Structure

After training, the output directory looks like:
```
saves/cosmos8b_condD/
├── checkpoint-100/          ← Step 100 checkpoint
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── trainer_state.pt     ← Optimizer/scheduler/RNG state
│   └── ...
├── checkpoint-200/
├── best_model/              ← Best validation loss
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── trainer_state.pt
│   └── ...
├── adapter_config.json      ← Final model
├── adapter_model.safetensors
└── ...
```

To resume: `--resume` picks up from the latest `checkpoint-*`.
To use for inference: point `--model` to `best_model/` or the output dir.
