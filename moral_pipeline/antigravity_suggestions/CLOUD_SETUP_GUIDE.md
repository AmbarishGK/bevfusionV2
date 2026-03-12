# Cloud GPU Setup Guide for MoRAL Training

This guide covers setting up a cloud GPU instance for fine-tuning Cosmos-Reason2-2B/8B.

---

## GPU Recommendations

| GPU | VRAM | Model | Batch Size | Est. Speed | Est. Cost |
|-----|------|-------|------------|------------|-----------|
| **H100 SXM** | 80 GB | 8B | 2 | ~4s/step | $2.50–3.50/hr |
| **A100 SXM** | 80 GB | 8B | 2 | ~6s/step | $1.50–2.50/hr |
| **A100 PCIe** | 40 GB | 8B | 1 | ~8s/step | $1.00–1.50/hr |
| **L40S** | 48 GB | 8B (tight) | 1 | ~10s/step | $0.80–1.50/hr |
| **RTX 4090** | 24 GB | 2B only | 1 | ~2.5s/step | $0.40–0.60/hr |

> **Recommendation:** H100 80GB for 8B training. A100 80GB is the best
> price-performance. L40S works but is tight for 8B — may need `--lora_rank 8`.

---

## Provider Options

### 1. Lambda Labs (Recommended for H100/A100)
```bash
# Create account: https://lambdalabs.com
# Launch instance: 1x H100 80GB SXM ($2.49/hr on-demand)
ssh ubuntu@<ip>
```

### 2. RunPod
```bash
# Create account: https://runpod.io
# Deploy pod: NVIDIA H100 SXM ($2.69/hr community cloud)
# Or: NVIDIA A100 80GB ($1.64/hr community cloud)
```

### 3. Vast.ai (Cheapest)
```bash
# Create account: https://vast.ai
# Search: H100 80GB or A100 80GB
# Typical: H100 ~$2.00/hr, A100 80GB ~$1.20/hr
vastai search offers --gpu_name H100 --min_gpu_ram 80 --order dph
```

### 4. AWS (EC2)
```bash
# p5.xlarge (H100 80GB) — $32/hr on-demand, ~$10/hr spot
# p4d.24xlarge (A100 80GB x8) — overkill for single-GPU
# g5.xlarge (A10G 24GB) — 2B only
aws ec2 run-instances --instance-type p5.xlarge --image-id ami-xxxxx
```

### 5. Google Cloud (GCP)
```bash
# a3-highgpu-1g (H100 80GB) — ~$3.50/hr
# a2-highgpu-1g (A100 40GB) — ~$1.20/hr
gcloud compute instances create moral-train \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --accelerator=type=nvidia-tesla-a100,count=1
```

### 6. Brev.dev (Easy Setup)
```bash
# Uses your cloud account (AWS/GCP/Lambda)
# One-click Jupyter/SSH with GPU
brev create moral-train --gpu a100-80gb
```

---

## Setup Script (Any Provider)

Run this after SSH into your instance:

```bash
#!/bin/bash
set -e

echo "=== MoRAL Training Setup ==="

# 1. System packages
sudo apt-get update -qq
sudo apt-get install -y git wget curl htop screen tmux -qq

# 2. Python (use uv for speed)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 3. Create project directory
mkdir -p ~/moral_pipeline
cd ~/moral_pipeline

# 4. Create virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# 5. Clone repo
git clone https://github.com/AmbarishGK/bevfusionV2.git ~/bevfusionV2
cp -r ~/bevfusionV2/moral_pipeline/* ~/moral_pipeline/

# 6. Install dependencies
uv pip install -r requirements.txt

# 7. Install FlashAttention-2 (H100/A100)
uv pip install flash-attn --no-build-isolation

# 8. Login to HuggingFace
# Get token: https://huggingface.co/settings/tokens (READ access minimum)
export HF_TOKEN="your_token_here"
huggingface-cli login --token $HF_TOKEN

# 9. Download model (with retry for large files)
# 2B: ~4GB, 8B: ~16GB
python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Cosmos-Reason2-8B',
                  local_dir='./models/Cosmos-Reason2-8B',
                  resume_download=True)
print('✅ Model downloaded')
"

# 10. Download dataset
huggingface-cli download AmbarishGK/MoRAL-nuScenes-BEV-850 \
    --repo-type dataset \
    --local-dir ./data/moral_outputs

# 11. Verify GPU
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU:  {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
import transformers; print(f'transformers: {transformers.__version__}')
import peft; print(f'peft: {peft.__version__}')
try:
    import flash_attn; print(f'flash-attn: {flash_attn.__version__}')
except: print('flash-attn: not installed')
"

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python train_cosmos_unified.py --model nvidia/Cosmos-Reason2-8B --dry_run"
```

---

## Training Commands

### Dry Run (verifies everything works)
```bash
cd ~/moral_pipeline
source .venv/bin/activate

python antigravity_suggestions/train_cosmos_unified.py \
    --model nvidia/Cosmos-Reason2-8B \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --dry_run
```

### Full Training (H100)
```bash
# Use screen/tmux so training survives SSH disconnect
screen -S train

python antigravity_suggestions/train_cosmos_unified.py \
    --model nvidia/Cosmos-Reason2-8B \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos8b_condD_finetuned \
    --profile h100 \
    --epochs 3 \
    --save_steps 100 \
    --save_total_limit 5

# Detach screen: Ctrl+A, D
# Reattach: screen -r train
```

### Full Training (A100)
```bash
python antigravity_suggestions/train_cosmos_unified.py \
    --model nvidia/Cosmos-Reason2-8B \
    --profile a100 \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos8b_condD_finetuned \
    --epochs 3
```

### Full Training (L40S — tight VRAM)
```bash
python antigravity_suggestions/train_cosmos_unified.py \
    --model nvidia/Cosmos-Reason2-8B \
    --profile l40s \
    --lora_rank 8 \
    --max_pixels 262144 \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos8b_condD_finetuned
```

### Resume Training
```bash
python antigravity_suggestions/train_cosmos_unified.py \
    --model nvidia/Cosmos-Reason2-8B \
    --resume \
    --output_dir saves/cosmos8b_condD_finetuned \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl
```

---

## Monitoring Training

### GPU utilization
```bash
# Watch GPU usage (refresh every 1s)
watch -n 1 nvidia-smi

# Or continuous logging
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total \
    --format=csv -l 10 > gpu_log.csv &
```

### Training logs
```bash
# Follow training output
tail -f saves/cosmos8b_condD_finetuned/training.log

# Check disk usage of checkpoints
du -sh saves/cosmos8b_condD_finetuned/checkpoint-*
```

### Download checkpoints to local machine
```bash
# From your LOCAL machine
rsync -avz --progress \
    ubuntu@<IP>:~/moral_pipeline/saves/cosmos8b_condD_finetuned/best_model/ \
    ./best_model/
```

---

## Post-Training

### Upload model to HuggingFace
```bash
# Create repo first
huggingface-cli repo create AmbarishGK/moral-cosmos8b-condD --type model

# Upload
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    repo_id='AmbarishGK/moral-cosmos8b-condD',
    folder_path='saves/cosmos8b_condD_finetuned/best_model',
    repo_type='model',
)
print('✅ Uploaded')
"
```

### Run evaluation
```bash
python evaluate_zeroshot.py \
    --model saves/cosmos8b_condD_finetuned/best_model \
    --val_file 02_cosmos_integration/hf_data/local_conditionB_val.jsonl \
    --input_level img \
    --condition B \
    --max_samples 200
```

---

## Cost Estimation

For 850 scenes (~2000 training samples), 3 epochs:
- **H100**: ~2-3 hours × $2.50 = **$6–8**
- **A100**: ~4-5 hours × $1.50 = **$6–8**
- **L40S**: ~6-8 hours × $1.00 = **$6–8**

> Budget $20-30 for training (including dry runs, debugging, and evaluations).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM (Out of Memory) | Reduce `--batch_size 1`, `--lora_rank 8`, `--max_pixels 262144` |
| HF 429 rate limit | The unified script handles this automatically with retry |
| FlashAttention build fails | `pip install flash-attn --no-build-isolation` or skip with `--attn_impl sdpa` |
| `torch.compile` errors | Skip with `--no-torch-compile` or use newer PyTorch |
| Slow data loading | Increase `--num_workers 8`, ensure data is on SSD |
| SSH disconnect kills training | ALWAYS use `screen` or `tmux` |
| Checkpoint disk full | Set `--save_total_limit 3` to keep fewer checkpoints |
