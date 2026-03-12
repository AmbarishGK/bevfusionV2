# ══════════════════════════════════════════════════════════════
# MoRAL Training — Docker Image
# Pre-compiles FlashAttention-2 so you never wait for it
#
# BUILD:
#   docker build -t moral-train:latest -f Dockerfile .
#
# RUN (H100, mount data + output):
#   docker run --gpus all \
#     -v /path/to/data:/workspace/data \
#     -v /path/to/saves:/workspace/saves \
#     -e HF_TOKEN=hf_your_token \
#     moral-train:latest \
#     python train_cosmos_unified.py \
#       --model nvidia/Cosmos-Reason2-8B \
#       --profile h100 \
#       --train_file data/clean_conditionD_train.jsonl \
#       --val_file   data/clean_conditionD_val.jsonl \
#       --output_dir saves/cosmos8b_condD \
#       --epochs 3
#
# DRY RUN:
#   docker run --gpus all \
#     -v /path/to/data:/workspace/data \
#     -e HF_TOKEN=hf_your_token \
#     moral-train:latest \
#     python train_cosmos_unified.py --dry_run \
#       --model nvidia/Cosmos-Reason2-8B \
#       --train_file data/clean_conditionD_train.jsonl \
#       --val_file   data/clean_conditionD_val.jsonl
#
# INTERACTIVE (debugging):
#   docker run --gpus all -it \
#     -v /path/to/data:/workspace/data \
#     -v /path/to/saves:/workspace/saves \
#     -e HF_TOKEN=hf_your_token \
#     moral-train:latest bash
# ══════════════════════════════════════════════════════════════

# ── Stage 1: CUDA + PyTorch base ─────────────────────────────
# PyTorch NGC image — includes CUDA 12.4, cuDNN, NCCL, PyTorch 2.4+
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System deps ──────────────────────────────────────────────
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        git wget curl htop screen tmux \
        build-essential ninja-build && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python deps (cached layer — only rebuilds if requirements change) ─
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── FlashAttention-2 (pre-compiled — the big time saver) ─────
# This takes 5-10 min to build but is cached in the Docker layer.
# Every subsequent `docker run` gets it instantly.
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# ── Optional: W&B ────────────────────────────────────────────
RUN pip install --no-cache-dir wandb

# ── Copy training scripts ────────────────────────────────────
COPY antigravity_suggestions/train_cosmos_unified.py /workspace/train_cosmos_unified.py
COPY evaluate_zeroshot.py /workspace/evaluate_zeroshot.py
COPY scene_utils.py /workspace/scene_utils.py
COPY train_cosmos2b.py /workspace/train_cosmos2b.py
COPY train_cosmos8b.py /workspace/train_cosmos8b.py

# Copy the batch run scripts
COPY antigravity_suggestions/run_all_finetune.sh /workspace/run_all_finetune.sh
COPY antigravity_suggestions/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/run_all_finetune.sh /workspace/entrypoint.sh

# ── HuggingFace config ───────────────────────────────────────
# HF_TOKEN is passed at runtime via -e HF_TOKEN=...
# Model cache goes to /workspace/models (mount a volume for persistence)
ENV HF_HOME=/workspace/hf_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV TRANSFORMERS_CACHE=/workspace/hf_cache

# ── Entrypoint ───────────────────────────────────────────────
# Default: run the unified trainer. Override with any command.
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["bash"]
