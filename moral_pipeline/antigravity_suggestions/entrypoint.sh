#!/bin/bash
set -e

# ══════════════════════════════════════════════════════════════
# MoRAL Docker Entrypoint
# Handles HF auth, model download, and then runs whatever command
# ══════════════════════════════════════════════════════════════

echo "=== MoRAL Training Container ==="
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no GPU')"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  FlashAttn: $(python -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null || echo 'not installed')"

# ── HuggingFace auth ──
if [ -n "$HF_TOKEN" ]; then
    echo "  HF token: set ✅"
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
else
    echo "  ⚠️  HF_TOKEN not set. Model download may fail for gated models."
    echo "  Set with: -e HF_TOKEN=hf_your_token"
fi

# ── Download model if not cached ──
# Check if model arg is present and starts with nvidia/
MODEL_ARG=""
for arg in "$@"; do
    if [[ "$arg" == nvidia/* ]] || [[ "$arg" == Qwen/* ]]; then
        MODEL_ARG="$arg"
        break
    fi
done

if [ -n "$MODEL_ARG" ]; then
    CACHE_DIR="/workspace/hf_cache/hub/models--$(echo $MODEL_ARG | tr '/' '--')"
    if [ -d "$CACHE_DIR" ]; then
        echo "  Model cached: $MODEL_ARG ✅"
    else
        echo "  Downloading model: $MODEL_ARG (this happens once, cached after)..."
        python -c "
from huggingface_hub import snapshot_download
import time, sys
for attempt in range(5):
    try:
        snapshot_download('$MODEL_ARG', resume_download=True)
        print('  ✅ Model downloaded')
        break
    except Exception as e:
        wait = 30 * (2 ** attempt)
        print(f'  ⚠️  Download failed (attempt {attempt+1}/5): {e}. Retrying in {wait}s...')
        time.sleep(wait)
else:
    print('  ❌ Model download failed after 5 attempts')
    sys.exit(1)
"
    fi
fi

echo ""

# ── Run the command ──
exec "$@"
