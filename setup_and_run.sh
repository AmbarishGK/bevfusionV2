#!/usr/bin/env bash
# =============================================================================
# MoRAL Pipeline — AWS Cosmos 8B Setup & Run Script
# =============================================================================
# Run this ONCE on a fresh g5.2xlarge (A10G 24GB) instance.
# It installs all dependencies, starts the vLLM server, and runs QA generation
# for both condition B and condition D.
#
# USAGE (from your laptop):
#   # 1. Copy files to AWS first:
#   bash upload_to_aws.sh <instance-ip>
#
#   # 2. SSH in and run this script:
#   ssh -i localbevfusion.pem ubuntu@<instance-ip>
#   bash ~/setup_and_run.sh
#
# Or run fully remote (after uploading):
#   ssh -i localbevfusion.pem ubuntu@<instance-ip> 'bash ~/setup_and_run.sh'
# =============================================================================

set -euo pipefail  # exit on error, undefined vars, pipe failures

# ── Colours for output ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Configuration ─────────────────────────────────────────────────────────────
PIPELINE_ROOT="$HOME"
VENV_DIR="$HOME/cosmos_env"
HF_TOKEN="${HF_TOKEN:-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}"
MODEL="nvidia/Cosmos-Reason2-8B"
VLLM_PORT=8000
MAX_MODEL_LEN=16384
LOG_DIR="$HOME/logs"
VLLM_LOG="$LOG_DIR/vllm_server.log"
QA_LOG_B="$LOG_DIR/qa_generation_B.log"
QA_LOG_D="$LOG_DIR/qa_generation_D.log"

mkdir -p "$LOG_DIR"

echo ""
echo "============================================================"
echo "  MoRAL Pipeline — Cosmos 8B Setup & Run"
echo "  $(date)"
echo "============================================================"
echo ""

# ── Step 1: Verify GPU ────────────────────────────────────────────────────────
info "Step 1/7: Verifying GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. Is this a GPU instance?"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
success "GPU: $GPU_NAME  |  VRAM: $GPU_MEM"
if [[ "$GPU_MEM" < "20000" ]]; then
    warn "Less than 20GB VRAM detected. Cosmos-8B needs ~18GB. Proceeding anyway."
fi

# ── Step 2: Install uv ────────────────────────────────────────────────────────
info "Step 2/7: Installing uv..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
    success "uv installed"
else
    source "$HOME/.local/bin/env" 2>/dev/null || true
    success "uv already installed: $(uv --version)"
fi

# ── Step 3: Create virtualenv and install packages ────────────────────────────
info "Step 3/7: Creating virtualenv and installing packages..."
if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR"
    success "Virtualenv created at $VENV_DIR"
else
    success "Virtualenv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

info "  Installing transformers, vllm, openai..."
uv pip install \
    "transformers>=4.57.0" \
    "vllm>=0.11.0" \
    "openai>=1.0.0" \
    "huggingface_hub>=0.23.0" \
    "torch>=2.1.0" \
    "numpy" \
    --quiet

success "All packages installed"

# Verify key packages
python3 -c "import vllm; print(f'  vllm {vllm.__version__}')"
python3 -c "import transformers; print(f'  transformers {transformers.__version__}')"
python3 -c "import openai; print(f'  openai {openai.__version__}')"

# ── Step 4: HuggingFace login + model prefetch ────────────────────────────────
info "Step 4/7: HuggingFace login and model download..."
if [ -z "$HF_TOKEN" ]; then
    error "HF_TOKEN not set. Set it: export HF_TOKEN=hf_xxxx"
fi

huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
success "HuggingFace login OK"

info "  Checking if $MODEL is already cached..."
if python3 -c "
from huggingface_hub import try_to_load_from_cache
result = try_to_load_from_cache('nvidia/Cosmos-Reason2-8B', 'config.json')
import sys; sys.exit(0 if result else 1)
" 2>/dev/null; then
    success "Model already cached — skipping download"
else
    info "  Downloading $MODEL (~16GB, this takes 5-10 min)..."
    huggingface-cli download "$MODEL" \
        --local-dir-use-symlinks False \
        --quiet
    success "Model downloaded"
fi

# ── Step 5: Verify input files ────────────────────────────────────────────────
info "Step 5/7: Verifying input files..."
CLEAN_SCENES=("scene-0061" "scene-0553" "scene-0655" "scene-0757" "scene-0796"
              "scene-0916" "scene-1077" "scene-1094" "scene-1100")

MISSING=0
for SCENE in "${CLEAN_SCENES[@]}"; do
    for COND in "01_gt_annotations" "02_gt_with_radar"; do
        DIR="$PIPELINE_ROOT/outputs/$COND/$SCENE"
        if [ ! -d "$DIR" ]; then
            warn "Missing: $DIR"
            MISSING=$((MISSING + 1))
        elif [ ! -f "$DIR/bev_map.png" ]; then
            warn "Missing bev_map.png in $DIR"
            MISSING=$((MISSING + 1))
        elif [ ! -f "$DIR/detections.json" ]; then
            warn "Missing detections.json in $DIR"
            MISSING=$((MISSING + 1))
        fi
    done
done

if [ $MISSING -gt 0 ]; then
    error "$MISSING required files/directories missing. Run upload_to_aws.sh first."
fi
success "All 9 scenes verified for both conditions B and D"

# ── Step 6: Start vLLM server ─────────────────────────────────────────────────
info "Step 6/7: Starting vLLM server..."

# Kill any existing vllm server
pkill -f "vllm serve" 2>/dev/null && sleep 2 || true

info "  Launching vLLM server (log: $VLLM_LOG)..."
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "$VLLM_PID" > "$LOG_DIR/vllm.pid"
info "  vLLM server PID: $VLLM_PID"

# Wait for server to be ready (model load takes 2-4 min)
info "  Waiting for vLLM server to be ready (model load ~3 min)..."
MAX_WAIT=300  # 5 minutes
WAITED=0
while true; do
    if curl -s "http://localhost:$VLLM_PORT/health" | grep -q "{}"; then
        break
    fi
    # Also check for error in log
    if grep -q "ERROR\|Error\|error" "$VLLM_LOG" 2>/dev/null && \
       ! grep -q "Uvicorn running" "$VLLM_LOG" 2>/dev/null; then
        # Check if it's a real error (not just a log-level message)
        if grep -q "CUDA out of memory\|RuntimeError\|Failed to" "$VLLM_LOG" 2>/dev/null; then
            error "vLLM server failed to start. Check $VLLM_LOG"
        fi
    fi
    if [ $WAITED -ge $MAX_WAIT ]; then
        error "vLLM server did not start within ${MAX_WAIT}s. Check $VLLM_LOG"
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo -n "."
done
echo ""
success "vLLM server ready on port $VLLM_PORT"

# Quick sanity check — confirm model is loaded
MODEL_CHECK=$(curl -s "http://localhost:$VLLM_PORT/v1/models" | python3 -c \
    "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")
info "  Loaded model: $MODEL_CHECK"

# ── Step 7: Run QA generation ─────────────────────────────────────────────────
info "Step 7/7: Running QA generation..."

# Ensure output directories exist
mkdir -p "$PIPELINE_ROOT/02_cosmos_integration/cosmos_qa"

# ── Condition B ──────────────────────────────────────────────────────────────
info "  Running Condition B (BEV + LiDAR GT)..."
echo "  Start: $(date)"
python3 "$PIPELINE_ROOT/generate_cosmos_qa.py" \
    --condition B \
    --api-mode local \
    --base-url "http://localhost:$VLLM_PORT/v1" \
    --pipeline-root "$PIPELINE_ROOT" \
    2>&1 | tee "$QA_LOG_B"

if grep -q "ERROR\|Traceback" "$QA_LOG_B"; then
    warn "Condition B completed with errors — check $QA_LOG_B"
else
    success "Condition B complete"
fi

# ── Condition D ──────────────────────────────────────────────────────────────
info "  Running Condition D (BEV + LiDAR + Radar GT)..."
echo "  Start: $(date)"
python3 "$PIPELINE_ROOT/generate_cosmos_qa.py" \
    --condition D \
    --api-mode local \
    --base-url "http://localhost:$VLLM_PORT/v1" \
    --pipeline-root "$PIPELINE_ROOT" \
    2>&1 | tee "$QA_LOG_D"

if grep -q "ERROR\|Traceback" "$QA_LOG_D"; then
    warn "Condition D completed with errors — check $QA_LOG_D"
else
    success "Condition D complete"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  DONE — $(date)"
echo "============================================================"
echo ""
echo "Output files:"
echo "  $PIPELINE_ROOT/02_cosmos_integration/all_qa_conditionB_sharegpt.jsonl"
echo "  $PIPELINE_ROOT/02_cosmos_integration/all_qa_conditionD_sharegpt.jsonl"
echo "  $PIPELINE_ROOT/02_cosmos_integration/cosmos_qa/scene-*/qa_pairs_condition*.json"
echo ""
echo "Logs:"
echo "  vLLM server:  $VLLM_LOG"
echo "  Condition B:  $QA_LOG_B"
echo "  Condition D:  $QA_LOG_D"
echo ""
echo "Next: run download_from_aws.sh on your laptop to pull results back."
echo ""

# Count QA pairs generated
B_COUNT=$(cat "$PIPELINE_ROOT/02_cosmos_integration/all_qa_conditionB_sharegpt.jsonl" \
    2>/dev/null | wc -l || echo 0)
D_COUNT=$(cat "$PIPELINE_ROOT/02_cosmos_integration/all_qa_conditionD_sharegpt.jsonl" \
    2>/dev/null | wc -l || echo 0)
echo "  QA pairs generated: $B_COUNT condition B + $D_COUNT condition D = $((B_COUNT + D_COUNT)) total"
echo ""
