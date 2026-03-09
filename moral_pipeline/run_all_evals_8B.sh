#!/bin/bash
# MoRAL — Cosmos-Reason2-8B Zero-Shot Ablation
# AWS g6e.xlarge (L40S 48GB) — us-east-2
#
# Runs 12 conditions (skip cam_only — not interesting for 8B)
# Est. ~17hrs total, ~$31 on-demand, ~$9.50 spot
#
# Setup on cloud instance:
#   git clone https://github.com/AmbarishGK/bevfusionV2
#   cd bevfusionV2/moral_pipeline
#   pip install transformers peft trl bitsandbytes accelerate qwen-vl-utils nltk Pillow --break-system-packages
#   huggingface-cli login   # paste HF token
#   huggingface-cli download AmbarishGK/MoRAL-nuscenes --repo-type dataset --local-dir .
#   DATA_ROOT=$(pwd)
#   for f in data/*.jsonl; do sed -i "s|outputs/|$DATA_ROOT/outputs/|g" "$f"; done
#   bash run_all_evals_8B.sh
#
# Monitor:  watch -n10 'cat logs/progress_8B.txt'
# Kill all: kill $(cat logs/8B_*.pid 2>/dev/null)

MAX=${1:-""}   # pass sample limit e.g. bash run_all_evals_8B.sh 50

COSMOS8B="nvidia/Cosmos-Reason2-8B"

COND_B="02_cosmos_integration/hf_data/local_conditionB_val_zeroshot_v2.jsonl"
COND_D="02_cosmos_integration/hf_data/local_conditionD_val_zeroshot_v2.jsonl"
OUT_ZS="saves/zeroshot_results_8B"
CLEAN="outputs/03_clean_bev"
CLEAN_LIDAR="outputs/04_clean_bev_lidar_only"

mkdir -p logs "$OUT_ZS"
> logs/progress_8B.txt

run_one() {
    local tag=$1; shift
    local logfile="logs/8B_${tag}.log"
    local start=$(date +%s)
    echo "[$(date '+%H:%M')] START  $tag" | tee -a logs/progress_8B.txt
    python evaluate_zeroshot.py "$@" \
        ${MAX:+--max_samples $MAX} \
        > "$logfile" 2>&1
    local code=$?
    local elapsed=$(( ($(date +%s) - start) / 60 ))
    if [ $code -eq 0 ]; then
        echo "[$(date '+%H:%M')] DONE   $tag  (${elapsed}min)" | tee -a logs/progress_8B.txt
    else
        echo "[$(date '+%H:%M')] FAILED $tag  (code=$code, see $logfile)" | tee -a logs/progress_8B.txt
    fi
}

# NOTE: 8B loads ~16GB in 4-bit — only 1 process at a time on L40S 48GB
# Running sequentially to avoid OOM

echo "MoRAL 8B Ablation — $(date)" | tee logs/progress_8B.txt
echo "Model: $COSMOS8B" | tee -a logs/progress_8B.txt
echo "MAX_SAMPLES: ${MAX:-all}" | tee -a logs/progress_8B.txt
echo "" | tee -a logs/progress_8B.txt

# ── GT-box BEV conditions ─────────────────────────────────────────────────────
echo "══ GT-box BEV ══" | tee -a logs/progress_8B.txt

run_one 8b__img__B \
    --model $COSMOS8B --val_file $COND_B \
    --input_level img --condition B \
    --output_dir $OUT_ZS

run_one 8b__img__D \
    --model $COSMOS8B --val_file $COND_D \
    --input_level img --condition D \
    --output_dir $OUT_ZS

run_one 8b__img_det__B \
    --model $COSMOS8B --val_file $COND_B \
    --input_level img+det --condition B \
    --output_dir $OUT_ZS

run_one 8b__img_det__D \
    --model $COSMOS8B --val_file $COND_D \
    --input_level img+det --condition D \
    --output_dir $OUT_ZS

run_one 8b__bev_only__B \
    --model $COSMOS8B --val_file $COND_B \
    --input_level bev_only --condition B \
    --output_dir $OUT_ZS

run_one 8b__bev_only__D \
    --model $COSMOS8B --val_file $COND_D \
    --input_level bev_only --condition D \
    --output_dir $OUT_ZS

# ── Clean BEV + camera (main claim) ──────────────────────────────────────────
echo "══ Clean BEV + Camera ══" | tee -a logs/progress_8B.txt

run_one 8b__clean_lidar__B \
    --model $COSMOS8B --val_file $COND_B \
    --input_level clean_lidar --condition B \
    --bev_root $CLEAN_LIDAR \
    --output_dir $OUT_ZS

run_one 8b__clean_lidar__D \
    --model $COSMOS8B --val_file $COND_D \
    --input_level clean_lidar --condition D \
    --bev_root $CLEAN \
    --output_dir $OUT_ZS

run_one 8b__clean_radar__B \
    --model $COSMOS8B --val_file $COND_B \
    --input_level clean_radar --condition B \
    --bev_root $CLEAN \
    --output_dir $OUT_ZS

run_one 8b__clean_radar__D \
    --model $COSMOS8B --val_file $COND_D \
    --input_level clean_radar --condition D \
    --bev_root $CLEAN \
    --output_dir $OUT_ZS

# ── Clean BEV only, no camera ─────────────────────────────────────────────────
echo "══ Clean BEV only (no camera) ══" | tee -a logs/progress_8B.txt

run_one 8b__clean_radar_only__D \
    --model $COSMOS8B --val_file $COND_D \
    --input_level clean_radar_only --condition D \
    --bev_root $CLEAN \
    --output_dir $OUT_ZS

run_one 8b__clean_lidar_only__D \
    --model $COSMOS8B --val_file $COND_D \
    --input_level clean_lidar_only --condition D \
    --bev_root $CLEAN \
    --output_dir $OUT_ZS

echo "" | tee -a logs/progress_8B.txt
echo "ALL DONE — $(date)" | tee -a logs/progress_8B.txt
echo "Results → $OUT_ZS"
ls -lh "$OUT_ZS"/summary_*.json 2>/dev/null
