#!/bin/bash
# MoRAL — Smart Parallel Ablation Launcher
# RTX 4090 (24GB): 3 processes in parallel max
#
# CURRENT STATE:
#   Batches 1-3: Zero-shot Cosmos-Reason2-2B (running now)
#   Batches 4-6: Fine-tuned Cosmos (DISABLED — run after fine-tuning on A100)
#
# Usage:
#   bash run_all_evals.sh 50     # test 50 samples
#   bash run_all_evals.sh        # full run ~30h (batches 1-3 only)
#
# Monitor:  watch -n5 'cat logs/progress.txt'
# Kill all: cat logs/*.pid | xargs kill 2>/dev/null

MAX=${1:-""}

COSMOS2B="nvidia/Cosmos-Reason2-2B"
COND_B="02_cosmos_integration/hf_data/local_conditionB_val_zeroshot_v2.jsonl" 
COND_D="02_cosmos_integration/hf_data/local_conditionD_val_zeroshot_v2.jsonl" 
OUT_ZS="saves/zeroshot_results"
CLEAN="outputs/03_clean_bev"

mkdir -p logs "$OUT_ZS"
rm -f logs/*.pid
> logs/progress.txt

run_one() {
    local tag=$1; shift
    local logfile="logs/${tag}.log"
    local start=$(date +%s)
    echo "[$(date '+%H:%M')] START  $tag" | tee -a logs/progress.txt
    python evaluate_zeroshot.py "$@" \
        ${MAX:+--max_samples $MAX} \
        > "$logfile" 2>&1
    local code=$?
    local elapsed=$(( ($(date +%s) - start) / 60 ))
    if [ $code -eq 0 ]; then
        echo "[$(date '+%H:%M')] DONE   $tag  (${elapsed}min)" | tee -a logs/progress.txt
    else
        echo "[$(date '+%H:%M')] FAILED $tag  (code=$code, see $logfile)" | tee -a logs/progress.txt
    fi
}

run_batch() {
    local batch_name=$1; shift
    echo "" | tee -a logs/progress.txt
    echo "══ BATCH: $batch_name ══" | tee -a logs/progress.txt
    local pids=()
    for job in "$@"; do
        eval "$job" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait $pid
    done
    echo "══ BATCH DONE: $batch_name ══" | tee -a logs/progress.txt
}

echo "MoRAL Ablation — $(date)" | tee logs/progress.txt
echo "MAX_SAMPLES: ${MAX:-all}" | tee -a logs/progress.txt

# ── BATCH 1: Zero-shot condB (3 parallel) ────────────────────────────
# Cosmos-2B zero-shot, GT-box BEV, condition B (LiDAR only)
run_batch "1/3 ZeroShot-condB" \
    "run_one zs__cam_only__B --model $COSMOS2B --val_file $COND_B --input_level cam_only --condition B --output_dir $OUT_ZS" \
    "run_one zs__img__B      --model $COSMOS2B --val_file $COND_B --input_level img       --condition B --output_dir $OUT_ZS" \
    "run_one zs__img_det__B  --model $COSMOS2B --val_file $COND_B --input_level img+det   --condition B --output_dir $OUT_ZS"

# ── BATCH 2: Zero-shot condD (3 parallel) ────────────────────────────
# Cosmos-2B zero-shot, GT-box BEV, condition D (LiDAR+Radar)
run_batch "2/3 ZeroShot-condD" \
    "run_one zs__cam_only__D --model $COSMOS2B --val_file $COND_D --input_level cam_only --condition D --output_dir $OUT_ZS" \
    "run_one zs__img__D      --model $COSMOS2B --val_file $COND_D --input_level img       --condition D --output_dir $OUT_ZS" \
    "run_one zs__img_det__D  --model $COSMOS2B --val_file $COND_D --input_level img+det   --condition D --output_dir $OUT_ZS"

# ── BATCH 3a: GT-box BEV only, no camera (2 parallel) ────────────────
# Professor: does BEV alone (with boxes) work without camera?
# condB = LiDAR BEV + boxes, condD = LiDAR+Radar BEV + boxes
run_batch "3a ZeroShot-BEVonly-withBoxes" \
    "run_one zs__bev_only__B --model $COSMOS2B --val_file $COND_B --input_level bev_only --condition B --output_dir $OUT_ZS" \
    "run_one zs__bev_only__D --model $COSMOS2B --val_file $COND_D --input_level bev_only --condition D --output_dir $OUT_ZS"

# ── BATCH 3b: Clean BEV + camera, no boxes (4 parallel) ──────────────
# Professor: BEV point cloud + camera but NO drawn boxes
# clean_lidar = lidar only point cloud + CAM_FRONT
# clean_radar = lidar+radar point cloud + CAM_FRONT
run_batch "3b ZeroShot-cleanBEV-withCAM" \
    "run_one zs__clean_lidar__B --model $COSMOS2B --val_file $COND_B --input_level clean_lidar --condition B --output_dir $OUT_ZS --bev_root $CLEAN" \
    "run_one zs__clean_radar__B --model $COSMOS2B --val_file $COND_B --input_level clean_radar --condition B --output_dir $OUT_ZS --bev_root $CLEAN" \
    "run_one zs__clean_lidar__D --model $COSMOS2B --val_file $COND_D --input_level clean_lidar --condition D --output_dir $OUT_ZS --bev_root $CLEAN" \
    "run_one zs__clean_radar__D --model $COSMOS2B --val_file $COND_D --input_level clean_radar --condition D --output_dir $OUT_ZS --bev_root $CLEAN"

# ── BATCH 3c: Clean BEV only, no camera, no boxes (4 parallel) ───────
# Professor: raw point cloud alone — no camera, no boxes
# This is the hardest condition — pure sensor reasoning
run_batch "3c ZeroShot-cleanBEV-noCAM" \
    "run_one zs__clean_lidar_only__B --model $COSMOS2B --val_file $COND_B --input_level clean_lidar_only --condition B --output_dir $OUT_ZS --bev_root $CLEAN" \
    "run_one zs__clean_radar_only__B --model $COSMOS2B --val_file $COND_B --input_level clean_radar_only --condition B --output_dir $OUT_ZS --bev_root $CLEAN" \
    "run_one zs__clean_lidar_only__D --model $COSMOS2B --val_file $COND_D --input_level clean_lidar_only --condition D --output_dir $OUT_ZS --bev_root $CLEAN" \
    "run_one zs__clean_radar_only__D --model $COSMOS2B --val_file $COND_D --input_level clean_radar_only --condition D --output_dir $OUT_ZS --bev_root $CLEAN"

# ════════════════════════════════════════════════════════════════════
# BATCHES 4-6 (fine-tuned Cosmos) — UNCOMMENT after fine-tuning on A100
# ════════════════════════════════════════════════════════════════════
# OUT_FT="saves/finetuned_results"
# COSMOS2B_FT="saves/cosmos2b_finetuned"    # path after fine-tuning
# COSMOS8B_FT="saves/cosmos8b_finetuned"    # path after fine-tuning
#
# run_batch "4 FineTuned-Cosmos2B-condB" \
#     "run_one ft2b__cam_only__B --model $COSMOS2B_FT --val_file $COND_B --input_level cam_only --condition B --output_dir $OUT_FT" \
#     "run_one ft2b__img__B      --model $COSMOS2B_FT --val_file $COND_B --input_level img       --condition B --output_dir $OUT_FT" \
#     "run_one ft2b__img_det__B  --model $COSMOS2B_FT --val_file $COND_B --input_level img+det   --condition B --output_dir $OUT_FT"
#
# run_batch "5 FineTuned-Cosmos2B-condD" \
#     "run_one ft2b__cam_only__D --model $COSMOS2B_FT --val_file $COND_D --input_level cam_only --condition D --output_dir $OUT_FT" \
#     "run_one ft2b__img__D      --model $COSMOS2B_FT --val_file $COND_D --input_level img       --condition D --output_dir $OUT_FT" \
#     "run_one ft2b__img_det__D  --model $COSMOS2B_FT --val_file $COND_D --input_level img+det   --condition D --output_dir $OUT_FT"
#
# run_batch "6 FineTuned-cleanBEV" \
#     "run_one ft2b__clean_lidar__B --model $COSMOS2B_FT --val_file $COND_B --input_level clean_lidar --condition B --output_dir $OUT_FT --bev_root $CLEAN" \
#     "run_one ft2b__clean_radar__B --model $COSMOS2B_FT --val_file $COND_B --input_level clean_radar --condition B --output_dir $OUT_FT --bev_root $CLEAN" \
#     "run_one ft2b__clean_lidar__D --model $COSMOS2B_FT --val_file $COND_D --input_level clean_lidar --condition D --output_dir $OUT_FT --bev_root $CLEAN" \
#     "run_one ft2b__clean_radar__D --model $COSMOS2B_FT --val_file $COND_D --input_level clean_radar --condition D --output_dir $OUT_FT --bev_root $CLEAN"

echo "" | tee -a logs/progress.txt
echo "ALL DONE — $(date)" | tee -a logs/progress.txt
echo "Results → $OUT_ZS"
ls -lh "$OUT_ZS"/summary_*.json 2>/dev/null
