#!/bin/bash
# ══════════════════════════════════════════════════════════════
# MoRAL — Complete 4090 Run: 2B eval + all 8B zero-shots
# Run everything unattended overnight. Each step logs individually.
# ══════════════════════════════════════════════════════════════
set -e
cd ~/workspace/amb_ws/bevfusionV2/moral_pipeline

LOGDIR="logs"
mkdir -p "$LOGDIR"

run_eval() {
    local MODEL="$1"
    local VAL="$2"
    local LEVEL="$3"
    local COND="$4"
    local BEV_ARGS="$5"
    local MAX="$6"
    local OUTDIR="$7"
    local TAG="$8"

    local LOG="$LOGDIR/${TAG}.log"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  [$TAG] model=$(basename $MODEL) level=$LEVEL cond=$COND"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════"

    # Check if results already exist
    # Build expected filename to check
    if ls "$OUTDIR"/results_*__${LEVEL}__${COND}*.jsonl 1>/dev/null 2>&1; then
        echo "  ⚠️  Results may already exist — running anyway (will overwrite)"
    fi

    python evaluate_zeroshot.py \
        --model "$MODEL" \
        --val_file "$VAL" \
        --input_level "$LEVEL" \
        --condition "$COND" \
        $BEV_ARGS \
        --max_samples "$MAX" \
        --output_dir "$OUTDIR" \
        2>&1 | tee "$LOG"

    echo "  ✅ Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Log: $LOG"
}

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " MoRAL 4090 Overnight Run"
echo " Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── PART 1: 2B fine-tuned eval (1 missing) ──────────────────

echo "▓▓▓ PART 1: 2B Fine-tuned Evaluation (1 run) ▓▓▓"

run_eval \
    "saves/cosmos2b_condB_finetuned" \
    "02_cosmos_integration/hf_data/clean_conditionB_val.jsonl" \
    "clean_radar_only" "B" \
    "--bev_root outputs/03_clean_bev" \
    "1245" \
    "saves/finetuned_results" \
    "ft_2b__clean_radar_only__B"

echo ""
echo "▓▓▓ PART 1 COMPLETE — 2B fine-tuned is 100% done ▓▓▓"

# ── PART 2: 8B zero-shot missing (4 never-run) ──────────────

echo ""
echo "▓▓▓ PART 2: 8B Zero-shot Missing (4 runs) ▓▓▓"

run_eval \
    "nvidia/Cosmos-Reason2-8B" \
    "02_cosmos_integration/hf_data/local_conditionB_val.jsonl" \
    "cam_only" "B" \
    "" \
    "200" \
    "saves/zeroshot_results_8B" \
    "zs_8b__cam_only__B"

run_eval \
    "nvidia/Cosmos-Reason2-8B" \
    "02_cosmos_integration/hf_data/local_conditionD_val.jsonl" \
    "cam_only" "D" \
    "" \
    "200" \
    "saves/zeroshot_results_8B" \
    "zs_8b__cam_only__D"

run_eval \
    "nvidia/Cosmos-Reason2-8B" \
    "02_cosmos_integration/hf_data/local_conditionB_val.jsonl" \
    "clean_lidar_only" "B" \
    "--bev_root outputs/04_clean_bev_lidar_only" \
    "200" \
    "saves/zeroshot_results_8B" \
    "zs_8b__clean_lidar_only__B"

run_eval \
    "nvidia/Cosmos-Reason2-8B" \
    "02_cosmos_integration/hf_data/local_conditionB_val.jsonl" \
    "clean_radar_only" "B" \
    "--bev_root outputs/03_clean_bev" \
    "200" \
    "saves/zeroshot_results_8B" \
    "zs_8b__clean_radar_only__B"

echo ""
echo "▓▓▓ PART 2 COMPLETE — 8B missing zero-shots done ▓▓▓"

# ── PART 3: 8B zero-shot re-runs (4 broken D-condition) ─────

echo ""
echo "▓▓▓ PART 3: 8B Zero-shot Re-runs (4 broken D runs) ▓▓▓"

run_eval \
    "nvidia/Cosmos-Reason2-8B" \
    "02_cosmos_integration/hf_data/local_conditionD_val.jsonl" \
    "img" "D" \
    "" \
    "200" \
    "saves/zeroshot_results_8B" \
    "zs_8b__img__D_rerun"

run_eval \
    "nvidia/Cosmos-Reason2-8B" \
    "02_cosmos_integration/hf_data/local_conditionD_val.jsonl" \
    "img+det" "D" \
    "" \
    "200" \
    "saves/zeroshot_results_8B" \
    "zs_8b__img_det__D_rerun"

run_eval \
    "nvidia/Cosmos-Reason2-8B" \
    "02_cosmos_integration/hf_data/local_conditionD_val.jsonl" \
    "bev_only" "D" \
    "" \
    "200" \
    "saves/zeroshot_results_8B" \
    "zs_8b__bev_only__D_rerun"

run_eval \
    "nvidia/Cosmos-Reason2-8B" \
    "02_cosmos_integration/hf_data/local_conditionD_val.jsonl" \
    "clean_lidar" "D" \
    "--bev_root outputs/03_clean_bev" \
    "200" \
    "saves/zeroshot_results_8B" \
    "zs_8b__clean_lidar__D_rerun"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " ✅ ALL DONE"
echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Results in:"
echo "  saves/finetuned_results/    (2B fine-tuned)"
echo "  saves/zeroshot_results_8B/  (8B zero-shot)"
echo ""
echo "Individual logs in: logs/"
echo "  ft_2b__clean_radar_only__B.log"
echo "  zs_8b__cam_only__B.log"
echo "  zs_8b__cam_only__D.log"
echo "  zs_8b__clean_lidar_only__B.log"
echo "  zs_8b__clean_radar_only__B.log"
echo "  zs_8b__img__D_rerun.log"
echo "  zs_8b__img_det__D_rerun.log"
echo "  zs_8b__bev_only__D_rerun.log"
echo "  zs_8b__clean_lidar__D_rerun.log"
