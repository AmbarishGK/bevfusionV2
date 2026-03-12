#!/bin/bash
set -e

# ══════════════════════════════════════════════════════════════════════════════
# MoRAL — Batch Fine-tuning & Zero-shot Runner
# ══════════════════════════════════════════════════════════════════════════════
#
# Runs all remaining training and evaluation jobs in priority order.
# Each job saves checkpoints and can be resumed if interrupted.
#
# USAGE:
#   # Run everything (inside Docker or direct machine):
#   bash run_all_finetune.sh
#
#   # Skip to a specific phase:
#   bash run_all_finetune.sh --start-phase 3
#
#   # Dry run (test all commands without training):
#   bash run_all_finetune.sh --dry-run
#
# REQUIRES:
#   - HF_TOKEN env var set
#   - GPU with >=48GB VRAM (H100/A100 for 8B, or 4090 for 2B-only)
#   - Training data in 02_cosmos_integration/hf_data/
# ══════════════════════════════════════════════════════════════════════════════

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"

# Default paths — override with env vars if needed
TRAIN_DIR="${TRAIN_DATA_DIR:-$PIPELINE_DIR/02_cosmos_integration/hf_data}"
SAVE_DIR="${SAVE_ROOT:-$PIPELINE_DIR/saves}"
EVAL_DIR="${EVAL_ROOT:-$SAVE_DIR/zeroshot_results}"

# Which trainer to use
TRAINER="${TRAINER_SCRIPT:-$SCRIPT_DIR/train_cosmos_unified.py}"
EVALUATOR="${EVAL_SCRIPT:-$PIPELINE_DIR/evaluate_zeroshot.py}"

# Default training args
COMMON_TRAIN_ARGS="--epochs 3 --save_steps 100 --save_total_limit 5 --resume"
H100_ARGS="--profile h100"
DRY_RUN_ARGS=""

# Parse CLI args
START_PHASE=1
for arg in "$@"; do
    case $arg in
        --start-phase=*|--start-phase)
            START_PHASE="${arg#*=}"
            if [ "$START_PHASE" = "--start-phase" ]; then
                shift; START_PHASE="$1"
            fi
            ;;
        --dry-run)
            DRY_RUN_ARGS="--dry_run"
            echo "⚡ DRY RUN MODE — will test pipeline but not actually train"
            ;;
    esac
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " MoRAL Batch Training — Priority Queue"
echo "═══════════════════════════════════════════════════════════"
echo "  Trainer:    $TRAINER"
echo "  Train data: $TRAIN_DIR"
echo "  Save dir:   $SAVE_DIR"
echo "  Start at:   Phase $START_PHASE"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Helper: run a training job ────────────────────────────────────────────────
run_train() {
    local MODEL="$1"
    local CONDITION="$2"   # B, D, E
    local OUTPUT_NAME="$3"
    local EXTRA_ARGS="$4"

    local TRAIN_FILE="$TRAIN_DIR/clean_condition${CONDITION}_train.jsonl"
    local VAL_FILE="$TRAIN_DIR/clean_condition${CONDITION}_val.jsonl"
    local OUTPUT_DIR="$SAVE_DIR/$OUTPUT_NAME"

    echo "──────────────────────────────────────────────────────"
    echo "  📦 Training: $OUTPUT_NAME"
    echo "  Model:     $MODEL"
    echo "  Condition: $CONDITION"
    echo "  Output:    $OUTPUT_DIR"
    echo "──────────────────────────────────────────────────────"

    if [ ! -f "$TRAIN_FILE" ]; then
        echo "  ⚠️  SKIP — train file not found: $TRAIN_FILE"
        return 0
    fi

    # Check if best_model already exists (already done)
    if [ -d "$OUTPUT_DIR/best_model" ] && [ -z "$DRY_RUN_ARGS" ]; then
        echo "  ✅ SKIP — best_model already exists"
        return 0
    fi

    python "$TRAINER" \
        --model "$MODEL" \
        --train_file "$TRAIN_FILE" \
        --val_file "$VAL_FILE" \
        --output_dir "$OUTPUT_DIR" \
        $COMMON_TRAIN_ARGS $H100_ARGS $DRY_RUN_ARGS $EXTRA_ARGS

    echo "  ✅ Done: $OUTPUT_NAME"
    echo ""
}

# ── Helper: run a zero-shot eval ──────────────────────────────────────────────
run_eval() {
    local MODEL="$1"
    local VAL_FILE="$2"
    local INPUT_LEVEL="$3"
    local CONDITION="$4"
    local EXTRA_ARGS="$5"

    local MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    local BEV_TAG=""
    if echo "$EXTRA_ARGS" | grep -q "bev_root"; then
        BEV_TAG="_cleanbev"
    fi
    local RUN_ID="${MODEL_SHORT}__${INPUT_LEVEL}__${CONDITION}${BEV_TAG}"

    echo "──────────────────────────────────────────────────────"
    echo "  🔍 Eval: $RUN_ID"
    echo "──────────────────────────────────────────────────────"

    # Check if results already exist
    if [ -f "$EVAL_DIR/summary_${RUN_ID}.json" ]; then
        echo "  ✅ SKIP — results already exist"
        return 0
    fi

    python "$EVALUATOR" \
        --model "$MODEL" \
        --val_file "$VAL_FILE" \
        --input_level "$INPUT_LEVEL" \
        --condition "$CONDITION" \
        --max_samples 200 \
        --output_dir "$EVAL_DIR" \
        $EXTRA_ARGS

    echo "  ✅ Done: $RUN_ID"
    echo ""
}

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: Complete 8B Zero-shot Gaps (3 missing runs — fast, ~30 min each)
# ══════════════════════════════════════════════════════════════════════════════
if [ "$START_PHASE" -le 1 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  PHASE 1: 8B Zero-shot Gaps (3 runs)               ║"
    echo "╚══════════════════════════════════════════════════════╝"

    VAL_B="$TRAIN_DIR/local_conditionB_val.jsonl"
    VAL_D="$TRAIN_DIR/local_conditionD_val.jsonl"
    CLEAN_BEV="--bev_root outputs/03_clean_bev"

    # Missing: cam_only D
    run_eval "nvidia/Cosmos-Reason2-8B" "$VAL_D" "cam_only" "D"

    # Missing: clean_lidar_only B
    run_eval "nvidia/Cosmos-Reason2-8B" "$VAL_B" "clean_lidar_only" "B" "$CLEAN_BEV"

    # Missing: clean_radar_only B
    run_eval "nvidia/Cosmos-Reason2-8B" "$VAL_B" "clean_radar_only" "B" "$CLEAN_BEV"

    echo "  ✅ Phase 1 complete — 8B zero-shot baseline is now complete"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: 2B Fine-tuning Gaps (highest priority — already have partial)
# ══════════════════════════════════════════════════════════════════════════════
if [ "$START_PHASE" -le 2 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  PHASE 2: 2B Fine-tuning Gaps                      ║"
    echo "╚══════════════════════════════════════════════════════╝"

    # Missing: clean_radar B (have D, E already)
    run_train "nvidia/Cosmos-Reason2-2B" "B" "cosmos2b_condB_clean_radar"

    # Missing: clean_lidar_only B
    run_train "nvidia/Cosmos-Reason2-2B" "B" "cosmos2b_condB_clean_lidar_only"

    # Missing: clean_lidar_only D
    run_train "nvidia/Cosmos-Reason2-2B" "D" "cosmos2b_condD_clean_lidar_only"

    # Missing: clean_radar_only B
    run_train "nvidia/Cosmos-Reason2-2B" "B" "cosmos2b_condB_clean_radar_only"

    # Missing: clean_radar_only D
    run_train "nvidia/Cosmos-Reason2-2B" "D" "cosmos2b_condD_clean_radar_only"

    echo "  ✅ Phase 2 complete — 2B fine-tuning gaps filled"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: 8B Fine-tuning (main goal — highest value)
# ══════════════════════════════════════════════════════════════════════════════
if [ "$START_PHASE" -le 3 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  PHASE 3: 8B Fine-tuning (Priority Order)          ║"
    echo "╚══════════════════════════════════════════════════════╝"

    # Priority 1: Single-sensor modalities (most deployment-relevant)
    run_train "nvidia/Cosmos-Reason2-8B" "B" "cosmos8b_condB_clean_lidar_only"
    run_train "nvidia/Cosmos-Reason2-8B" "D" "cosmos8b_condD_clean_lidar_only"

    run_train "nvidia/Cosmos-Reason2-8B" "B" "cosmos8b_condB_clean_radar_only"
    run_train "nvidia/Cosmos-Reason2-8B" "D" "cosmos8b_condD_clean_radar_only"

    # Priority 2: Combined sensor inputs
    run_train "nvidia/Cosmos-Reason2-8B" "B" "cosmos8b_condB_clean_lidar"
    run_train "nvidia/Cosmos-Reason2-8B" "D" "cosmos8b_condD_clean_lidar"

    run_train "nvidia/Cosmos-Reason2-8B" "B" "cosmos8b_condB_clean_radar"
    # cosmos8b_condD_clean_radar already done — skip

    echo "  ✅ Phase 3 complete — 8B core fine-tuning done"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: Optional — img/img+det/cam_only/bev_only fine-tuning
# ══════════════════════════════════════════════════════════════════════════════
if [ "$START_PHASE" -le 4 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  PHASE 4: Optional — img/det/cam/bev Fine-tuning   ║"
    echo "╚══════════════════════════════════════════════════════╝"

    # 8B img+det (camera + detections text — useful if camera-heavy deployment)
    run_train "nvidia/Cosmos-Reason2-8B" "B" "cosmos8b_condB_img_det"
    run_train "nvidia/Cosmos-Reason2-8B" "D" "cosmos8b_condD_img_det"

    # 8B img (BEV + camera, no text)
    run_train "nvidia/Cosmos-Reason2-8B" "B" "cosmos8b_condB_img"
    run_train "nvidia/Cosmos-Reason2-8B" "D" "cosmos8b_condD_img"

    echo "  ✅ Phase 4 complete — all optional fine-tuning done"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
echo " 🎉 All phases complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Checkpoints saved in: $SAVE_DIR/"
echo "Eval results in:      $EVAL_DIR/"
echo ""
echo "Next: run evaluation on fine-tuned models:"
echo "  python evaluate_zeroshot.py \\"
echo "    --model saves/<model_dir>/best_model \\"
echo "    --val_file <val_file> \\"
echo "    --input_level <level> --condition <B/D>"
