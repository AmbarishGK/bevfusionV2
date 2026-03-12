# MoRAL — Exact Step-by-Step Run Plan

## Overview: What Remains

| Phase | What | Where | Est. Time |
|-------|------|-------|-----------|
| 1 | MoRAL-Score on existing results | 4090 (local) | 10 min |
| 2 | Fix broken 8B zero-shot D runs | 4090 | 2-3 hrs |
| 3 | Missing 8B zero-shot gaps | 4090 | 1.5 hrs |
| 4 | Missing 2B fine-tuning | 4090 | 6-8 hrs |
| 5 | 8B fine-tuning (main goal) | H100 (cloud) | 12-20 hrs |
| 6 | Evaluate all fine-tuned models | 4090 | 4-6 hrs |
| 7 | LLM Judge scoring | 4090 | 1-2 hrs |
| 8 | HIL evaluation | You (manual) | 2-3 hrs |

---

## Phase 1: Run MoRAL-Score on ALL Existing Results (10 min, local)

```bash
# cd to moral_pipeline directory
cd ~/workspaces/bevfusionV2/moral_pipeline

# Score ALL existing results
python antigravity_suggestions/moral_score.py \
    --results_dir \
        ~/workspaces/data/moral_all_results/saves/zeroshot_results \
        ~/workspaces/data/moral_all_results/saves/zeroshot_results_8B \
        ~/workspaces/data/moral_all_results/saves/finetuned_results \
        ~/workspaces/data/moral_all_results/saves/finetuned_results_aws \
    --output_dir ~/workspaces/data/moral_all_results/saves/moral_score_results \
    --compare

# Generate HIL evaluation sheet
python antigravity_suggestions/moral_score.py \
    --results_dir \
        ~/workspaces/data/moral_all_results/saves/zeroshot_results \
        ~/workspaces/data/moral_all_results/saves/finetuned_results \
    --output_dir ~/workspaces/data/moral_all_results/saves/moral_score_results \
    --hil --hil_per_qtype 2
```

**Output:** `moral_score_comparison.md` (thesis table) + `hil/` folder with evaluation sheets.

---

## Phase 2: Fix Broken 8B Zero-shot Condition-D Runs (4090, ~2-3 hrs)

These 4 runs showed 0% quality tags — almost certainly broken image paths.
**Before running, check the data paths are correct.**

```bash
# First, verify image paths in the val file
python3 -c "
import json
with open('02_cosmos_integration/hf_data/local_conditionD_val.jsonl') as f:
    rec = json.loads(f.readline())
for part in rec['messages'][0]['content']:
    if part.get('type') == 'image':
        import os
        exists = os.path.exists(part['image'])
        print(f\"  {'✅' if exists else '❌'} {part['image']}\")
"

# If paths are wrong, you may need to update them.
# Then re-run the 4 broken evals:

# 2a. img D (was 0% tags)
python evaluate_zeroshot.py \
    --model nvidia/Cosmos-Reason2-8B \
    --val_file 02_cosmos_integration/hf_data/local_conditionD_val.jsonl \
    --input_level img --condition D \
    --max_samples 200 \
    --output_dir ~/workspaces/data/moral_all_results/saves/zeroshot_results_8B

# 2b. img+det D (was 0% tags)
python evaluate_zeroshot.py \
    --model nvidia/Cosmos-Reason2-8B \
    --val_file 02_cosmos_integration/hf_data/local_conditionD_val.jsonl \
    --input_level img+det --condition D \
    --max_samples 200 \
    --output_dir ~/workspaces/data/moral_all_results/saves/zeroshot_results_8B

# 2c. bev_only D (was 0% tags)
python evaluate_zeroshot.py \
    --model nvidia/Cosmos-Reason2-8B \
    --val_file 02_cosmos_integration/hf_data/local_conditionD_val.jsonl \
    --input_level bev_only --condition D \
    --max_samples 200 \
    --output_dir ~/workspaces/data/moral_all_results/saves/zeroshot_results_8B

# 2d. clean_lidar D (was 0% tags)
python evaluate_zeroshot.py \
    --model nvidia/Cosmos-Reason2-8B \
    --val_file 02_cosmos_integration/hf_data/local_conditionD_val.jsonl \
    --input_level clean_lidar --condition D \
    --max_samples 200 \
    --output_dir ~/workspaces/data/moral_all_results/saves/zeroshot_results_8B
```

---

## Phase 3: Missing 8B Zero-shot Gaps (4090, ~1.5 hrs)

3 runs that were never done:

```bash
# 3a. cam_only D (never run)
python evaluate_zeroshot.py \
    --model nvidia/Cosmos-Reason2-8B \
    --val_file 02_cosmos_integration/hf_data/local_conditionD_val.jsonl \
    --input_level cam_only --condition D \
    --max_samples 200 \
    --output_dir ~/workspaces/data/moral_all_results/saves/zeroshot_results_8B

# 3b. clean_lidar_only B (never run)
python evaluate_zeroshot.py \
    --model nvidia/Cosmos-Reason2-8B \
    --val_file 02_cosmos_integration/hf_data/local_conditionB_val.jsonl \
    --input_level clean_lidar_only --condition B \
    --max_samples 200 \
    --output_dir ~/workspaces/data/moral_all_results/saves/zeroshot_results_8B

# 3c. clean_radar_only B (never run)
python evaluate_zeroshot.py \
    --model nvidia/Cosmos-Reason2-8B \
    --val_file 02_cosmos_integration/hf_data/local_conditionB_val.jsonl \
    --input_level clean_radar_only --condition B \
    --max_samples 200 \
    --output_dir ~/workspaces/data/moral_all_results/saves/zeroshot_results_8B
```

---

## Phase 4: Missing 2B Fine-tuning (4090, ~6-8 hrs total)

Run these sequentially on your 4090:

```bash
# 4a. clean_radar B (have D/E, missing B)
python train_cosmos2b.py \
    --train_file 02_cosmos_integration/hf_data/clean_conditionB_train.jsonl \
    --val_file 02_cosmos_integration/hf_data/clean_conditionB_val.jsonl \
    --output_dir saves/cosmos2b_condB_clean_radar \
    --epochs 3

# 4b. clean_lidar_only B
python train_cosmos2b.py \
    --train_file 02_cosmos_integration/hf_data/clean_conditionB_train.jsonl \
    --val_file 02_cosmos_integration/hf_data/clean_conditionB_val.jsonl \
    --output_dir saves/cosmos2b_condB_clean_lidar_only \
    --epochs 3

# 4c. clean_lidar_only D
python train_cosmos2b.py \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file 02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos2b_condD_clean_lidar_only \
    --epochs 3

# 4d. clean_radar_only B
python train_cosmos2b.py \
    --train_file 02_cosmos_integration/hf_data/clean_conditionB_train.jsonl \
    --val_file 02_cosmos_integration/hf_data/clean_conditionB_val.jsonl \
    --output_dir saves/cosmos2b_condB_clean_radar_only \
    --epochs 3

# 4e. clean_radar_only D
python train_cosmos2b.py \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file 02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos2b_condD_clean_radar_only \
    --epochs 3
```

---

## Phase 5: 8B Fine-tuning (H100 cloud, ~12-20 hrs)

**Use Docker or setup script from CLOUD_SETUP_GUIDE.md.**

```bash
# On H100 — run in screen/tmux:
screen -S train

# 5a. clean_lidar_only B (highest priority — single-sensor robustness)
python antigravity_suggestions/train_cosmos_unified.py \
    --model nvidia/Cosmos-Reason2-8B --profile h100 \
    --train_file data/clean_conditionB_train.jsonl \
    --val_file data/clean_conditionB_val.jsonl \
    --output_dir saves/cosmos8b_condB_clean_lidar_only \
    --epochs 3 --save_steps 100

# 5b. clean_lidar_only D
python antigravity_suggestions/train_cosmos_unified.py \
    --model nvidia/Cosmos-Reason2-8B --profile h100 \
    --train_file data/clean_conditionD_train.jsonl \
    --val_file data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos8b_condD_clean_lidar_only \
    --epochs 3 --save_steps 100

# 5c. clean_radar_only B
python antigravity_suggestions/train_cosmos_unified.py \
    --model nvidia/Cosmos-Reason2-8B --profile h100 \
    --train_file data/clean_conditionB_train.jsonl \
    --val_file data/clean_conditionB_val.jsonl \
    --output_dir saves/cosmos8b_condB_clean_radar_only \
    --epochs 3 --save_steps 100

# 5d. clean_radar_only D
python antigravity_suggestions/train_cosmos_unified.py \
    --model nvidia/Cosmos-Reason2-8B --profile h100 \
    --train_file data/clean_conditionD_train.jsonl \
    --val_file data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos8b_condD_clean_radar_only \
    --epochs 3 --save_steps 100

# 5e. clean_lidar B
python antigravity_suggestions/train_cosmos_unified.py \
    --model nvidia/Cosmos-Reason2-8B --profile h100 \
    --train_file data/clean_conditionB_train.jsonl \
    --val_file data/clean_conditionB_val.jsonl \
    --output_dir saves/cosmos8b_condB_clean_lidar \
    --epochs 3 --save_steps 100

# 5f. clean_lidar D
python antigravity_suggestions/train_cosmos_unified.py \
    --model nvidia/Cosmos-Reason2-8B --profile h100 \
    --train_file data/clean_conditionD_train.jsonl \
    --val_file data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos8b_condD_clean_lidar \
    --epochs 3 --save_steps 100

# 5g. clean_radar B (have D already)
python antigravity_suggestions/train_cosmos_unified.py \
    --model nvidia/Cosmos-Reason2-8B --profile h100 \
    --train_file data/clean_conditionB_train.jsonl \
    --val_file data/clean_conditionB_val.jsonl \
    --output_dir saves/cosmos8b_condB_clean_radar \
    --epochs 3 --save_steps 100
```

---

## Phase 6: Evaluate ALL Fine-tuned Models (4090, ~4-6 hrs)

After downloading 8B checkpoints from cloud, run eval for each:

```bash
# For EACH fine-tuned model, run eval with matching input_level and condition.
# Template:
# python evaluate_zeroshot.py \
#     --model saves/<MODEL_DIR>/best_model \
#     --val_file 02_cosmos_integration/hf_data/local_condition<COND>_val.jsonl \
#     --input_level <LEVEL> --condition <COND> \
#     --max_samples 200 \
#     --output_dir ~/workspaces/data/moral_all_results/saves/finetuned_results

# 6a. 2B clean_radar B
python evaluate_zeroshot.py \
    --model saves/cosmos2b_condB_clean_radar/best_model \
    --val_file 02_cosmos_integration/hf_data/local_conditionB_val.jsonl \
    --input_level clean_radar --condition B \
    --output_dir ~/workspaces/data/moral_all_results/saves/finetuned_results

# 6b-6e. Repeat for each 2B fine-tuned model (clean_lidar_only B/D, clean_radar_only B/D)

# 6f-6l. Repeat for each 8B fine-tuned model
# Example:
python evaluate_zeroshot.py \
    --model saves/cosmos8b_condB_clean_lidar_only/best_model \
    --val_file 02_cosmos_integration/hf_data/local_conditionB_val.jsonl \
    --input_level clean_lidar_only --condition B \
    --output_dir ~/workspaces/data/moral_all_results/saves/finetuned_results_8B
```

---

## Phase 7: LLM Judge + MoRAL-Score on Everything (4090, 1-2 hrs)

```bash
# 7a. Re-run MoRAL-Score on ALL results (including new ones)
python antigravity_suggestions/moral_score.py \
    --results_dir \
        ~/workspaces/data/moral_all_results/saves/zeroshot_results \
        ~/workspaces/data/moral_all_results/saves/zeroshot_results_8B \
        ~/workspaces/data/moral_all_results/saves/finetuned_results \
        ~/workspaces/data/moral_all_results/saves/finetuned_results_aws \
        ~/workspaces/data/moral_all_results/saves/finetuned_results_8B \
    --output_dir ~/workspaces/data/moral_all_results/saves/moral_score_final \
    --compare --hil --hil_per_qtype 2

# 7b. LLM Judge on a subset (50 samples per run — costs ~$5-10 in API)
export ANTHROPIC_API_KEY="sk-ant-..."
python antigravity_suggestions/llm_judge.py \
    --results_dir \
        ~/workspaces/data/moral_all_results/saves/zeroshot_results \
        ~/workspaces/data/moral_all_results/saves/finetuned_results \
    --provider anthropic --model claude-sonnet-4-20250514 \
    --max_samples 50 --compare \
    --output_dir ~/workspaces/data/moral_all_results/saves/llm_judge_results
```

---

## Phase 8: HIL Evaluation (Manual, 2-3 hrs)

```bash
# Open the generated evaluation sheet:
open ~/workspaces/data/moral_all_results/saves/moral_score_final/hil/hil_evaluation_sheet.csv
# Or use the text version:
cat ~/workspaces/data/moral_all_results/saves/moral_score_final/hil/hil_full_evaluation.txt
```

1. Open `hil_evaluation_sheet.csv` in Google Sheets
2. Rate each sample on 6 dimensions (1-5)
3. Mark any novel detections (objects model found that GT missed)
4. Save completed sheet
5. Import back for analysis

---

## Parallel Execution Plan (Minimize Wall Time)

```
DAY 1 (4090):
  Morning:  Phase 1 (MoRAL-Score, 10 min) → start Phase 2 (fix D evals)
  Afternoon: Phase 3 (missing zero-shots) → start Phase 4a-4b (2B fine-tuning)
  Evening:  Phase 4c-4e (2B fine-tuning continues overnight)

DAY 1 (H100 in parallel):
  Set up cloud → start Phase 5a (8B training, runs overnight)

DAY 2 (4090):
  Morning:  Phase 6 (eval 2B fine-tuned models)
  Afternoon: Download 8B checkpoints, eval those too

DAY 2 (H100):
  Phase 5b-5g continues (training runs ~3 hrs each)

DAY 3:
  Phase 7 (LLM Judge) → Phase 8 (HIL) → thesis tables ready
```

**Total wall time: ~3 days with parallel 4090 + H100.**
**Total cloud cost: ~$20-30.**
