# MoRAL Pipeline — Training & Evaluation Scripts

**MoRAL: Multi-modal Reasoning for Autonomous driving with LiDAR**

This directory contains the training, evaluation, and utility scripts for fine-tuning
NVIDIA Cosmos-Reason2 models on sensor-enriched autonomous driving QA data from nuScenes.

---

## Files Overview

### `train_cosmos2b.py` — Fine-tune Cosmos-Reason2-2B (QLoRA)
Fine-tunes `nvidia/Cosmos-Reason2-2B` using 4-bit QLoRA with PEFT/TRL on a single GPU (designed for RTX 4090 24GB).

**What it does:**
1. Pre-flight checks: CUDA, packages, data files, image paths, RAM, training config
2. Loads model in 4-bit (NF4 + double quantization) with SDPA attention
3. Applies LoRA adapters to all projection layers (q/k/v/o/gate/up/down)
4. Custom `MoRALDataset` loads JSONL records, validates images, filters error responses
5. Custom `collate_fn` builds Qwen3VL chat format with masked prompt labels (-100)
6. Trains with AdamW + cosine LR schedule + gradient accumulation
7. Saves checkpoints every 200 steps + best model by validation loss
8. Post-training inference sanity check

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_file` | `02_cosmos_integration/hf_data/clean_conditionD_train.jsonl` | JSONL train data |
| `--val_file` | `02_cosmos_integration/hf_data/clean_conditionD_val.jsonl` | JSONL val data |
| `--output_dir` | `saves/cosmos2b_condD_finetuned` | Checkpoint output dir |
| `--model` | `nvidia/Cosmos-Reason2-2B` | HuggingFace model ID or path |
| `--epochs` | `3` | Training epochs |
| `--batch_size` | `1` | Per-device batch size |
| `--grad_accum` | `16` | Gradient accumulation steps (effective batch = 16) |
| `--lr` | `2e-4` | Learning rate |
| `--lora_rank` | `16` | LoRA rank |
| `--max_length` | `4096` | Max token sequence length |
| `--max_pixels` | `262144` (512×512) | Cap image resolution in pixels |
| `--wandb_key` | `None` | W&B API key (disabled if None) |
| `--dry_run` | `False` | Run 20 samples / 2 steps to test pipeline |
| `--resume` | `False` | Resume from latest checkpoint |

**Example:**
```bash
python train_cosmos2b.py \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos2b_condD_finetuned \
    --dry_run
```

---

### `train_cosmos8b.py` — Fine-tune Cosmos-Reason2-8B (QLoRA, H100-optimized)
Same architecture as the 2B trainer but with additional performance flags for high-end GPUs (A100/H100).

**Extra features over 2B trainer:**
- `--resume_from <path>` to resume from a specific checkpoint (not just latest)
- Full trainer state persistence (`trainer_state.pt`): optimizer, scheduler, RNG states, global step, epoch
- Checkpoint rotation (`--save_total_limit`) to cap disk usage
- FlashAttention-2 auto-detection (falls back to SDPA)
- TF32 matmul toggle (`--tf32`) for A100/H100 speedup
- `torch.compile` support (`--torch_compile`)
- Gradient checkpointing toggle (`--grad_ckpt`)
- Fused AdamW optimizer (PyTorch 2.x)
- Configurable DataLoader workers and prefetch
- FP16/BF16 precision selection with proper gradient scaler
- Tokens/second throughput logging

**Additional parameters (beyond 2B):**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--resume_from` | `None` | Specific checkpoint dir path |
| `--precision` | `bf16` | `bf16` or `fp16` autocast dtype |
| `--attn_impl` | `auto` | `auto` / `sdpa` / `flash_attention_2` |
| `--tf32` | `False` | Enable TF32 matmul (recommended on A100/H100) |
| `--torch_compile` | `False` | Use `torch.compile` (PyTorch 2.x) |
| `--grad_ckpt` | `False` | Enable gradient checkpointing (saves VRAM) |
| `--num_workers` | `8` | DataLoader workers |
| `--prefetch_factor` | `4` | DataLoader prefetch factor |
| `--save_steps` | `200` | Save checkpoint every N steps |
| `--save_total_limit` | `5` | Keep last N checkpoints (0=keep all) |
| `--log_steps` | `10` | Log metrics every N steps |

**Example (H100 optimized):**
```bash
python train_cosmos8b.py \
    --model nvidia/Cosmos-Reason2-8B \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos8b_condD_finetuned \
    --tf32 --torch_compile --grad_ckpt \
    --batch_size 2 --grad_accum 8 \
    --num_workers 8 --prefetch_factor 4 \
    --save_steps 100 --save_total_limit 5 \
    --epochs 3
```

**Resume training:**
```bash
python train_cosmos8b.py \
    --resume \
    --output_dir saves/cosmos8b_condD_finetuned \
    ... (same args as original run)
```

---

### `evaluate_zeroshot.py` — Zero-shot Ablation Evaluator
Evaluates any VLM (Cosmos-Reason2, Qwen, etc.) on MoRAL data **without fine-tuning**.
Tests different input modalities (camera-only, BEV+camera, BEV+camera+detections text).

**What it does:**
1. Loads the model via Unsloth (preferred) or HuggingFace Transformers (fallback)
2. For each validation record, builds messages with the requested input level
3. Runs inference (greedy for non-thinking models, sampled for thinking/reasoning models)
4. Extracts metrics: action accuracy (exact + family match), TTC error, BLEU, quality tags
5. Computes per-category accuracy using Cosmos taxonomy (Spatial & Temporal, Actions & Motion, etc.)
6. Writes per-sample results `.jsonl` + summary `.json`

**Parameters:**
| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--model` | ✅ | — | HuggingFace model ID or local path |
| `--val_file` | ✅ | — | JSONL val file path |
| `--input_level` | ✅ | — | See input levels below |
| `--condition` | | `B` | Sensor condition: `B`=LiDAR, `D`=LiDAR+Radar |
| `--det_root` | | `None` | Root dir for detections.json (auto-detected) |
| `--bev_root` | | `None` | Override BEV image root dir |
| `--max_samples` | | `200` | Max samples to evaluate |
| `--max_new_tokens` | | `1500` | Max generation tokens |
| `--output_dir` | | `saves/zeroshot_results` | Results output dir |
| `--fp16` | | `False` | Load in fp16 instead of 4-bit |

**Input levels:**
| Level | Description |
|-------|-------------|
| `cam_only` | CAM_FRONT only (no BEV) — hardest baseline |
| `img` | GT-box BEV + CAM_FRONT (standard training format) |
| `img+det` | GT-box BEV + CAM_FRONT + detections.json text in prompt |
| `bev_only` | GT-box BEV only (no camera) |
| `clean_lidar` | Lidar-only clean BEV + CAM_FRONT |
| `clean_radar` | Lidar+radar clean BEV + CAM_FRONT |
| `clean_lidar_only` | Lidar-only clean BEV (no camera) |
| `clean_radar_only` | Lidar+radar clean BEV (no camera) |

**Example:**
```bash
python evaluate_zeroshot.py \
    --model nvidia/Cosmos-Reason2-8B \
    --val_file 02_cosmos_integration/hf_data/local_conditionB_val.jsonl \
    --input_level img \
    --condition B \
    --max_samples 200
```

---

### `scene_utils.py` — Shared Utility Library (984 lines)
Core utility module used by all generation and processing scripts.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `get_ego_pose()` | Get ego vehicle pose from LIDAR_TOP sample data |
| `get_ego_speed()` | Compute ego speed via frame differencing |
| `get_lidar_points()` | Load LiDAR point cloud, transform sensor→ego frame |
| `_load_radar_pcd()` | Parse nuScenes mixed-type binary PCD files (struct-based) |
| `get_radar_points_ego()` | Load all 5 radar sensors, transform to ego frame, clip to 55m |
| `enrich_detections_with_radar()` | Match detections↔radar points, classify quality |
| `get_detections_from_gt()` | Extract GT detections from nuScenes annotations |
| `make_scene_text()` | Generate rich 8-section scene description for VLM QA |
| `save_bev_map()` | Render publication-quality BEV map image |
| `copy_camera_images()` | Copy camera images from nuScenes to output dir |
| `save_metadata()` | Save scene metadata JSON |

**Radar quality classification:**
| Quality | Meaning | Velocity source |
|---------|---------|-----------------|
| `reliable` | Vehicle, radar speed agrees with GT | radar-confirmed |
| `radial_ambiguous` | Pedestrian/cyclist — crosswise motion invisible | gt-estimated |
| `range_mismatch` | GT says fast but radar says stationary | gt-estimated |
| `unconfirmed` | No radar point within match radius | gt-estimated |

---

## Quick Start with `uv`

### 1. Create environment
```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
cd moral_pipeline
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Dry run (test pipeline end-to-end)
```bash
# Test 2B training pipeline
python train_cosmos2b.py \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --dry_run

# Test 8B training pipeline
python train_cosmos8b.py \
    --model nvidia/Cosmos-Reason2-8B \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --dry_run
```

### 3. Full training (H100)
```bash
python train_cosmos8b.py \
    --model nvidia/Cosmos-Reason2-8B \
    --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
    --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos8b_condD_finetuned \
    --tf32 --torch_compile --grad_ckpt \
    --batch_size 2 --grad_accum 8 \
    --epochs 3 --save_steps 100
```

### 4. Evaluate
```bash
python evaluate_zeroshot.py \
    --model saves/cosmos8b_condD_finetuned \
    --val_file 02_cosmos_integration/hf_data/local_conditionB_val.jsonl \
    --input_level img --condition B
```

---

## Data Format

Training data is JSONL where each line contains:
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/path/to/bev_map.png"},
        {"type": "image", "image": "/path/to/CAM_FRONT.jpg"},
        {"type": "text", "text": "Question about the scene..."}
      ]
    },
    {
      "role": "assistant",
      "content": "<think>[BEV]...[CAM]...[GT]...[DECISION]...</think>\n<answer>...</answer>"
    }
  ],
  "_meta": {
    "scene": "scene-0061",
    "question_type": "spatial",
    "gt_action": "BRAKE",
    "gt_value": 16.84,
    "gt_field": "distance_m"
  }
}
```

---

## Project Context

This pipeline is part of a master's thesis investigating whether fine-tuning a
camera-based driving VLM on reasoning chains generated from LiDAR BEV and
radar-enriched sensor data improves physical reasoning vs camera-only training.

**Pipeline flow:**
```
nuScenes trainval (850 scenes)
    → MoRAL pipeline (conditions A/B/D)
    → Cosmos-Reason2-8B (teacher — generates reasoning chains)
    → ShareGPT JSONL training data
    → Fine-tune Cosmos-Reason2-2B/8B (student — LoRA)
    → Evaluate on ablation benchmarks
```

See `MoRAL_Progress_*.md` files and `README_HF.md` for full research context.
