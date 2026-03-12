# MoRAL Training Progress Tracker

## Zero-Shot Evaluation Status

### Cosmos-Reason2-2B — ✅ COMPLETE
All conditions B/D across all modalities done.

### Cosmos-Reason2-8B — 3 GAPS remaining
| Modality | Cond B | Cond D |
|----------|--------|--------|
| bev_only | ✅ | ✅ |
| cam_only | ✅ | ❌ **missing** |
| img | ✅ | ✅ |
| img+det | ✅ | ✅ |
| clean_lidar | ✅ | ✅ |
| clean_lidar_only | ❌ **missing** | ✅ |
| clean_radar | ✅ | ✅ |
| clean_radar_only | ❌ **missing** | ✅ |

---

## Fine-Tuned Status

### 2B Fine-tuned — 5 GAPS remaining
| Modality | Cond B | Cond D |
|----------|--------|--------|
| clean_lidar | ✅ | ✅ |
| clean_radar | ❌ **missing** | ✅ (+ E) |
| clean_lidar_only | ❌ **missing** | ❌ **missing** |
| clean_radar_only | ❌ **missing** | ❌ **missing** |
| img / img+det / cam_only / bev_only | — optional — | — optional — |

### 8B Fine-tuned — Nearly everything missing
| Modality | Cond B | Cond D |
|----------|--------|--------|
| clean_radar | ❌ | ✅ (only one done) |
| clean_lidar | ❌ | ❌ |
| clean_lidar_only | ❌ | ❌ |
| clean_radar_only | ❌ | ❌ |
| img / img+det | ❌ optional | ❌ optional |

---

## Priority Execution Order

### Phase 1 — 8B Zero-shot Gaps ⏱️ ~1.5 hours
3 eval runs, ~30 min each. No training. Completes the baseline.

### Phase 2 — 2B Fine-tuning Gaps ⏱️ ~3-5 hours (H100)
5 training runs. Fills the 2B fine-tuning matrix.

### Phase 3 — 8B Fine-tuning (Main Goal) ⏱️ ~12-20 hours (H100)
7 training runs. This is the core thesis contribution.

### Phase 4 — Optional img/det fine-tuning ⏱️ ~8-12 hours (H100)
4 training runs. Only if camera-based deployment is a goal.

---

## How to Run

```bash
# All phases sequentially (let it run overnight on H100):
bash antigravity_suggestions/run_all_finetune.sh

# Skip to Phase 3 (8B training only):
bash antigravity_suggestions/run_all_finetune.sh --start-phase 3

# Dry run (test everything):
bash antigravity_suggestions/run_all_finetune.sh --dry-run
```

## Docker (recommended for cloud):
```bash
# Build once (pre-compiles FlashAttention — cached after first build)
docker build -t moral-train -f antigravity_suggestions/Dockerfile .

# Run all phases on H100
docker run --gpus all \
    -v $(pwd)/02_cosmos_integration/hf_data:/workspace/data \
    -v $(pwd)/saves:/workspace/saves \
    -e HF_TOKEN=$HF_TOKEN \
    moral-train \
    bash run_all_finetune.sh
```
