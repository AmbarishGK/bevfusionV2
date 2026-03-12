# MoRAL — Ablation Strategy & Thesis Evidence Matrix

## Your Core Claim
> Fine-tuning a VLM on reasoning chains generated from **LiDAR BEV + radar-enriched**
> sensor data improves physical/spatial reasoning compared to camera-only training.

To prove this, you need **controlled comparisons** where only ONE variable changes.

---

## Complete Status Matrix — Where You Stand

### 2B Zero-shot — ✅ COMPLETE (16/16)

| Input Modality | Cond B | Cond D |
|----------------|:---:|:---:|
| bev_only | ✅ | ✅ |
| cam_only | ✅ | ✅ |
| img | ✅ | ✅ |
| img+det | ✅ | ✅ |
| clean_lidar | ✅ | ✅ |
| clean_lidar_only | ✅ | ✅ |
| clean_radar | ✅ | ✅ |
| clean_radar_only | ✅ | ✅ |

**Total: 16/16 — nothing to do here.**

---

### 2B Fine-tuned — 5 MISSING (4/9 done)

| Input Modality | Cond B | Cond D | Notes |
|----------------|:---:|:---:|-------|
| clean_lidar | ✅ (aws) | ✅ | |
| clean_radar | ✅ (aws) | ✅ | Also have cond E |
| clean_lidar_only | ❌ **NEEDED** | ❌ **NEEDED** | Single-sensor robustness |
| clean_radar_only | ❌ **NEEDED** | ❌ **NEEDED** | Single-sensor robustness |
| clean_radar B | ❌ **NEEDED** | — | Have D/E, missing B |
| bev_only | — optional | — optional | Low priority |
| cam_only | — optional | — optional | Low priority |
| img | — optional | — optional | Low priority |
| img+det | — optional | — optional | Low priority |

**Remaining: 5 must-have (clean_radar B, clean_lidar_only B/D, clean_radar_only B/D)**
**Run on: 4090 (~6-8 hrs)**

---

### 8B Zero-shot — 3 MISSING + 4 BROKEN (9/16 working)

| Input Modality | Cond B | Cond D | Notes |
|----------------|:---:|:---:|-------|
| bev_only | ✅ | ⚠️ **BROKEN** (0% tags, BLEU=0.005) | Re-run D |
| cam_only | ❌ **MISSING** | ❌ **MISSING** | Never run |
| img | ✅ | ⚠️ **BROKEN** (0% tags, BLEU=0.005) | Re-run D |
| img+det | ✅ | ⚠️ **BROKEN** (0% tags, BLEU=0.005) | Re-run D |
| clean_lidar | ✅ | ⚠️ **BROKEN** (0% tags, BLEU=0.005) | Re-run D |
| clean_lidar_only | ❌ **MISSING** | ✅ | Never run B |
| clean_radar | ✅ | ✅ | Both working |
| clean_radar_only | ❌ **MISSING** | ✅ | Never run B |

**Remaining: 3 never-run (cam_only B+D, clean_lidar_only B, clean_radar_only B) + 4 re-runs (broken D conditions)**
**Run on: 4090 (~3-4 hrs)**

---

### 8B Fine-tuned — Almost everything MISSING (0-1/14 done)

| Input Modality | Cond B | Cond D | Notes |
|----------------|:---:|:---:|-------|
| clean_lidar | ❌ **NEEDED** | ❌ **NEEDED** | Priority 2 |
| clean_lidar_only | ❌ **NEEDED** | ❌ **NEEDED** | Priority 1 — single-sensor |
| clean_radar | ❌ **NEEDED** | ✅ (partial — 1 run exists) | Priority 2 |
| clean_radar_only | ❌ **NEEDED** | ❌ **NEEDED** | Priority 1 — single-sensor |
| bev_only | — optional | — optional | |
| cam_only | — optional | — optional | |
| img | — optional | — optional | |
| img+det | — optional | — optional | |

**Remaining: 7 must-have (clean_lidar B/D, clean_lidar_only B/D, clean_radar B, clean_radar_only B/D)**
**Run on: H100 (~12-20 hrs)**

---

## Grand Total Summary

| Category | Done | Remaining | Priority |
|----------|:---:|:---:|---------|
| 2B Zero-shot | 16/16 | 0 | ✅ Complete |
| 2B Fine-tuned | 4/9 | **5** | 🔶 Run on 4090 |
| 8B Zero-shot | 9/16 | **7** (3 missing + 4 broken) | 🔶 Run on 4090 |
| 8B Fine-tuned | ~1/8 | **7** | 🔴 Run on H100 |
| **TOTAL** | **30/49** | **19** | |

---

## The 5 Critical Ablations

### Ablation 1: Does BEV Help? (Camera-Only vs BEV+Camera)

| Run | Model | Input | What It Tests |
|-----|-------|-------|---------------|
| `cam_only` zero-shot | 2B/8B | CAM_FRONT only | Baseline — no spatial augmentation |
| `img` zero-shot | 2B/8B | BEV + CAM_FRONT | Does BEV image add spatial reasoning? |
| `img+det` zero-shot | 2B/8B | BEV + CAM + detections text | Does structured text help beyond image? |

**Expected thesis result:** `img` > `cam_only` on spatial/distance/TTC questions.

| Data Point | 2B | 8B | Status |
|------------|:---:|:---:|--------|
| cam_only B ZS | ✅ | ❌ MISSING | Need 8B |
| cam_only D ZS | ✅ | ❌ MISSING | Need 8B |
| img B ZS | ✅ | ✅ | Ready |
| img D ZS | ✅ | ⚠️ BROKEN | Re-run |
| img+det B ZS | ✅ | ✅ | Ready |
| img+det D ZS | ✅ | ⚠️ BROKEN | Re-run |

---

### Ablation 2: Does Radar Help? (LiDAR-only vs LiDAR+Radar)

| Run | Model | Input | What It Tests |
|-----|-------|-------|---------------|
| `clean_lidar` | 2B/8B | LiDAR BEV + CAM | LiDAR spatial reasoning |
| `clean_radar` | 2B/8B | LiDAR+Radar BEV + CAM | Does radar Doppler velocity improve reasoning? |
| `clean_lidar_only` | 2B/8B | LiDAR BEV only | Pure spatial, no camera |
| `clean_radar_only` | 2B/8B | LiDAR+Radar BEV only | Pure spatial+velocity, no camera |

**Expected thesis result:** `clean_radar` > `clean_lidar` on velocity/TTC/safety questions.

| Data Point | 2B ZS | 2B FT | 8B ZS | 8B FT | Status |
|------------|:---:|:---:|:---:|:---:|--------|
| clean_lidar B | ✅ | ✅ | ✅ | ❌ | Need 8B FT |
| clean_lidar D | ✅ | ✅ | ⚠️ BROKEN | ❌ | Re-run 8B ZS, need 8B FT |
| clean_radar B | ✅ | ✅ | ✅ | ❌ | Need 8B FT |
| clean_radar D | ✅ | ✅ | ✅ | ✅ partial | Verify 8B FT |
| clean_lidar_only B | ✅ | ❌ | ❌ MISSING | ❌ | Need 2B FT, 8B ZS+FT |
| clean_lidar_only D | ✅ | ❌ | ✅ | ❌ | Need 2B FT, 8B FT |
| clean_radar_only B | ✅ | ❌ | ❌ MISSING | ❌ | Need 2B FT, 8B ZS+FT |
| clean_radar_only D | ✅ | ❌ | ✅ | ❌ | Need 2B FT, 8B FT |

---

### Ablation 3: Does Fine-tuning Help? (Zero-shot vs Fine-tuned)

**Expected thesis result:** Fine-tuned >> Zero-shot on all structured metrics.

| Data Point | ZS | FT | Status |
|------------|:---:|:---:|--------|
| 2B clean_lidar B | ✅ | ✅ | **✅ Can compare** |
| 2B clean_lidar D | ✅ | ✅ | **✅ Can compare** |
| 2B clean_radar B | ✅ | ✅ | **✅ Can compare** |
| 2B clean_radar D | ✅ | ✅ | **✅ Can compare** |
| 2B clean_lidar_only B | ✅ | ❌ | Need FT |
| 2B clean_lidar_only D | ✅ | ❌ | Need FT |
| 2B clean_radar_only B | ✅ | ❌ | Need FT |
| 2B clean_radar_only D | ✅ | ❌ | Need FT |
| 8B clean_lidar B | ✅ | ❌ | Need FT |
| 8B clean_lidar D | ⚠️ BROKEN | ❌ | Re-run ZS, need FT |
| 8B clean_radar B | ✅ | ❌ | Need FT |
| 8B clean_radar D | ✅ | ✅ partial | **✅ Can compare** |
| 8B clean_lidar_only B | ❌ | ❌ | Need both |
| 8B clean_lidar_only D | ✅ | ❌ | Need FT |
| 8B clean_radar_only B | ❌ | ❌ | Need both |
| 8B clean_radar_only D | ✅ | ❌ | Need FT |

**Already comparable pairs: 5 (4 × 2B + 1 × 8B)**
**Need to complete: 11 more pairs**

---

### Ablation 4: Does Model Scale Help? (2B vs 8B)

**Expected thesis result:** 8B > 2B, but gap narrows with fine-tuning.

| Data Point | 2B ZS | 8B ZS | 2B FT | 8B FT | Comparison Ready? |
|------------|:---:|:---:|:---:|:---:|:---:|
| clean_lidar B | ✅ | ✅ | ✅ | ❌ | ZS only |
| clean_lidar D | ✅ | ⚠️ | ✅ | ❌ | Need 8B ZS fix |
| clean_radar B | ✅ | ✅ | ✅ | ❌ | ZS only |
| clean_radar D | ✅ | ✅ | ✅ | ✅ | **✅ Full comparison** |
| clean_lidar_only B | ✅ | ❌ | ❌ | ❌ | Need everything |
| clean_lidar_only D | ✅ | ✅ | ❌ | ❌ | ZS only |
| clean_radar_only B | ✅ | ❌ | ❌ | ❌ | Need everything |
| clean_radar_only D | ✅ | ✅ | ❌ | ❌ | ZS only |

---

### Ablation 5: Does BEV Quality Matter? (GT Boxes vs Clean BEV)

| Data Point | 2B ZS | 8B ZS | Status |
|------------|:---:|:---:|--------|
| img (GT BEV) B | ✅ | ✅ | Ready |
| img (GT BEV) D | ✅ | ⚠️ BROKEN | Re-run |
| clean_lidar (clean BEV) B | ✅ | ✅ | Ready |
| clean_lidar (clean BEV) D | ✅ | ⚠️ BROKEN | Re-run |

**Can compare B conditions now. Need D re-runs for full picture.**

---

## Minimum Viable Table for Thesis

This is the **single table** that proves your claim. Everything else is supporting evidence.

| Input Modality | 2B Zero-shot | 2B Fine-tuned | 8B Zero-shot | 8B Fine-tuned |
|----------------|:---:|:---:|:---:|:---:|
| cam_only | ✅ | optional | ❌ need 2 runs | optional |
| clean_lidar | ✅ | ✅ | ⚠️ need 1 re-run | ❌ **NEEDED** (2 runs) |
| clean_radar | ✅ | ✅ | ✅ | ❌ **NEEDED** (1 run) |
| clean_lidar_only | ✅ | ❌ **NEEDED** (2) | ❌ need 1 run | ❌ **NEEDED** (2 runs) |
| clean_radar_only | ✅ | ❌ **NEEDED** (2) | ❌ need 1 run | ❌ **NEEDED** (2 runs) |

---

## What to Run on Your 4090 (24GB)

Your 4090 can handle ALL 2B work and ALL evaluations:

```bash
# ── 2B fine-tuning gaps (5 runs, ~6-8 hrs) ──
python train_cosmos2b.py --train_file hf_data/clean_conditionB_train.jsonl \
    --val_file hf_data/clean_conditionB_val.jsonl \
    --output_dir saves/cosmos2b_condB_clean_radar --epochs 3

python train_cosmos2b.py --train_file hf_data/clean_conditionB_train.jsonl \
    --val_file hf_data/clean_conditionB_val.jsonl \
    --output_dir saves/cosmos2b_condB_clean_lidar_only --epochs 3

python train_cosmos2b.py --train_file hf_data/clean_conditionD_train.jsonl \
    --val_file hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos2b_condD_clean_lidar_only --epochs 3

python train_cosmos2b.py --train_file hf_data/clean_conditionB_train.jsonl \
    --val_file hf_data/clean_conditionB_val.jsonl \
    --output_dir saves/cosmos2b_condB_clean_radar_only --epochs 3

python train_cosmos2b.py --train_file hf_data/clean_conditionD_train.jsonl \
    --val_file hf_data/clean_conditionD_val.jsonl \
    --output_dir saves/cosmos2b_condD_clean_radar_only --epochs 3

# ── 8B zero-shot gaps + re-runs (7 runs, ~3-4 hrs) ──
# see RUN_PLAN.md Phase 2 and 3 for exact commands
```

---

## Metrics Per Question Type (From Your Data)

| qtype | Key Metric | GT Field | What LLM Judge Should Evaluate |
|-------|-----------|----------|-------------------------------|
| `spatial` | distance accuracy | `distance_m` | Is predicted distance within 20% of GT? |
| `safety` | TTC accuracy + action | `ttc_s` | Is TTC within 30%? Correct threat identified? |
| `velocity` | speed accuracy | `velocity_ms` | Is speed within 30%? Correct moving objects identified? |
| `occlusion` | qualitative | — | Does answer identify correct occluded objects? |
| `gap` | qualitative | — | Correct safe/unsafe judgment? |
| `physics` | distance + stopping | `nearest_ahead_m` | Correct physics reasoning? |
| `zone` | qualitative | — | Correct zone identification? |
| `planning` | action match | `min_ttc_s` | GT action match + correct reasoning? |
| `counterfactual` | time prediction | `t_impact_s` | Correct projected trajectory? |
| `trajectory` | object count | `objects_entering_path` | Correct count of objects entering path? |
| `near_miss` | count | `near_miss_count` | Correct near-miss identification? |
| `multi_conflict` | risk score | `top_risk_score` | Correct risk ranking? |
| `sensor_limit` | count | `gt_estimated_count` | Correctly identifies sensor limits? |
| `ethical` | TTC + reasoning | `min_dilemma_ttc_s` | Reasonable ethical reasoning? |
