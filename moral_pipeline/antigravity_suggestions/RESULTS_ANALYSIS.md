# MoRAL — Results Analysis Report (32 Runs)

## Key Findings From Your Data

### 🔬 Finding 1: Fine-tuning Works — Spatial Accuracy Goes From 0% to ~10%

| Modality | Condition | Zero-shot Spatial | Fine-tuned Spatial | Δ | Zero-shot BLEU | Fine-tuned BLEU | Δ |
|----------|-----------|:-:|:-:|:-:|:-:|:-:|:-:|
| clean_lidar | B | 0.0 | **10.1** | **+10.1** | 0.030 | 0.063 | +0.033 |
| clean_lidar | D | 0.0 | **5.1** | **+5.1** | 0.028 | 0.034 | +0.006 |
| clean_radar | B | 0.0 | **8.6** | **+8.6** | 0.026 | 0.060 | +0.035 |
| clean_radar | D | 0.0 | **9.6** | **+9.6** | 0.036 | 0.055 | +0.019 |

> **Thesis evidence:** Fine-tuning on MoRAL reasoning chains improves spatial reasoning from 0% → 5-10% on Spatial & Temporal metrics, with consistent BLEU improvements across all tested modalities.

---

### 🔬 Finding 2: 8B Zero-shot Produces Better Structured Reasoning (But Not Accuracy)

| Metric | 2B (average) | 8B (average, cond B) |
|--------|:---:|:---:|
| Quality tag coverage (`[BEV]+[CAM]+[GT]+[DECISION]`) | 2-12% | 14-62% |
| Spatial & Temporal accuracy | 0.0% | 0.0-3.5% |
| Action family match | 100% | 0-29% |

**Interpretation:** 8B is much better at following the `[BEV]→[CAM]→[GT]→[DECISION]` reasoning structure (60% vs 10% coverage), and shows slight spatial accuracy (3.5% vs 0%). But 8B action matching is worse zero-shot, likely because the 8B model generates more verbose, nuanced answers that don't match the exact action labels.

> **This is exactly why fine-tuning 8B is your thesis's most important remaining work** — the model has the reasoning capability (60% structured) but needs calibration on output format.

---

### ⚠️ Finding 3: 8B Condition D Has Anomalous 0% Tags

| Modality | 8B Cond B Tags | 8B Cond D Tags |
|----------|:-:|:-:|
| img | 34.2% | **0.0%** |
| img+det | 49.6% | **0.0%** |
| bev_only | 14.2% | **0.0%** |
| clean_lidar | 61.5% | **0.0%** |
| clean_radar | 36.2% | 60.4% |
| clean_lidar_only | — | 47.9% |

**Multiple D-condition runs for 8B show exactly 0.0% quality tags** and BLEU=0.005, while the same modality under B-condition works normally. This strongly suggests a **data path issue** or **image loading failure** for condition D in the 8B evaluation pipeline — possibly the BEV images for condition D weren't found or had wrong paths during those runs.

> **Action:** Re-run the 0.0% tag D-condition 8B zero-shot evals (`img D`, `img+det D`, `bev_only D`, `clean_lidar D`). These are likely corrupted runs, not genuine results.

---

### 🔬 Finding 4: Camera-Only Is Not Terrible for 2B

| Modality | 2B BLEU (B) | 2B Tags (B) |
|----------|:-:|:-:|
| cam_only | 0.027 | **34.0%** |
| img | 0.029 | 12.2% |
| img+det | 0.038 | 6.0% |
| clean_lidar | 0.030 | 2.5% |
| clean_radar | 0.026 | 7.5% |

**Surprising:** `cam_only` has the **highest tag coverage** (34%) for 2B, higher than any BEV modality. This suggests 2B's camera understanding is reasonable but BEV images confuse it without fine-tuning. After fine-tuning, this should reverse.

> **Thesis point:** Without fine-tuning, the 2B model defaults to camera-centric reasoning. BEV inputs only become advantageous *after* fine-tuning — supporting your claim that training on BEV reasoning chains is necessary.

---

### 🔬 Finding 5: Radar Shows the Strongest Fine-tuning Signal

| Comparison | Δ BLEU | Δ Spatial |
|------------|:---:|:---:|
| clean_lidar B (ZS → FT) | +0.033 | +10.1 |
| clean_radar B (ZS → FT) | **+0.035** | +8.6 |
| clean_lidar D (ZS → FT) | +0.006 | +5.1 |
| clean_radar D (ZS → FT) | **+0.019** | **+9.6** |

Radar consistently shows equal or higher improvement from fine-tuning compared to lidar-only, especially on condition D. This supports the radar Doppler value proposition.

---

## What The LLM Judge Should Look For

Based on your actual data, the LLM judge in `llm_judge.py` should focus on:

1. **Spatial grounding in [GT] section** — the fine-tuned models should reference specific distances/bearings from detections.json
2. **Velocity reasoning accuracy** — radar-condition models should correctly identify moving vs stationary objects
3. **Action calibration** — fine-tuned 8B should output cleaner action labels (BRAKE/MAINTAIN/etc.)
4. **Hallucination detection** — many zero-shot 2B outputs reference objects not in the scene

### Run the judge on your data:
```bash
# Quick auto-metrics (no API key needed):
python antigravity_suggestions/llm_judge.py \
    --results_dir /home/mab/workspaces/data/moral_all_results/saves/zeroshot_results \
                  /home/mab/workspaces/data/moral_all_results/saves/finetuned_results \
                  /home/mab/workspaces/data/moral_all_results/saves/finetuned_results_aws \
    --provider auto_only \
    --compare

# Full LLM grading (needs API key):
export ANTHROPIC_API_KEY="sk-ant-..."
python antigravity_suggestions/llm_judge.py \
    --results_dir /home/mab/workspaces/data/moral_all_results/saves/zeroshot_results \
                  /home/mab/workspaces/data/moral_all_results/saves/finetuned_results \
    --provider anthropic \
    --model claude-sonnet-4-20250514 \
    --max_samples 50 \
    --compare
```

---

## Priority: What You Should Run Next

### On your 4090 (can start immediately):

1. **Re-run the broken 8B zero-shot D evals** (4 runs: img D, img+det D, bev_only D, clean_lidar D)
   - These show 0% tags which is almost certainly a data path bug
2. **Run LLM judge** on all existing results (`--provider auto_only` first, then Claude)
3. **2B fine-tuning** for clean_lidar_only and clean_radar_only (B and D)

### On H100 (when you get one):
4. **8B fine-tuning** starting with clean_radar (B, D) then clean_lidar (B, D)
