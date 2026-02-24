# MoRAL Pipeline — Session Handoff
**Date:** 2026-02-23 (end of Session 2)
**For:** Fresh Claude session — read completely before doing anything
**Do not:** Rewrite scene_utils.py, suggest switching frameworks, or redesign the pipeline

---

## Thesis In One Paragraph

**MoRAL: Multimodal Reasoning for Autonomous Language Models with LiDAR/Sensor-Enhanced BEV Fusion**

Use Cosmos-Reason2 (large teacher VLM) to generate structured chain-of-thought QA pairs grounded in nuScenes LiDAR/radar data. Fine-tune Qwen-VL-7B (small student model) on these pairs so it learns to do spatial reasoning from BEV maps + camera images alone — without needing Cosmos at inference time. Result: a fast, deployable model that reasons about driving scenes using metric spatial understanding that camera-only VLMs lack. Counterfactual reasoning (OmniDrive-style) is future work.

---

## Pipeline Flow (Complete Picture)

```
nuScenes data (mini: 10 scenes, or val: 150 scenes)
        ↓
[LAPTOP — DONE ✅]
scene_utils.py generates per scene:
  - bev_map.png          (LiDAR point cloud + GT detection boxes, forward=up)
  - CAM_FRONT.jpg        (+ 5 other cameras)
  - scene_description.txt (structured natural language, 8 sections)
  - detections.json      (all objects with distance, velocity, radar quality)
  - metadata.json

3 conditions done (10 scenes each):
  A: 00_camera_only   — cameras + metadata only (baseline, hardest for VLM)
  B: 01_gt_annotations — cameras + BEV + detections + scene_description
  D: 02_gt_with_radar  — same as B + radar Doppler enrichment

        ↓
[AWS — NEXT STEP]
Cosmos-Reason2 (TEACHER MODEL)
  Input:  BEV map + scene_description.txt (grounding context)
  Task:   Generate structured QA pairs with chain-of-thought reasoning
  Output per scene: ~10 QA pairs covering:
    - spatial   ("What is directly ahead and how far?")
    - safety    ("What is the most immediate collision risk?")
    - velocity  ("Which objects are moving and how fast?")
    - occlusion ("What does LiDAR detect that cameras miss?")
    - gap       ("Is there room to pass the truck?")
    - physics   ("Can ego stop before reaching the pedestrian?")
    - zone      ("What is in the front-left zone?")
    - comparative ("Which car ahead is moving faster?")

QA pair format:
  {
    "question": "...",
    "reasoning": "chain-of-thought explanation referencing metric values",
    "answer": "short final answer",
    "question_type": "safety|spatial|velocity|...",
    "gt_verifiable": true,
    "gt_value": 16.84,
    "gt_field": "distance_m",
    "scene": "scene-0061",
    "condition": "B"
  }

        ↓
Training dataset built from QA pairs:
  input_image_1 = bev_map.png
  input_image_2 = CAM_FRONT.jpg
  output        = reasoning + answer  (chain-of-thought)

        ↓
[LAB CLUSTER / AWS A100 — LATER]
Fine-tune Qwen-VL-7B with LoRA
  Input at inference: BEV map + CAM_FRONT only (no scene_description)
  Output: spatial reasoning + answer
  Goal: model learns metric spatial reasoning from images alone

        ↓
Evaluation:
  Baseline A:   Qwen-VL zero-shot, camera only (condition A)
  Baseline B:   Qwen-VL zero-shot, BEV + front camera (condition B)
  Fine-tuned:   Qwen-VL after LoRA, BEV + front camera
  Metric:       Accuracy on held-out QA pairs (auto-scorable against detections.json)
  Ablation:     BEV only vs BEV+front vs BEV+all6 cameras
```

---

## Current Status — What Is Done

### Laptop data generation: ✅ Complete

All files at: `~/Desktop/thesi/forked/bevfusionV2/moral_pipeline/`

```
outputs/
  00_camera_only/     10 scenes × (6 cameras + metadata)
  01_gt_annotations/  10 scenes × (BEV + cameras + detections + scene_desc + metadata)
  02_gt_with_radar/   10 scenes × (same + radar_points.npy + radar-enriched detections)
```

### scene_utils.py: ✅ Finalised (887 lines)

Located at: `moral_pipeline/utils/scene_utils.py`

**Do not rewrite. Edit surgically if needed.**

Key features:
- BEV orientation correct: forward = UP, left = left (transform: `plot_x=-y_ego, plot_y=x_ego`)
- Radar quality classification: `reliable` / `radial_ambiguous` / `range_mismatch` / `unconfirmed`
- Pedestrian radar: uses GT velocity (radar radial-only physics limitation)
- TTC correct: only computed ahead (closing speed) or behind (only if overtaking)
- Two-tier scene description: DETAILED (≤50m, ≥3 LiDAR pts) vs MARGINAL (count only)
- BEVFusion compatible: uses confidence threshold instead of lidar_pts when source=bevfusion

---

## Unresolved Issue — Annotation Gaps

nuScenes mini GT annotations have gaps. Oncoming lane vehicles sometimes not annotated.

**Scan result:**
```
✅ scene-0061: clean (3 cars ahead confirmed)
⚠️  scene-0103: bridge scene — 2 cars visible in CAM_FRONT, zero GT boxes for them
✅ scene-0553: stopped, legitimate empty forward cone
✅ scene-0655: clean
⚠️  scene-0757: ego 5.2m/s, no cars ahead — NEEDS CAM_FRONT VISUAL CHECK
⚠️  scene-0796: ego 12.7m/s, no cars ahead — NEEDS CAM_FRONT VISUAL CHECK
✅ scene-0916: slow (4m/s), legitimate
⚠️  scene-1077: ego 6.6m/s, no cars ahead — NEEDS CAM_FRONT VISUAL CHECK
✅ scene-1094: clean
✅ scene-1100: stopped, legitimate
```

**scene-0103 is confirmed bad** — camera shows 2 clear cars ahead, GT has none.
**3 scenes still need visual check** before deciding to keep or drop.

To check:
```bash
xdg-open moral_pipeline/outputs/01_gt_annotations/scene-0757/CAM_FRONT.jpg
xdg-open moral_pipeline/outputs/01_gt_annotations/scene-0796/CAM_FRONT.jpg
xdg-open moral_pipeline/outputs/01_gt_annotations/scene-1077/CAM_FRONT.jpg
```

Rule: if camera shows unannotated vehicles clearly ahead → drop scene from training/eval.

---

## Data Strategy — Decision Needed

**Constraint:** 209GB free disk. nuScenes trainval = ~320GB. Does not fit.

**Option A — Stay on mini (recommended if timeline is tight)**
- Use 6-8 verified clean scenes
- Cosmos generates ~10 QA pairs per scene = 60-80 training pairs
- Frame as proof-of-concept in thesis
- Honest and defensible for MSc

**Option B — Get trainval (recommended if you have time + can free space)**
- Free ~120GB (check: Docker images, Downloads, build caches)
- Download nuScenes trainval (~320GB) from nuscenes.org
- Run pipeline on val split (150 scenes)
- Cosmos generates 1500 QA pairs = proper training dataset
- Much stronger thesis + publishable paper

Check disk before deciding:
```bash
du -sh ~/Downloads/ ~/.cache/
docker system df
df -h
```

**If going for trainval:** same scripts work, just change `--version v1.0-trainval` and
`--out-dir` in both generate scripts. scene_utils.py needs zero changes.

---

## Next Session Tasks (In Order)

### Task 1 — Visual check 3 flagged scenes (10 min)
Open CAM_FRONT for scenes 0757, 0796, 1077. Decide keep or drop.
Update the clean scene list.

### Task 2 — Decide mini vs trainval (5 min)
Run disk check. Make the call.

### Task 3 — AWS Cosmos smoke test (1-2 hrs)

```bash
# Launch g4dn.xlarge spot (~$0.16/hr)
docker pull ambarishgk007/moral-bevfusion-thesis:latest

# Inside container
pip install transformers>=4.57.0 qwen-vl-utils accelerate

# Test script (write this in the session)
# Feed scene-0061 condition B:
#   - bev_map.png
#   - scene_description.txt (as system context for Cosmos)
# Question: "What is directly ahead of the ego vehicle and how far?"
# Expected answer: truck at 16.84m
# If Cosmos gets this right → pipeline is working
```

### Task 4 — Cosmos QA generation script (1-2 hrs)
Write `02_generate_cosmos_qa/generate_cosmos_qa.py` that:
- Loops over all clean scenes
- Feeds BEV + scene_description to Cosmos
- Generates 8-10 QA pairs per scene (one per question type)
- Saves as `cosmos_qa/scene-XXXX/qa_pairs.json`
- Saves training pairs as `(bev_map.png, CAM_FRONT.jpg, reasoning+answer)`

---

## Infrastructure Reference

```
Laptop:   Dell G16-7630, Ubuntu
          ~/Desktop/thesi/forked/bevfusionV2/
          .venv (Python 3.10, activated with: source .venv/bin/activate)

Docker:   ambarishgk007/moral-bevfusion-thesis (BEVFusion GPU inference, 7.4GB)

AWS:      g4dn.xlarge, NVIDIA T4 GPU
          Spot price: ~$0.16/hr
          BEVFusion checkpoint: inside Docker as bevfusion_WORKING.pth

nuScenes: data/nuscenes/ (v1.0-mini only, 10 scenes)
          samples/ sweeps/ maps/ v1.0-mini/
```

---

## Key Constants In scene_utils.py

```python
BEV_RANGE_M       = 50.0   # plot range + radar clip
BEV_LABEL_RANGE   = 35.0   # max distance for box labels on BEV
MIN_STRONG_PTS    = 10     # min LiDAR pts for solid box + label
MIN_VLM_PTS       = 3      # min LiDAR pts for scene description detail
DESC_DETAIL_RANGE = 50.0   # max distance for full object detail
BRAKE_DECEL       = 4.0    # m/s²
EGO_WIDTH_M       = 2.0
EGO_LENGTH_M      = 4.5
RADIAL_AMBIGUOUS_CLASSES = {'pedestrian', 'bicycle', 'motorcycle'}

# BEV coordinate transform (forward=up, left=left)
def px(x_ego, y_ego): return -y_ego
def py(x_ego, y_ego): return  x_ego
```

---

## Thesis Evaluation Story (What To Present)

**Three-row results table:**

| Model | Input | Spatial Acc | Safety Acc | Velocity Acc | Overall |
|---|---|---|---|---|---|
| Qwen-VL zero-shot | Camera only (A) | ? | ? | ? | ? |
| Qwen-VL zero-shot | BEV + front cam (B) | ? | ? | ? | ? |
| Qwen-VL fine-tuned | BEV + front cam (B) | ? | ? | ? | ? |

**Ablation table:**

| Fine-tuned Qwen input | Overall Acc |
|---|---|
| BEV map only | ? |
| Front camera only | ? |
| BEV + front camera | ? |
| BEV + all 6 cameras | ? |

**The story:** Fine-tuned Qwen on Cosmos-generated BEV reasoning data outperforms
zero-shot Qwen on spatial/safety questions. BEV input is necessary — camera alone
misses metric distances. Radar adds measurable improvement on velocity questions.

**Novelty claim:** First work to use Cosmos-Reason2 as a teacher to distill
BEV spatial reasoning into a fine-tuned Qwen-VL student model for autonomous driving.

---

## What To Say To Start Next Session

Paste this as your first message:

```
I'm working on my MSc thesis — MoRAL: Multimodal Reasoning for Autonomous
Language Models with LiDAR/Sensor-Enhanced BEV Fusion.

Please read the attached progress file completely before doing anything.
[attach: MoRAL_Progress_2026-02-23_Session3_Handoff.md]
[attach: scene_utils.py]
[attach: CAM_FRONT images for scene-0757, scene-0796, scene-1077]

The pipeline: nuScenes → BEV maps + cameras + scene descriptions →
Cosmos-Reason2 generates QA pairs → fine-tune Qwen-VL on those pairs →
deployable small model that reasons about BEV maps from images alone.

Today's tasks in order:
1. Check the 3 attached CAM_FRONT images — are those scenes clean or do
   they have annotation gaps? Tell me which scenes to keep.
2. Decide: mini (209GB free, trainval needs 320GB) or find space for trainval?
3. Write the Cosmos QA generation script and test it on scene-0061.

Do not rewrite scene_utils.py. Do not redesign the pipeline.
Ask me before making any architectural decisions.
```
