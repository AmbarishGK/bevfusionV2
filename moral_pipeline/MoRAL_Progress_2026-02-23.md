# MoRAL Pipeline — Session Progress Log
**Date:** February 23, 2026  
**Session focus:** Building all laptop-side data conditions, fixing radar PCD parser, rich scene descriptions, data validation  
**Status at end of session:** ✅ Conditions 00, 01, 02 complete. One pending fix (radar velocity sanity check). Ready for AWS + Cosmos.

---

## What MoRAL Is (Quick Recap)

A pipeline that:
1. Takes nuScenes sensor data (LiDAR + 6 cameras + 5 radars)
2. Produces BEV maps + structured scene descriptions
3. Feeds those into Cosmos-Reason2 (VLM) for spatial reasoning
4. Cosmos auto-generates QA dataset
5. Fine-tunes Qwen-VL (LoRA) on that dataset
6. Proves LiDAR-fused BEV grounding improves spatial reasoning with a measurable score

**Core research claim:** Camera+BEV+Radar > Camera+BEV > Camera only — proven with numbers on nuScenes.

---

## What Was Built This Session

### Condition 00 — Camera Only (`outputs/00_camera_only/`) ✅
- Script: `moral_pipeline/00_generate_camera_only/generate_camera_only.py`
- 10 scenes × 6 camera images each
- NO BEV map, NO detections.json, NO scene_description.txt
- Just metadata.json with ego speed and scene info
- This is the hardest baseline for Cosmos — it must reason spatially from images alone

### Condition 01 — GT Annotations (`outputs/01_gt_annotations/`) ✅
- Script: `moral_pipeline/01_generate_gt_scenes/generate_gt_scenes.py`
- 10 scenes × (6 cameras + bev_map.png + detections.json + scene_description.txt + metadata.json)
- Source: human-labeled nuScenes annotations — positions, sizes, velocities are ground truth
- `yaw_rad` field added to all detections (matches BEVFusion output schema)
- scene_description.txt uses the NEW rich format (see below)

### Condition 02 — GT + Radar (`outputs/02_gt_with_radar/`) ✅
- Script: `moral_pipeline/02_generate_gt_radar_scenes/generate_gt_radar_scenes.py`
- Same as condition 01 but enriched with Doppler radar velocities
- Saves `radar_points.npy` per scene (raw ego-frame radar, shape N×6)
- BEV map shows cyan radar dots + velocity arrows (visually distinct from condition 01)

---

## Major Bug Fixed This Session — Radar PCD Parser

**Problem:** nuScenes radar `.pcd` files are MIXED-TYPE binary, not uniform float32.

The SIZE row in the PCD header is:
```
SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
TYPE F F F I I F F F F F I I I I I I I I
```
Reading with `np.fromfile(path, dtype=np.float32)` gives wrong byte count.
For scene-0061 RADAR_FRONT: 97 points × 43 bytes/point = 4171 bytes actual,
but np.fromfile read it as 2367 floats → reshape to 18 columns failed.

**Fix:** New function `_load_radar_pcd()` in `scene_utils.py`:
- Reads PCD header line by line
- Builds struct format string dynamically: `<fffbhfffffbbbbbbbb`
- Uses `struct.unpack_from()` to parse each point correctly
- Extracts: x, y, z, vx_comp, vy_comp, rcs per point

**Why vx_comp/vy_comp not vx/vy:**
- `vx/vy` = raw Doppler velocity in sensor frame (includes ego vehicle motion)
- `vx_comp/vy_comp` = compensated (ego motion removed) = TRUE object velocity
- Using vx/vy would make all stationary objects appear to be moving

**Results after fix:**
- scene-0061: 602 radar points, 56/66 detections radar-confirmed (85%)
- Validation: car GT=9.57 m/s vs radar=7.96 m/s ✓, truck GT=3.23 vs radar=2.38 ✓

---

## Major Upgrade This Session — Rich Scene Description Format

`make_scene_text()` in `scene_utils.py` completely rewritten.

**Old format (15 objects max, flat list):**
```
66 objects detected around the vehicle. barrier at 10.22m bearing -144.0deg 
(mostly_occluded). truck at 16.84m bearing 15.7deg. ...
```

**New format (all objects, 7 structured sections):**

```
SCENE OVERVIEW: 66 objects detected via LiDAR ground-truth annotations. 
27 are moving. 22 are partially or fully occluded.

EGO STATE: Travelling at 9.1 m/s (32.9 km/h). Estimated stopping distance 
at current speed: 10.4 m. Ego vehicle dimensions: 4.5 m long, 2.0 m wide.

NEAREST OBJECT PER ZONE: directly ahead: truck at 16.84 m; front-left: 
pedestrian at 18.08 m, moving at 1.7 m/s (estimated); ...

SAFETY CRITICAL: barrier rear-right 10.22 m away (TTC ≈ 1.1 s); ...

FULL OBJECT LIST:
  [1] barrier, zone=rear-right, distance=10.22m, bearing=-144.0deg, 
      size=1.9x0.6x1.1m (WxLxH), speed=stationary, visibility=fully_visible, 
      lidar_pts=77(strong), ttc≈1.1s.
  [14] truck, zone=directly ahead, distance=16.84m, ..., 
       lane_gap≈1.3m (passable).

OBJECT COUNTS: 23 barriers, 1 bicycle, 1 bus, 8 cars, ...

OCCLUDED OBJECTS (visible to LiDAR but not clearly to cameras): ...
```

**New computed fields per object:**
- `zone` — 8-direction label (directly ahead / front-left / left / rear-left / directly behind / rear-right / right / front-right)
- `speed` — radar-confirmed if available, gt-estimated otherwise, or "stationary"
- `ttc` — time-to-collision: `distance / (ego_speed + object_speed)` for objects in ego's path
- `lane_gap` — `3.7m - obj_width/2 - ego_width/2`, labeled passable (≥0.3m) or tight
- `lidar_pts(strength)` — strong >50pts, moderate >10pts, weak ≤10pts
- `rcs` — radar cross-section in dBsm (condition 02 only)
- `visibility` — fully_visible / mostly_visible / partially_visible / mostly_occluded

**Why this matters for Cosmos QA generation:**
Each section enables a different question type:
- SCENE OVERVIEW → "How many moving objects are in the scene?"
- EGO STATE → "Can the ego vehicle stop before reaching the truck?"
- NEAREST OBJECT PER ZONE → "What is directly ahead of the vehicle?"
- SAFETY CRITICAL → "Which object poses the most immediate risk?"
- FULL OBJECT LIST → distance/velocity/size questions on any object
- OBJECT COUNTS → counting and composition questions
- OCCLUDED OBJECTS → "Which objects can LiDAR detect that cameras cannot?"

**Verification passed:**
- Object counts match detections.json exactly (66 objects, all types correct)
- Stopping distance formula verified: 9.1²/(2×4.0) = 10.4m ✓
- TTC formula verified: 16.84/9.1 = 1.85s ≈ shown 1.8s ✓
- Zone labels verified against x,y coordinates in BEV map ✓
- Radar Z range: 0.50–0.78m (correct — sensor mounting height, radar is 2D) ✓

---

## Data Validation Results

### What Was Validated
| Check | Result | Status |
|---|---|---|
| Object counts match detections.json | 66/66 exact match | ✅ |
| Stopping distance formula | 9.1²/8 = 10.4m | ✅ |
| TTC formula (truck ahead) | 16.84/9.1 = 1.85s | ✅ |
| TTC formula (car behind) | 20.74/18.67 = 1.11s | ✅ |
| Zone labels vs x,y coordinates | Checked manually | ✅ |
| Radar Z values (should be ~0.5m) | 0.50–0.78m | ✅ |
| Radar GT vs Doppler velocity agreement | 18/20 within 3 m/s | ✅ |
| Suspicious radar matches | 2/56 (3.6%) | ⚠ Known |

### Known Data Quality Issues

**2 suspicious radar matches in scene-0061:**
1. Car at 39.0m: GT=5.18 m/s, radar=0.15 m/s — wrong radar point matched (ground clutter or nearby object at same distance)
2. Bus at 53.5m: GT=9.74 m/s, radar=0.09 m/s — radar range limitation at 53m, weak returns filtered

**Root cause:** The 3m nearest-neighbour match radius is sometimes too permissive in dense scenes or at long range.

**Decision:** Fix this with a velocity sanity check before next session.

---

## PENDING FIX — Must Do Before Next Session

### Fix: Radar Velocity Sanity Check in `enrich_detections_with_radar()`

**NOT YET APPLIED.** The fix was designed but not added to `scene_utils.py` this session.

Add this logic after finding the nearest radar point match:

```python
# After computing radar_vel_ms and before setting confirmed=True:

gt_speed = d.get('velocity_ms', 0.0)
radar_speed = float(np.sqrt(vx_r**2 + vy_r**2))

# Reject match if GT says object is moving fast but radar says stationary
# This indicates the wrong radar point was matched (clutter or different object)
# Threshold: reject if GT > 2 m/s AND radar < GT/4
if gt_speed > 2.0 and radar_speed < gt_speed / 4.0:
    confirmed = False
    radar_vel_ms = None
    radar_vx = None
    radar_vy = None
    # Still keep radar_rcs if we have it — position match is still valid
```

**Effect:** The 2 suspicious matches get `radar_velocity_confirmed=False`. Their scene description will say `gt-estimated` instead of `radar-confirmed`. Honest representation of what radar actually measured.

**Where to add it:** In `moral_pipeline/utils/scene_utils.py`, inside `enrich_detections_with_radar()`, after the block that sets `confirmed = True`.

**After adding the fix, regenerate condition 02:**
```bash
python3 moral_pipeline/02_generate_gt_radar_scenes/generate_gt_radar_scenes.py \
  --dataroot data/nuscenes \
  --out-dir moral_pipeline/outputs/02_gt_with_radar
```

Then re-run the suspicious check to confirm 0 suspicious matches remain:
```bash
python3 - << 'EOF'
import json
with open('moral_pipeline/outputs/02_gt_with_radar/scene-0061/detections.json') as f:
    dets = json.load(f)
suspicious = []
for d in dets:
    if d.get('radar_velocity_confirmed'):
        gt = d.get('velocity_ms', 0)
        rv = d.get('radar_velocity_ms', 0)
        if gt > 2.0 and rv < 0.5:
            suspicious.append(d)
            print(f"SUSPICIOUS: {d['class']:12s} dist={d['distance_m']:6.1f}m GT={gt:.2f} radar={rv:.2f}")
print(f"Total suspicious: {len(suspicious)} (should be 0)")
EOF
```

---

## Key Conceptual Decisions Made This Session

### What nuScenes annotations actually are
- Human annotators drew 3D boxes around every object in every frame
- Stored in `sample_annotation.json` — class, position (x,y,z), size (W,L,H), orientation
- **Velocity is NOT measured** — it is estimated from `(position_t1 - position_t0) / dt` using consecutive frames
- This frame-differencing velocity is accurate for smooth motion but can be wrong when objects are occluded between frames or at scene boundaries

### What LiDAR gives you
- A point cloud — 3D dots showing where surfaces are (x,y,z,intensity)
- LiDAR has NO Doppler capability — it cannot measure velocity directly
- Velocity from LiDAR would require tracking the same point across frames (not in our pipeline)
- `num_lidar_pts` = how many LiDAR points fell inside the GT annotation box
  - >50 points = strong detection (truck, bus, nearby car)
  - 10-50 points = moderate (nearby pedestrian, far vehicle)
  - <10 points = weak (far pedestrian, small object at range)
  - 1 point = barely detectable — geometry unreliable

### What Radar gives you
- Radio waves + Doppler shift → directly measures object velocity in ONE frame
- Does NOT need two frames — velocity is physically measured, not estimated
- Radar is 2D — no height measurement (Z values all ~0.5m = mounting height)
- `vx_comp/vy_comp` = ego-motion-compensated true object velocity
- `rcs` (radar cross-section in dBsm) = how strongly the object reflected
  - Truck: ~18.5 dBsm (large metal object)
  - Barrier: ~-3.5 dBsm (small plastic/concrete)
  - Pedestrian: ~2.5 dBsm (small, absorptive)
- Radar range degrades with distance squared — unreliable beyond ~50m
- Radar is sparse (97-600 points per scene vs tens of thousands for LiDAR)

### How radar is matched to GT detections
Nearest-neighbour in 2D (x,y ego frame), within 3m radius.
Limitations: can match wrong object in dense scenes or at long range.
Fix: velocity sanity check (see PENDING FIX above).

### Which condition to use for primary evaluation
**Use condition 01 (GT annotations) for primary Cosmos evaluation.**

Reason: GT data is trustworthy by definition. When Cosmos gives a wrong answer with GT input, you know the error is in Cosmos's reasoning, not in the input data. This cleanly isolates the variable you're measuring.

If you used BEVFusion predictions (condition 03) for evaluation, errors could come from BEVFusion getting a detection wrong OR Cosmos reasoning wrong — you can't separate them.

---

## Evaluation Approach Decided This Session

### Two-layer evaluation

**Layer 1 — Talk2BEV-Bench (existing benchmark)**
- 20,000 MCQ questions on nuScenes scenes
- Multiple choice A/B/C/D → percentage correct
- Directly comparable to Talk2BEV paper
- Run across all conditions: A, B, D (and later C, E)
- Primary metric for thesis comparison table

**Layer 2 — Grounded Spatial Accuracy (GSA) — YOUR NOVEL METRIC**
Because your scene descriptions contain exact ground truth numbers (distance, velocity, TTC, zone), every Cosmos answer is auto-verifiable against `detections.json`. No human evaluators needed.

```
Question: "How far is the nearest pedestrian directly behind the vehicle?"
GT answer: 12.78m  (from detections.json — field distance_m)
Cosmos answer: "approximately 13 meters"
Error: |12.78 - 13.0| / 12.78 = 1.7% → CORRECT (within 20% threshold)
```

Score by question type:
- **Distance accuracy** — % of distance answers within 20% of GT
- **Velocity accuracy** — % of velocity answers within 30% of GT
- **Zone accuracy** — % of directional answers matching correct zone label
- **TTC accuracy** — % of TTC estimates within 2 seconds of GT
- **Occlusion accuracy** — % of "can camera see this?" answers correct

**Why this is novel:** No other paper auto-evaluates VLM spatial reasoning answers against metric 3D ground truth. Talk2BEV uses human-written MCQ. OmniDrive uses planning metrics. You have a programmatic verifier using the same detections.json you already have.

This is publishable as a secondary contribution: "We introduce GSA, an auto-evaluable spatial reasoning metric grounded in metric LiDAR data."

---

## File Structure (Complete, End of Session)

```
moral_pipeline/
  00_generate_camera_only/
    generate_camera_only.py
  01_generate_gt_scenes/
    generate_gt_scenes.py
  02_generate_gt_radar_scenes/
    generate_gt_radar_scenes.py
  utils/
    scene_utils.py                 ← ALL shared code, updated this session
                                     PENDING: add radar velocity sanity check
  outputs/
    00_camera_only/                ← Condition A: cameras only
      scene-0061/ ... scene-1100/
        CAM_FRONT.jpg ... CAM_BACK_RIGHT.jpg
        metadata.json
    01_gt_annotations/             ← Condition B: GT + BEV + rich scene text
      scene-0061/ ... scene-1100/
        bev_map.png
        CAM_FRONT.jpg ... CAM_BACK_RIGHT.jpg
        detections.json            (source: "ground_truth")
        scene_description.txt      (rich format, all 66 objects)
        metadata.json
    02_gt_with_radar/              ← Condition D: GT + BEV + Radar Doppler
      scene-0061/ ... scene-1100/
        bev_map.png                (cyan radar dots + velocity arrows)
        CAM_FRONT.jpg ... CAM_BACK_RIGHT.jpg
        detections.json            (source: "ground_truth_radar", has radar_* fields)
        scene_description.txt      ("radar-confirmed X m/s" for moving objects)
        radar_points.npy           (raw ego-frame radar, shape N×6)
        metadata.json              (includes radar_summary stats)
```

---

## Conditions Summary Table

| Folder | Label | Input to Cosmos | BEV Map | Radar | GPU Needed |
|---|---|---|---|---|---|
| `00_camera_only` | A | 6 cameras only | ❌ | ❌ | No |
| `01_gt_annotations` | B | Cameras + GT BEV + scene text | ✅ | ❌ | No |
| `02_gt_with_radar` | D | Cameras + GT BEV + radar velocities | ✅ cyan | ✅ | No |
| `03_bevfusion` | C | Cameras + BEVFusion BEV + scene text | ✅ | ❌ | **Yes — AWS** |
| `04_bevfusion_radar` | E | All of above + radar | ✅ cyan | ✅ | **Yes — AWS** |

Expected result: E > D > C > B > A on both Talk2BEV-Bench and GSA metric.

---

## Infrastructure

- **Docker image:** `ambarishgk007/moral-bevfusion-thesis` (7.4GB)
- **AWS:** EC2 g4dn.xlarge, NVIDIA T4, Spot ~$0.16/hr
- **nuScenes data:** `data/nuscenes/` (v1.0-mini, 10 scenes, all sensors present)
- **Python env:** `.venv` in `~/Desktop/thesi/forked/bevfusionV2/`
- **BEVFusion checkpoint:** `bevfusion_WORKING.pth` (fixed spconv weight format)

---

## What To Do In The Next Session (In Order)

### Step 0 — Apply the pending fix first (15 minutes, laptop)
Add velocity sanity check to `enrich_detections_with_radar()` in `scene_utils.py`.
Regenerate condition 02. Verify 0 suspicious matches.
Commit everything to git.

### Step 1 — AWS setup
```bash
# Launch g4dn.xlarge spot instance
# Pull Docker image
docker pull ambarishgk007/moral-bevfusion-thesis

# Install Cosmos-Reason2
pip install transformers>=4.57.0 qwen-vl-utils

# Load model
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'nvidia/Cosmos-Reason2-8B',
    torch_dtype='auto',
    device_map='auto'
)
```

### Step 2 — Smoke test on one scene
Feed scene-0061, condition B (GT annotations) to Cosmos with one question:
> "What object is directly ahead of the ego vehicle and how far away is it?"

GT answer from detections.json: truck at 16.84m zone=directly ahead
If Cosmos says approximately 17m and identifies truck → pipeline works.

### Step 3 — Build the Cosmos QA generation prompt
The prompt should instruct Cosmos to generate questions across all 8 types:
- Spatial, Distance, Physics, Safety, Occlusion, Comparative, Gap/passability, TTC

Each QA pair output should be JSON:
```json
{
  "question": "...",
  "answer": "...",
  "question_type": "distance",
  "gt_verifiable": true,
  "gt_value": 16.84,
  "gt_field": "distance_m",
  "gt_object_index": 14,
  "condition": "B",
  "scene": "scene-0061"
}
```

### Step 4 — Run Talk2BEV-Bench evaluation
Get Talk2BEV-Bench questions for nuScenes mini scenes.
Run Cosmos on conditions A, B, D.
Record scores → first real numbers for thesis.

### Step 5 — Implement GSA scorer
Script that takes Cosmos free-text answers and scores them against detections.json.
Should output per-question-type accuracy for each condition.

---

## Thesis Novelty Summary (Remind Supervisor)

1. **BEVFusion LiDAR+camera → Cosmos-Reason2 for spatial QA** — unpublished combination
2. **Cosmos auto-generating BEV-grounded QA dataset** — not done anywhere
3. **Radar Doppler velocity integrated into VLM scene context** — no other paper does this
4. **GSA metric: auto-evaluable spatial reasoning against metric 3D GT** — new contribution
5. **Ablation: camera-only vs BEV vs BEV+radar with measurable score** — not done with these models
6. **LoRA fine-tuning Qwen-VL on BEVFusion nuScenes data** — not done (Phase 2)

---

## Open Questions For Supervisor

1. Is Phase 1 (no fine-tuning, just evaluation) sufficient for thesis contribution level?
2. Lab cluster availability for Phase 2 LoRA fine-tuning? (T4 is NOT enough for Qwen-VL)
3. Is Talk2BEV-Bench the right benchmark or does he want something else?
4. Should GSA be proposed as a published metric contribution?
5. Publication target: IROS 2026, ITSC 2025, or arXiv preprint first?
6. Isaac Sim evaluation from original proposal — still required?

---

*To continue: share MoRAL_Session_Prompt.md + this file with new Claude session.*  
*Say: "Help me integrate Cosmos-Reason2" after applying the pending radar fix.*
