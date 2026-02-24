# MoRAL Pipeline — Session Progress Log
**Date:** February 23, 2026 (final update)
**Status:** ✅ All laptop work complete. scene_utils.py finalised. Ready for AWS + Cosmos.

---

## What Is Done — Complete Picture

### Three Conditions Built (10 scenes each, all validated)

| Folder | Label | Contents | Script |
|---|---|---|---|
| `outputs/00_camera_only` | A | 6 cameras + metadata.json | `00_generate_camera_only/generate_camera_only.py` |
| `outputs/01_gt_annotations` | B | cameras + bev_map + detections.json + scene_description.txt + metadata.json | `01_generate_gt_scenes/generate_gt_scenes.py` |
| `outputs/02_gt_with_radar` | D | same as B + radar Doppler + radar_points.npy + cyan BEV | `02_generate_gt_radar_scenes/generate_gt_radar_scenes.py` |

### Conditions Still Needed (AWS GPU)

| Folder | Label | What's needed |
|---|---|---|
| `outputs/03_bevfusion` | C | BEVFusion model inference |
| `outputs/04_bevfusion_radar` | E | BEVFusion + radar enrichment |

### To Regenerate (if ever needed)
```bash
python3 moral_pipeline/01_generate_gt_scenes/generate_gt_scenes.py \
  --dataroot data/nuscenes --out-dir moral_pipeline/outputs/01_gt_annotations

python3 moral_pipeline/02_generate_gt_radar_scenes/generate_gt_radar_scenes.py \
  --dataroot data/nuscenes --out-dir moral_pipeline/outputs/02_gt_with_radar \
  --match-radius 2.0
```

---

## scene_utils.py — Complete Feature List (Current Final Version)

File: `moral_pipeline/utils/scene_utils.py` (887 lines)

### Constants
```python
BEV_RANGE_M       = 50.0   # plot range and radar clip distance
BEV_LABEL_RANGE   = 35.0   # max distance for box labels on BEV
MIN_STRONG_PTS    = 10     # min LiDAR pts for solid box + label
MIN_VLM_PTS       = 3      # min LiDAR pts for full detail in scene text
DESC_DETAIL_RANGE = 50.0   # max distance for detailed scene description
BRAKE_DECEL       = 4.0    # m/s² braking deceleration
EGO_WIDTH_M       = 2.0
EGO_LENGTH_M      = 4.5
RADIAL_AMBIGUOUS_CLASSES = {'pedestrian', 'bicycle', 'motorcycle'}
```

### `_load_radar_pcd(pcd_path)`
Parses nuScenes mixed-type binary PCD files correctly.
- Reads header dynamically → builds struct format `<fffbhfffffbbbbbbbb`
- Uses `struct.unpack_from()` — NOT `np.fromfile(dtype=float32)`
- Extracts: `[x, y, z, vx_comp, vy_comp, rcs]` per point
- Uses `vx_comp/vy_comp` (ego-motion-compensated) NOT `vx/vy`

### `get_radar_points_ego()`
- Loads all 5 radar sensors, transforms to ego frame
- Clips to 55m — removes ghost/multipath returns (previously went to 124m)
- Returns shape (N, 6): `[x_ego, y_ego, z_ego, vx_ego, vy_ego, rcs]`

### `enrich_detections_with_radar(detections, radar_pts, match_radius_m=2.0)`
Adds `radar_quality` field to every detection:

| Quality | Meaning | Velocity used in scene text |
|---|---|---|
| `reliable` | Vehicle, radar speed agrees with GT | radar-confirmed |
| `radial_ambiguous` | Pedestrian/cyclist — crosswise motion invisible to Doppler | gt-estimated |
| `range_mismatch` | Vehicle where GT says fast but radar says stationary (wrong match) | gt-estimated |
| `unconfirmed` | No radar point within 2m | gt-estimated |

Match radius reduced from 3.0m → 2.0m to reduce wrong matches.
Velocity sanity check: if GT > 2 m/s and radar < 0.5 m/s → `range_mismatch`.

scene-0061 breakdown: reliable=27, radial_ambiguous=12, range_mismatch=2, unconfirmed=25

### `_ttc(dist_m, obj_spd, ego_spd, bearing_deg)` — FIXED THIS SESSION
**Old (wrong):** computed TTC for ALL objects ahead OR behind using `closing = ego + obj speed`
This incorrectly flagged stationary barriers/cones behind the vehicle as safety critical.

**New (correct):**
- **Ahead** (|bearing| < 45°): `closing = ego_speed + obj_speed` ← ego approaching
- **Behind** (|bearing| > 135°): TTC only if `obj_speed > ego_speed + 1.0` ← overtaking from rear
- **Side** (45–135°): no TTC

Effect: stationary traffic_cone rear-right 11m away no longer appears in SAFETY CRITICAL.

### `_lidar_tier(d)` — FIXED THIS SESSION
**Old:** `_lidar_tier(pts)` — took integer, only worked for GT.

**New:** `_lidar_tier(d)` — takes full detection dict:
- GT mode (has `num_lidar_pts`): high >50, medium >10, low ≥3, marginal <3
- BEVFusion mode (`num_lidar_pts` absent): maps confidence to same tiers
  - high ≥0.8, medium ≥0.5, low ≥0.3, marginal <0.3

### `make_scene_text()` — Tiering Also Fixed For BEVFusion

Old: `d.get('num_lidar_pts', 99) >= MIN_VLM_PTS` — default 99 put all BEVFusion detections in detailed tier regardless of confidence.

New:
```python
if is_bev:
    detailed = [d for d in detections
                if d['distance_m'] <= DESC_DETAIL_RANGE
                and d.get('confidence', 0.0) >= 0.3]
else:
    detailed = [d for d in detections
                if d['distance_m'] <= DESC_DETAIL_RANGE
                and d.get('num_lidar_pts', 99) >= MIN_VLM_PTS]
```

Eight sections in scene description:
1. SCENE OVERVIEW — object counts, data source, radar note
2. EGO STATE — speed, stopping distance, vehicle size
3. NEAREST OBJECT PER ZONE — 8 directional zones
4. SAFETY CRITICAL — TTC < 5s (with corrected TTC logic)
5. OBJECTS WITHIN 50m — full detail for high-confidence objects
6. MARGINAL DETECTIONS — count-only summary for far/sparse objects
7. TOTAL OBJECT COUNTS — all classes, all distances
8. OCCLUDED — LiDAR-visible but camera-may-miss

### `save_bev_map()` — MAJOR OVERHAUL THIS SESSION

**New CLASS_COLORS dict — fixed assignment for all 10 classes:**
```python
'car':                  '#4c8eda'   # blue
'truck':                '#f0883e'   # orange
'bus':                  '#e05252'   # red
'trailer':              '#c678dd'   # purple
'motorcycle':           '#e5c07b'   # yellow
'bicycle':              '#56b6c2'   # teal
'pedestrian':           '#98c379'   # green
'traffic_cone':         '#00d4ff'   # cyan
'barrier':              '#c8a97e'   # tan/brown
'construction_vehicle': '#3cb371'   # medium-green
```
Previously used `plt.cm.tab10` which assigned colors by CLASS_LIST index — unpredictable. Now fixed per-class.

**Visual hierarchy (new):**
1. Radar dots → LiDAR cloud → detection boxes → labels → ego (layered z-order)
2. Strong objects (≥10 LiDAR pts): solid fill + label + color edge
3. Radar-confirmed objects: cyan edge highlight
4. Moving objects (GT mode): velocity arrow from vx/vy
5. Weak objects: ghost outline only, no label, no fill

**Forward direction arrow on ego** — white arrow pointing forward so VLM can orient itself.

**Legend fixes:**
- Always includes ALL classes present in scene (was only classes with strong detections)
- Radar point entry added when radar data present
- Legend ordered: vehicles first, infrastructure, radar
- `bicycle` and `construction_vehicle` no longer missing

**Axis labels updated:** "X (meters — forward)" and "Y (meters — left)" so VLM understands orientation.

**Radar improvements:**
- Dots: s=2.5, alpha=0.35 (smaller and more transparent)
- Arrows: capped at 4m visual, speed threshold >2 m/s
- Drawn under LiDAR (zorder 2 vs 3)

---

## All Fixes Applied This Session (Chronological)

1. **Radar PCD parser** — mixed-type binary, struct.unpack_from (earlier session, in place)
2. **Radar range clip** — 55m filter on get_radar_points_ego (earlier session, in place)
3. **Radar quality classification** — reliable/radial_ambiguous/range_mismatch/unconfirmed (earlier session, in place)
4. **Match radius 3m → 2m** — reduces wrong matches in dense scenes
5. **_best_vel()** — uses GT for pedestrians (radial ambiguity), radar for reliable vehicles
6. **Two-tier scene description** — 50m + 3pt threshold splits detailed from marginal
7. **TTC rear-object fix** — stationary objects behind are not safety critical ← THIS SESSION
8. **_lidar_tier dict-based** — BEVFusion confidence fallback ← THIS SESSION
9. **make_scene_text BEVFusion tiering** — confidence ≥0.3 threshold ← THIS SESSION
10. **BEV CLASS_COLORS** — fixed per-class colors, all 10 classes ← THIS SESSION
11. **BEV forward arrow on ego** — orientation indicator ← THIS SESSION
12. **BEV GT velocity arrows** — moving objects get arrows from vx/vy ← THIS SESSION
13. **BEV legend completeness** — all classes + radar entry, sorted order ← THIS SESSION
14. **BEV axis labels** — "forward/left" direction clarified ← THIS SESSION

---

## Data Quality Summary

### Validated For scene-0061
| Check | Result |
|---|---|
| Object counts | 66 exact match detections.json ↔ scene_description.txt ✅ |
| Stopping distance | 9.1²/8 = 10.4m ✅ |
| TTC ahead (truck 16.84m) | 16.84/9.1 = 1.85s ✅ |
| TTC closing (car behind) | 20.74/18.67 = 1.11s ✅ |
| Radar Z range | 0.50–0.78m (sensor mounting height, correct) ✅ |
| Suspicious radar matches | 2/56 = 3.6% (flagged as range_mismatch) ✅ |
| Zone labels vs x,y coords | Verified manually ✅ |
| BEV map geometry | Objects at correct positions vs scene_description ✅ |

### Known Limitations (Document In Thesis)
- **Pedestrian radar**: crosswise walking = near-zero Doppler. Not a bug — radar physics. GT velocity used.
- **TTC is approximate**: uses simplified closing speed formula, not full trajectory prediction.
- **Radar match 2m radius**: occasionally wrong in very dense object clusters.
- **1-2 LiDAR point detections**: present in detections.json but relegated to MARGINAL tier in scene text.

---

## Sensor Data Explained (For Thesis Background)

### nuScenes GT Annotations
Human-drawn 3D boxes stored in `sample_annotation.json`.
- Position, size, orientation: directly annotated
- **Velocity: derived** from `(pos_t1 - pos_t0) / dt` — frame differencing, not measured
- Can be unreliable at scene boundaries or when objects are briefly occluded

### LiDAR
- 3D point cloud (x,y,z,intensity) — geometric surface map
- No velocity — purely geometric
- `num_lidar_pts` inside a box = evidence quality
  - 495 pts (truck 17m) = very strong
  - 1 pt (pedestrian 68m) = barely detectable

### Radar
- Radio waves + Doppler shift = single-frame velocity measurement
- 2D only (Z = mounting height ~0.5m)
- `vx_comp/vy_comp` = ego-motion-compensated true object velocity
- `rcs` (dBsm) = radar cross-section (size/reflectivity proxy)
  - Truck: ~18.5 dBsm; Car: 3–9; Pedestrian: ~2.5; Barrier: ~-3.5
- Range degrades with distance²; unreliable beyond ~50m
- Blind to motion perpendicular to beam (radial-only)

---

## Evaluation Strategy

### Use Condition 01 (GT) For Primary Evaluation
GT isolates Cosmos reasoning errors from detection errors. Clean variable.

### Two-Layer Evaluation

**Layer 1 — Talk2BEV-Bench** (existing)
MCQ % correct across conditions A, B, D, C, E.
Primary thesis comparison table.

**Layer 2 — Grounded Spatial Accuracy (GSA)** (novel)
Auto-evaluable: Cosmos free-text answers checked against detections.json numbers.
```
Question: "How far is the nearest pedestrian directly behind?"
GT:       12.78m  (detections.json → pedestrian, zone=directly behind)
Cosmos:   "approximately 13 meters"
Score:    1.7% error → CORRECT (within 20% threshold)
```
Five sub-scores: distance, velocity, zone, TTC, occlusion accuracy.
No human evaluators needed. Fully programmatic. Novel contribution.

### Target QA JSON Format
```json
{
  "question": "...",
  "answer": "...",
  "question_type": "distance|velocity|zone|ttc|occlusion|safety|gap|comparative",
  "gt_verifiable": true,
  "gt_value": 12.78,
  "gt_field": "distance_m",
  "gt_object_index": 5,
  "condition": "B",
  "scene": "scene-0061"
}
```

---

## Infrastructure

- **Docker:** `ambarishgk007/moral-bevfusion-thesis`
- **AWS:** g4dn.xlarge, T4 GPU, Spot ~$0.16/hr
- **nuScenes:** `data/nuscenes/` v1.0-mini (10 scenes)
- **Python env:** `.venv` in `~/Desktop/thesi/forked/bevfusionV2/`

---

## Next Session (AWS) — Steps In Order

### 0. Commit everything first (laptop, 5 min)
```bash
cp ~/Downloads/scene_utils.py moral_pipeline/utils/scene_utils.py
git add moral_pipeline/
git commit -m "feat: final scene_utils - TTC fix, BEVFusion compat, BEV legend complete"
git push
```

### 1. Cosmos smoke test
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'nvidia/Cosmos-Reason2-8B', torch_dtype='auto', device_map='auto')
# Feed scene-0061 condition B: bev_map.png + scene_description.txt
# Question: "What is directly ahead of the ego vehicle and how far away?"
# Expected: truck, ~17m
```

### 2. Talk2BEV-Bench on conditions A, B, D → first real numbers

### 3. GSA scorer implementation
```python
def gsa_score(cosmos_answer, gt_value, question_type, threshold=0.20):
    # Extract number from Cosmos text answer
    # Compare to gt_value with threshold
    pass
```

### 4. BEVFusion conditions (03, 04) — run inference, apply same pipeline

---

## Thesis Novelty (5 Points For Supervisor)

1. BEVFusion BEV → Cosmos-Reason2 spatial QA — unpublished combination
2. Cosmos auto-generating nuScenes-grounded QA dataset — not done elsewhere
3. Radar Doppler velocity in VLM scene context (physically measured, not estimated) — novel
4. GSA metric: auto-evaluable spatial reasoning against metric 3D ground truth — new
5. Ablation: camera-only / BEV / BEV+radar with measurable score — not done with these models

---

*Next session: share MoRAL_Session_Prompt.md + this file. Say: "Help me set up Cosmos-Reason2 on AWS"*
