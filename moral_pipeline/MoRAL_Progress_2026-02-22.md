# MoRAL Pipeline — Session Progress Log
**Date:** February 22, 2026  
**Session focus:** Building all laptop-side data conditions (A, B, D) with rich scene descriptions for Cosmos QA generation  
**Status at end of session:** ✅ All laptop work complete. Ready for AWS + Cosmos.

---

## What Was Accomplished This Session

### 1. Condition 01 — GT Annotations (`outputs/01_gt_annotations/`) ✅
- 10 nuScenes mini scenes processed (all scenes in v1.0-mini)
- Each scene: 6 camera images + BEV map + detections.json + scene_description.txt + metadata.json
- `yaw_rad` added to all detections (matches BEVFusion output schema exactly)
- Script: `moral_pipeline/01_generate_gt_scenes/generate_gt_scenes.py`

### 2. Condition 02 — GT + Radar (`outputs/02_gt_with_radar/`) ✅
- Same 10 scenes, enriched with Doppler radar from all 5 nuScenes radar sensors
- **Bug fixed this session:** nuScenes radar PCD files are mixed-type binary, NOT uniform float32
  - Old (broken): `np.fromfile(path, dtype=np.float32)` → wrong byte layout, 0 radar points
  - Fixed: `_load_radar_pcd()` — parses PCD header, builds struct format dynamically
  - Key struct format for nuScenes radar: `<fffbhfffffbbbbbbbb` (18 fields, mixed sizes)
- Uses `vx_comp` / `vy_comp` columns (ego-motion compensated Doppler velocity)
- Match radius: 3.0m nearest-neighbour between radar points and GT detections
- Results: scene-0061 gets 56/66 detections radar-confirmed, 602 total radar points
- Radar-confirmed moving objects show close agreement with GT velocity:
  - car: GT=9.57 m/s, radar=7.96 m/s
  - car: GT=11.26 m/s, radar=11.12 m/s  
  - truck: GT=3.23 m/s, radar=2.38 m/s
  - pedestrian: GT=1.29 m/s, radar=1.15 m/s
- BEV map shows cyan radar dots + Doppler velocity arrows (distinct from condition 01)
- Saves `radar_points.npy` per scene (raw ego-frame radar for future use)
- Script: `moral_pipeline/02_generate_gt_radar_scenes/generate_gt_radar_scenes.py`

### 3. Condition 00 — Camera Only (`outputs/00_camera_only/`) ✅
- 10 scenes, 6 camera images each
- NO BEV map, NO detections.json, NO scene_description.txt
- This is the hardest baseline for Cosmos — camera-only spatial reasoning
- Script: `moral_pipeline/00_generate_camera_only/generate_camera_only.py`

### 4. Rich Scene Description Format — Major Upgrade ✅
The `make_scene_text()` function in `scene_utils.py` was completely rewritten.

**Old format (bad for QA generation):**
```
66 objects detected around the vehicle. barrier at 10.22m bearing -144.0deg 
(mostly_occluded). truck at 16.84m bearing 15.7deg. ...
```

**New format (rich, structured, enables diverse QA):**
```
SCENE OVERVIEW: 66 objects detected via LiDAR ground-truth annotations. 
27 are moving. 22 are partially or fully occluded.

EGO STATE: Travelling at 9.1 m/s (32.9 km/h). Estimated stopping distance 
at current speed: 10.4 m. Ego vehicle dimensions: 4.5 m long, 2.0 m wide.

NEAREST OBJECT PER ZONE: directly ahead: truck at 16.84 m; front-left: 
pedestrian at 18.08 m, moving at 1.7 m/s (estimated); ...

SAFETY CRITICAL: barrier rear-right 10.22 m away (TTC ≈ 1.1 s); car 
rear-right 20.74 m away at 9.6 m/s (TTC ≈ 1.1 s); ...

FULL OBJECT LIST:
  [1] barrier, zone=rear-right, distance=10.22m, bearing=-144.0deg, 
      size=1.9x0.6x1.1m (WxLxH), speed=stationary, visibility=fully_visible, 
      lidar_pts=77(strong), ttc≈1.1s.
  [14] truck, zone=directly ahead, distance=16.84m, ..., 
       lane_gap≈1.3m (passable).
  ...all 66 objects...

OBJECT COUNTS: 23 barriers, 1 bicycle, 1 bus, 8 cars, ...

OCCLUDED OBJECTS (visible to LiDAR but not clearly to cameras): 
barrier at 10.62m rear-right (mostly_occluded); ...
```

**What each section enables Cosmos to ask:**
- `SCENE OVERVIEW` → scene-level summary questions
- `EGO STATE` → physics questions (stopping distance, speed)
- `NEAREST OBJECT PER ZONE` → spatial/directional questions
- `SAFETY CRITICAL` → safety reasoning, TTC questions
- `FULL OBJECT LIST` → object-specific questions, comparative questions, gap/passability
- `OBJECT COUNTS` → counting questions, composition questions
- `OCCLUDED OBJECTS` → sensor fusion questions (why LiDAR sees what cameras miss)

**New computed fields per object:**
- `zone` — human-readable direction (directly ahead, front-left, rear-right, etc.)
- `speed` — radar-confirmed if available, gt-estimated otherwise
- `ttc` — time-to-collision estimate for objects in ego's path
- `lane_gap` — whether ego can pass beside the object (passable / tight)
- `lidar_pts(strength)` — evidence quality (strong >50, moderate >10, weak ≤10)
- `rcs` — radar cross-section in dBsm (when radar available)
- `visibility` — fully_visible / mostly_visible / partially_visible / mostly_occluded

---

## Current File Structure

```
moral_pipeline/
  00_generate_camera_only/
    generate_camera_only.py          ← Condition A script
  01_generate_gt_scenes/
    generate_gt_scenes.py            ← Condition B script
  02_generate_gt_radar_scenes/
    generate_gt_radar_scenes.py      ← Condition D script
  utils/
    scene_utils.py                   ← ALL shared utilities (updated this session)
  outputs/
    00_camera_only/                  ← Condition A: 10 scenes, cameras only
      scene-0061/ ... scene-1100/
        CAM_FRONT.jpg ... CAM_BACK_RIGHT.jpg
        metadata.json
    01_gt_annotations/               ← Condition B: 10 scenes, GT + BEV
      scene-0061/ ... scene-1100/
        bev_map.png
        CAM_FRONT.jpg ... CAM_BACK_RIGHT.jpg
        detections.json              ← source: "ground_truth"
        scene_description.txt        ← rich format, all objects
        metadata.json
    02_gt_with_radar/                ← Condition D: 10 scenes, GT + BEV + Radar
      scene-0061/ ... scene-1100/
        bev_map.png                  ← cyan radar dots + velocity arrows
        CAM_FRONT.jpg ... CAM_BACK_RIGHT.jpg
        detections.json              ← source: "ground_truth_radar", has radar_* fields
        scene_description.txt        ← "radar-confirmed X m/s" for moving objects
        radar_points.npy             ← raw ego-frame radar (N, 6)
        metadata.json                ← includes radar_summary stats
```

---

## Conditions Summary

| Folder | Condition | Source | BEV Map | Radar | Scene Text | GPU? |
|---|---|---|---|---|---|---|
| `00_camera_only` | A | — | ❌ | ❌ | ❌ | No |
| `01_gt_annotations` | B | GT annotations | ✅ | ❌ | ✅ rich | No |
| `02_gt_with_radar` | D | GT + radar | ✅ cyan | ✅ Doppler | ✅ rich | No |
| `03_bevfusion` | C | BEVFusion model | ✅ | ❌ | ✅ rich | **Yes — AWS** |
| `04_bevfusion_radar` | E | BEVFusion + radar | ✅ cyan | ✅ Doppler | ✅ rich | **Yes — AWS** |

---

## Key Technical Decisions Made This Session

### Why the PCD parser had to be rewritten
nuScenes radar `.pcd` files have a mixed-type binary layout:
```
SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
TYPE F F F I I F F F F F I I I I I I I I
```
Reading with `np.fromfile(dtype=float32)` gives the wrong byte count (2367 bytes ≠ 97 × 18 floats). The fix is `_load_radar_pcd()` which reads the header, builds `struct.unpack_from` format `<fffbhfffffbbbbbbbb`, and unpacks each point correctly.

### Why we use vx_comp / vy_comp not vx / vy
`vx_comp` and `vy_comp` are ego-motion-compensated. They represent the true velocity of the detected object. `vx` and `vy` include the ego vehicle's own motion and would give wrong velocities for all stationary objects.

### Why max_objects=None in make_scene_text
The old limit of 15 objects cut off most of the scene. For Cosmos QA generation we want ALL detections — scene-0061 has 66 objects and Cosmos should be able to ask about any of them. The full list is included.

### Why zone labels instead of bearing degrees
"directly ahead" is interpretable by a VLM. "bearing -144.0deg" is not. The 8-zone system (directly ahead, front-left, left, rear-left, directly behind, rear-right, right, front-right) gives Cosmos human-readable spatial context.

---

## What Was NOT Done (Next Session — AWS)

### Conditions 03 and 04 — BEVFusion predictions
These require running BEVFusion on the nuScenes data to get predicted detections instead of GT annotations. Needs GPU. Plan:
1. Launch AWS g4dn.xlarge spot instance (~$0.16/hr)
2. Pull Docker image: `ambarishgk007/moral-bevfusion-thesis`
3. Run BEVFusion inference on all 10 scenes
4. Extract predicted bounding boxes into detections.json format (source: "bevfusion")
5. Run `generate_gt_radar_scenes.py` logic with BEVFusion detections instead of GT

### Cosmos-Reason2 Integration
1. Install: `pip install transformers>=4.57.0 qwen-vl-utils`
2. Load model: `nvidia/Cosmos-Reason2-8B` (runs on T4)
3. Feed each scene: BEV map image + 6 camera images + scene_description.txt
4. Ask Talk2BEV-Bench questions for evaluation
5. Auto-generate QA pairs for fine-tuning dataset

### The QA Generation Prompt for Cosmos
When generating QA pairs, Cosmos should be prompted to generate questions across these categories:
- **Spatial:** "What is directly ahead of the ego vehicle?"
- **Distance:** "How far is the nearest pedestrian?"
- **Physics:** "Given current speed, can the ego vehicle stop before hitting the truck?"
- **Safety:** "Which object poses the most immediate collision risk?"
- **Occlusion:** "Which objects are visible to LiDAR but hidden from cameras?"
- **Comparative:** "Which of the two cars ahead is moving faster?"
- **Gap/passability:** "Is there enough room for the ego vehicle to pass beside the truck?"
- **TTC:** "Approximately how many seconds until the approaching car reaches the ego vehicle?"

Each QA pair should include:
- `question`: the question text
- `answer`: Cosmos's answer
- `condition`: which condition folder this came from (A/B/D/C/E)
- `scene`: scene name
- `question_type`: spatial/distance/physics/safety/occlusion/comparative/gap/ttc
- `gt_verifiable`: True if the answer can be checked against detections.json values

### LoRA Fine-tuning (Phase 2)
- Model: Qwen-VL (base)
- Training data: Cosmos-generated QA pairs scored against GT
- Hardware: A100 or lab cluster (T4 is NOT enough)
- Method: LoRA adapters only (~1% of parameters)
- Talk to supervisor about lab cluster access

---

## Infrastructure

- **Docker image:** `ambarishgk007/moral-bevfusion-thesis` (7.4GB on Docker Hub)
- **AWS:** EC2 g4dn.xlarge, NVIDIA T4 GPU, Spot ~$0.16/hr
- **BEVFusion checkpoint:** `bevfusion_WORKING.pth` (inside Docker image)
- **nuScenes data:** `data/nuscenes/` (v1.0-mini, 10 scenes)
- **Python env:** `.venv` in `~/Desktop/thesi/forked/bevfusionV2/`

---

## To Start Next Session

Use the MoRAL_Session_Prompt.md and say:

> **"Help me integrate Cosmos-Reason2"**

Then show this progress file. The next session will:
1. Set up Cosmos-Reason2 on AWS g4dn.xlarge
2. Run a smoke test on scene-0061 with condition B (GT + BEV)
3. Batch all 10 scenes across conditions A, B, D
4. Score answers against nuScenes GT detections
5. Get the first real evaluation numbers

---

## Open Questions for Supervisor

1. Is Phase 1 evaluation (no fine-tuning) sufficient for thesis contribution?
2. Lab cluster availability for LoRA fine-tuning?
3. Number of Cosmos QA pairs target: 500 scenes × 10 QA = 5000, or use full trainval?
4. Is Talk2BEV-Bench the right evaluation benchmark?
5. Should Isaac Sim evaluation from original proposal be included?
6. Publication target: IROS 2026, ITSC 2025, or arXiv preprint?
