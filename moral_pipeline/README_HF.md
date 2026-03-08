---
license: cc-by-nc-sa-4.0
task_categories:
  - visual-question-answering
  - image-to-text
language:
  - en
tags:
  - autonomous-driving
  - nuscenes
  - bev
  - lidar
  - radar
  - chain-of-thought
  - knowledge-distillation
  - driving-vlm
  - physical-reasoning
size_categories:
  - 1K<n<10K
---

# MoRAL Pipeline Outputs — 850 nuScenes Trainval Scenes

**MoRAL: Multi-modal Reasoning for Autonomous driving with LiDAR**

This dataset contains processed scene outputs from the nuScenes trainval split (850 scenes), 
generated as part of a master's thesis on sensor-enriched knowledge distillation for driving VLMs.

It is the intermediate representation used to generate training data for fine-tuning Qwen2.5-VL-7B 
using reasoning chains produced by Cosmos-Reason2-8B.

> ⚠️ **License note**: Raw sensor data is derived from nuScenes (CC BY-NC-SA 4.0). 
> This dataset is for non-commercial academic research only. 
> You must agree to the [nuScenes terms of use](https://www.nuscenes.org/terms-of-use) before using this dataset.

---

## What This Dataset Is

This is **not** the raw nuScenes data. It is a processed, structured representation designed 
for use as input to a VLM-based QA generation pipeline.

For each of 850 scenes, three conditions are provided:

| Condition | Folder | Description |
|-----------|--------|-------------|
| A | `condition_A_camera_only/` | Front camera image only — baseline, no sensor enrichment |
| B | `condition_B_lidar_bev/` | LiDAR BEV map + front camera + GT detections |
| D | `condition_D_lidar_bev_radar/` | LiDAR BEV map with radar Doppler overlay + front camera + GT detections |

The difference between B and D is that condition D's `bev_map.png` has **cyan arrows** overlaid 
on detected objects showing radar-confirmed Doppler velocity vectors.

---

## Thesis Context

### The Research Question

*Does fine-tuning a camera-based driving VLM on reasoning chains generated from LiDAR BEV 
and radar-enriched sensor data improve physical reasoning performance compared to camera-only training?*

### The Pipeline

```
nuScenes trainval (850 scenes)
        │
        ▼
MoRAL pipeline (this dataset)
  ├── Condition A: CAM_FRONT only
  ├── Condition B: BEV map (LiDAR) + CAM_FRONT + detections.json
  └── Condition D: BEV map (LiDAR + radar Doppler) + CAM_FRONT + detections.json
        │
        ▼
Cosmos-Reason2-8B (teacher model — generates reasoning chains)
  Questions per scene:
  spatial, safety, velocity, occlusion, gap, physics, 
  zone, planning, counterfactual, radar (D only)
        │
        ▼
ShareGPT JSONL training data
        │
        ▼
Fine-tune Qwen2.5-VL-7B (student model — LoRA, LLaMA-Factory)
        │
        ▼
Evaluate on DriveLM benchmark
```

### Key Design Decisions

**Why BEV image not raw LiDAR?**  
Qwen2.5-VL-7B is a vision-language model — it takes images as input. 
The BEV map encodes full spatial layout as a visual representation the student model 
can learn to interpret at inference time. Raw point clouds require a separate encoder.

**Why only CAM_FRONT?**  
Token budget. With 8192 context on a single L40S, BEV (1024 tokens) + CAM_FRONT (1024 tokens) 
+ GT JSON (~400 tokens) + question + system prompt = ~3500 tokens input, leaving ~4500 tokens 
for reasoning output. Adding more cameras would overflow context.

**Why condition A/B/D not A/B/C/D?**  
Condition C (LiDAR BEV without GT annotations) was considered but removed — the controlled 
experiment is whether radar adds value over LiDAR alone, so B vs D is the clean comparison.

**Why RADAR_FRONT only?**  
Spatial correspondence with CAM_FRONT. Side and rear radar sensors cover objects 
not relevant to the forward-facing driving questions asked. Multi-radar fusion is future work.

---

## Dataset Structure

```
moral-pipeline-outputs/
├── README.md                          ← this file
├── generate_cosmos_qa.py              ← QA generation script (run on cloud)
│
├── condition_A_camera_only/           ← Condition A: baseline
│   ├── run_config.json                ← generation parameters
│   ├── scene-0001/
│   │   ├── CAM_FRONT.jpg              ← front camera image
│   │   └── metadata.json             ← scene token, timestamp, location
│   ├── scene-0002/
│   └── ... (850 scenes total)
│
├── condition_B_lidar_bev/             ← Condition B: LiDAR BEV enriched
│   ├── run_config.json
│   ├── scene-0001/
│   │   ├── bev_map.png                ← bird's-eye-view from LiDAR (no radar)
│   │   ├── CAM_FRONT.jpg              ← front camera image
│   │   ├── detections.json            ← GT object detections (see schema below)
│   │   ├── scene_description.txt      ← natural language scene summary
│   │   └── metadata.json
│   └── ... (850 scenes total)
│
└── condition_D_lidar_bev_radar/       ← Condition D: LiDAR BEV + radar Doppler
    ├── run_config.json
    ├── scene-0001/
    │   ├── bev_map.png                ← BEV with cyan radar Doppler arrows overlaid
    │   ├── CAM_FRONT.jpg              ← front camera image (same as condition B)
    │   ├── detections.json            ← GT detections + radar_velocity_* fields
    │   ├── scene_description.txt      ← includes radar-confirmed velocity summary
    │   ├── radar_points.npy           ← raw radar points in ego frame (N, 6)
    │   └── metadata.json
    └── ... (850 scenes total)
```

---

## File Schemas

### detections.json

Array of detected objects, filtered to ≤50m from ego vehicle, sorted nearest-first.

```json
[
  {
    "class": "car",
    "distance_m": 8.3,
    "bearing_deg": 2.1,
    "velocity_ms": 0.0,
    "visibility": "full",
    "width_m": 1.9,
    "radar_velocity_ms": 0.1,
    "radar_velocity_confirmed": true,
    "radar_quality": "reliable"
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `class` | string | Object category (car, pedestrian, truck, bus, bicycle, motorcycle, barrier, traffic_cone) |
| `distance_m` | float | Euclidean distance from ego vehicle in metres |
| `bearing_deg` | float | Bearing in nuScenes ego frame (positive=left, negative=right, 0=directly ahead) |
| `velocity_ms` | float | GT object speed in m/s from nuScenes annotations |
| `visibility` | string | full / mostly_visible / partially_visible / mostly_occluded |
| `width_m` | float | Object width in metres |
| `radar_velocity_ms` | float | Radar-measured radial velocity in m/s (condition D only, else null) |
| `radar_velocity_confirmed` | bool | True if radar point matched to this detection within 3m radius |
| `radar_quality` | string | reliable / radial_ambiguous / low_rcs (condition D only, else null) |

### bev_map.png

- **Resolution**: ~2000×2000px
- **Range**: 50m radius around ego vehicle
- **Ego vehicle**: white rectangle at centre
- **Object colours**: green=car, red=pedestrian, blue=truck, orange=bus, purple=bicycle/motorcycle, grey=barrier/cone
- **Condition D only**: cyan arrows on radar-confirmed objects showing Doppler velocity direction and magnitude

### scene_description.txt

Natural language summary including: location, time of day, weather, ego speed, number of objects by class, notable hazards. Condition D additionally includes radar-confirmed velocity summary.

### radar_points.npy (condition D only)

NumPy array of shape (N, 6) in ego vehicle frame:
```
columns: [x, y, z, vx_compensated, vy_compensated, rcs]
units:   [m, m, m, m/s, m/s, dBsm]
```

---

## How to Replicate This Dataset

### Requirements

```bash
pip install nuscenes-devkit matplotlib numpy
```

### Step 1 — Download nuScenes trainval

Download from [nuscenes.org](https://www.nuscenes.org/nuscenes#download):
- `v1.0-trainval_meta.tgz`
- `v1.0-trainval01_blobs.tgz` through `v1.0-trainval10_blobs.tgz`

Extract all into a single directory:
```bash
tar -xzf v1.0-trainval_meta.tgz
tar -xzf v1.0-trainval01_blobs.tgz &
# ... repeat for all 10 blobs
wait
```

### Step 2 — Clone the pipeline

```bash
git clone https://github.com/AmbarishGK/YOUR_REPO
cd moral_pipeline
pip install -r requirements.txt
```

### Step 3 — Generate all three conditions

```bash
DATAROOT=/path/to/nuscenes

# Condition A — camera only
python3 00_generate_camera_only/generate_camera_only.py \
    --dataroot $DATAROOT \
    --version v1.0-trainval \
    --max-scenes -1 \
    --cameras CAM_FRONT \
    --out-dir outputs/00_camera_only

# Condition B — LiDAR BEV
python3 01_generate_gt_scenes/generate_gt_scenes.py \
    --dataroot $DATAROOT \
    --version v1.0-trainval \
    --max-scenes -1 \
    --cameras CAM_FRONT \
    --out-dir outputs/01_gt_annotations

# Condition D — LiDAR BEV + radar
python3 02_generate_gt_radar_scenes/generate_gt_radar_scenes.py \
    --dataroot $DATAROOT \
    --version v1.0-trainval \
    --max-scenes -1 \
    --cameras CAM_FRONT \
    --radar-sensors RADAR_FRONT \
    --out-dir outputs/02_gt_with_radar
```

### Step 4 — Generate QA training data (requires cloud GPU)

```bash
# Requires L40S 48GB or equivalent (Cosmos-Reason2-8B needs 48GB VRAM)
# Start vLLM server
python3 -m vllm.entrypoints.openai.api_server \
    --model nvidia/Cosmos-Reason2-8B \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --reasoning-parser qwen3 \
    --port 8000

# Generate QA pairs
python3 generate_cosmos_qa.py \
    --condition B \
    --scene-list all_scenes.txt \
    --api-mode local \
    --base-url http://localhost:8000/v1 \
    --pipeline-root /path/to/moral_pipeline

python3 generate_cosmos_qa.py \
    --condition D \
    --scene-list all_scenes.txt \
    --api-mode local \
    --base-url http://localhost:8000/v1 \
    --pipeline-root /path/to/moral_pipeline
```

### Step 5 — Fine-tune Qwen2.5-VL-7B

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# Register datasets in data/dataset_info.json
# Run training
python src/train.py configs/train_condition_B.yaml
python src/train.py configs/train_condition_D.yaml
```

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total scenes | 850 |
| Total samples (keyframes) | 34,149 |
| Locations | Boston Seaport, Singapore (Queenstown, One North, Holland Village) |
| Conditions | Day / Night / Rain |
| Avg detections per scene | ~16 objects within 50m |
| Radar-confirmed detections | 5,861 across all scenes |
| Condition A size | 138MB |
| Condition B size | 414MB |
| Condition D size | 428MB |
| Total size | ~980MB |

---

## Question Types Generated by Cosmos-Reason2-8B

| Question Type | Condition | GT Verifiable | Description |
|---------------|-----------|---------------|-------------|
| `spatial` | B, D | ✅ | Nearest object in vehicle path |
| `safety` | B, D | ✅ | Highest risk object + TTC calculation |
| `velocity` | B, D | ✅ | Moving objects + collision risk |
| `occlusion` | B, D | ❌ | Objects LiDAR sees but camera cannot |
| `gap` | B, D | ✅ | Lane change safety + lateral gap |
| `physics` | B, D | ✅ | Stopping distance vs object distance |
| `zone` | B, D | ✅ | Front-left vs front-right hazard comparison |
| `planning` | B, D | ✅ | Concrete action: BRAKE/MAINTAIN/YIELD/STOP |
| `counterfactual` | B, D | ✅ | Outcome if vehicle does nothing for 4 seconds |
| `radar` | D only | ✅ | What radar reveals that camera cannot |

---

## Reasoning Chain Format

Each Cosmos-generated answer follows this structure, designed to teach the student model 
to ground reasoning in visual inputs before confirming with numbers:

```
<think>
[BEV] Spatial description from bird's-eye-view map
[CAM] Visual description from front camera
[GT]  Exact values from detections.json + calculations (TTC, stopping distance, etc.)
[DECISION] Recommended action with physical justification
</think>

<answer>
Concise answer: situation, key numbers, action
</answer>
```

This format ensures the student model (Qwen) learns to:
1. Read spatial layout from BEV images
2. Confirm visually from camera
3. Compute physical quantities (TTC, stopping distance)
4. Make grounded driving decisions

---

## Citation

If you use this dataset or pipeline in your research, please cite:

```bibtex
@mastersthesis{moral2026,
  title     = {MoRAL: Sensor-Enriched Knowledge Distillation for Physical Reasoning in Driving VLMs},
  author    = {Ambarish},
  year      = {2026},
  school    = {SJSU},
  note      = {Dataset available at https://huggingface.co/datasets/AmbarishGK/moral-pipeline-outputs}
}
```

Also cite nuScenes:
```bibtex
@inproceedings{caesar2020nuscenes,
  title     = {nuScenes: A multimodal dataset for autonomous driving},
  author    = {Caesar, Holger and others},
  booktitle = {CVPR},
  year      = {2020}
}
```

---

## Contact

For questions about the pipeline or dataset, open an issue on the GitHub repository.
