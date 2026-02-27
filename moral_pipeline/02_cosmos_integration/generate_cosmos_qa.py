"""
moral_pipeline/02_cosmos_integration/generate_cosmos_qa.py
============================================================
MoRAL Pipeline — Cosmos Reason 2 QA Generation Script  [v3 — cloud ready]

PURPOSE:
  For each clean scene, sends BEV map + all camera images + full GT JSON
  to Cosmos Reason 2-8B and generates structured QA pairs with chain-of-thought.
  Saves output as ShareGPT-format JSONL for LLaMA-Factory fine-tuning.

CHANGES vs v1 (2B run):
  - Model: Cosmos-Reason2-8B (was 2B)
  - Inputs: BEV map + CAM_FRONT/FRONT_LEFT/FRONT_RIGHT/BACK* + full detections JSON
  - Condition D: radar BEV image also passed
  - System prompt: English-only enforcement + explicit format
  - Question templates: tightened to anchor model to GT data
  - max_tokens: 4096 (was 2048)
  - skip_existing: False by default (full regeneration)
  - Fix: </tool_call> typo in answer parser → </think>
  - ShareGPT: all camera paths stored in images list

USAGE:
  # Regenerate all scenes, condition B:
  python generate_cosmos_qa.py --condition B --api-mode local \\
    --base-url http://localhost:8000/v1 --pipeline-root /home/ubuntu

  # Regenerate all scenes, condition D:
  python generate_cosmos_qa.py --condition D --api-mode local \\
    --base-url http://localhost:8000/v1 --pipeline-root /home/ubuntu

  # Single scene smoke test:
  python generate_cosmos_qa.py --scene scene-0061 --condition B --api-mode local \\
    --base-url http://localhost:8000/v1 --pipeline-root /home/ubuntu

LOCAL SETUP (RTX 4090 24GB — 8192 context, CAM_FRONT only):
  python3 -m vllm.entrypoints.openai.api_server \
    --model nvidia/Cosmos-Reason2-8B \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --reasoning-parser qwen3 \
    --enforce-eager \
    --port 8000

CLOUD SETUP for full dataset (A100 80GB — 16384 context, 3 front cameras):
  python3 -m vllm.entrypoints.openai.api_server \
    --model nvidia/Cosmos-Reason2-8B \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --reasoning-parser qwen3 \
    --port 8000
  # Restore: CAMERA_FILENAMES = ["CAM_FRONT.jpg","CAM_FRONT_LEFT.jpg","CAM_FRONT_RIGHT.jpg"]
  # Restore: max_tokens = 8000

SCENE STATUS:
  CLEAN (9 scenes): 0061, 0553, 0655, 0757, 0796, 0916, 1077, 1094, 1100
  DROPPED: 0103 (annotation gaps)
"""

import os
import sys
import json
import base64
import argparse
import time
from pathlib import Path
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────

CLEAN_SCENES = [
    "scene-0061",
    "scene-0553",
    "scene-0655",
    "scene-0757",
    "scene-0796",
    "scene-0916",
    "scene-1077",
    "scene-1094",
    "scene-1100",
]

CONDITION_DIRS = {
    "A": "00_camera_only",
    "B": "01_gt_annotations",
    "D": "02_gt_with_radar",
}

# Camera images — CAM_FRONT only for 4090 (8192 context).
# BEV map encodes full spatial layout. Front camera covers the safety-critical forward view.
# FRONT_LEFT/RIGHT each cost ~1024 tokens and push input over context limit on 4090.
#
# Token budget (4090 / 8192 context):
#   System prompt:   ~450t  |  BEV map:     ~1024t  |  CAM_FRONT: ~1024t
#   GT JSON trimmed: ~400t  |  Scene desc:   ~200t  |  Question:   ~270t
#   Labels/overhead: ~150t  |  Total input: ~3518t  |  Output:    ~4624t
#
# To restore 3 front cameras (A100 80GB cloud, 16384 context), change to:
#   CAMERA_FILENAMES = ["CAM_FRONT.jpg", "CAM_FRONT_LEFT.jpg", "CAM_FRONT_RIGHT.jpg"]
#   and set --max-model-len 16384 on the vLLM server.
CAMERA_FILENAMES = [
    "CAM_FRONT.jpg",
]

# ── System prompt ──────────────────────────────────────────────────────────────
# Purpose: force Cosmos to use ALL input modalities — BEV map, front camera,
# AND GT JSON — so the reasoning chains it produces teach Qwen2.5-VL-7B to
# reason from both BEV and camera images together.
#
# The student (Qwen) trains on [BEV + front camera + question] → answer.
# So the teacher reasoning must demonstrate how to read BOTH images and tie
# them to physical numbers. If Cosmos only reads the GT JSON text, the
# training signal teaches Qwen to ignore its visual inputs.

COSMOS_SYSTEM_PROMPT = (
    "You are an expert autonomous driving perception and safety analyst. "
    "Respond ONLY in English.\n\n"
    "You are given three inputs for each question:\n"
    "  1. A BEV (bird's-eye-view) map rendered from LiDAR.\n"
    "  2. A front-facing camera image.\n"
    "  3. GROUND TRUTH DATA — structured sensor annotations.\n\n"

    "BEV MAP VISUAL LEGEND — memorise these before reading any BEV:\n"
    "  EGO VEHICLE: white/blue rectangle at the exact centre. "
    "A white arrow points forward (up = forward direction of travel).\n"
    "  OBJECT BOXES — colour encodes class:\n"
    "    Blue box       = car\n"
    "    Orange box     = truck\n"
    "    Red box        = bus\n"
    "    Purple box     = trailer\n"
    "    Yellow box     = motorcycle\n"
    "    Teal box       = bicycle\n"
    "    Green box      = pedestrian\n"
    "    Cyan box       = traffic cone  ← NOTE: cyan box = cone, NOT radar\n"
    "    Tan/brown box  = barrier\n"
    "  OBJECT OPACITY: solid fill = strong detection (≥10 LiDAR points); "
    "faint outline only = weak/distant detection.\n"
    "  RANGE RINGS: dashed circles at 10m, 20m, 30m, 40m, 50m from ego.\n\n"

    "VELOCITY ARROWS — two distinct types, do NOT confuse them:\n"
    "  COLOURED arrows (same colour as the object box): present in ALL conditions "
    "(B and D). These are GT velocity vectors computed from nuScenes annotations "
    "(position difference between consecutive keyframes). They show the object's "
    "estimated speed and direction but are NOT directly measured — they are derived "
    "from frame-to-frame position changes.\n"
    "  CYAN arrows (#00d4ff, bright cyan): present ONLY in condition D. "
    "These are radar Doppler velocity vectors — physically measured in a single frame "
    "via frequency shift. They appear on radar-confirmed objects only. "
    "A cyan ARROW on a moving object = radar Doppler. "
    "A cyan BOX = traffic cone (completely different — do not confuse).\n"
    "  CYAN DOTS (small, faint): raw radar return points in condition D. "
    "These show where radar echoes were detected, independent of GT detections.\n\n"

    "KEY DISTINCTION FOR REASONING:\n"
    "  Coloured arrow = estimated velocity (needs multiple frames, can lag)\n"
    "  Cyan arrow     = measured velocity (single frame, instantaneous, more reliable "
    "for fast-changing situations like sudden braking or merging)\n"
    "  When both appear on the same object, compare them — agreement confirms "
    "the velocity reading; large disagreement suggests the object changed speed "
    "between frames.\n\n"

    "STRICT REASONING ORDER — follow this for every answer:\n"
    "  Step 1 — READ THE BEV MAP: Describe the spatial layout. Identify object "
    "positions, classes (by box colour), and motion (by arrow colour and direction). "
    "In condition D, explicitly distinguish cyan Doppler arrows from coloured GT arrows "
    "and describe what each tells you about the object's velocity.\n"
    "  Step 2 — READ THE FRONT CAMERA: Describe what is visible ahead. "
    "Connect camera observations to BEV objects — which BEV boxes correspond to "
    "what you can see in the camera?\n"
    "  Step 3 — CONFIRM WITH GT DATA: Use exact numerical values from GROUND TRUTH "
    "DATA. Do all calculations here (TTC, stopping distance, lateral gap). "
    "Never invent numbers — every figure must come from GT data.\n"
    "  Step 4 — DRIVING DECISION: State a clear specific action grounded in "
    "the numbers from Step 3.\n\n"

    "Always respond in this exact format:\n"
    "<think>\n"
    "[BEV] What you observe in the BEV map — objects, positions, arrow types.\n"
    "[CAM] What you observe in the front camera image.\n"
    "[GT]  Exact values from GT data + calculations.\n"
    "[DECISION] Recommended action and physical justification.\n"
    "</think>\n\n"
    "<answer>\n"
    "Concise answer: situation, key number(s), recommended action.\n"
    "</answer>"
)

FORMAT_INSTRUCTION = ""  # Format is fully specified in system prompt

# ── Cosmos 8B inference parameters ────────────────────────────────────────────
# Cloud (L40S 48GB, 16384 context):
#   3 images ~3072t + text ~1200t = ~4272t input → 6000t output budget → total 10272t < 16384 ✅
# Local 4090 (8192 context):
#   Change max_tokens to 3500 AND CAMERA_FILENAMES to ["CAM_FRONT.jpg"] only
COSMOS_PARAMS = {
    "model":             "nvidia/Cosmos-Reason2-8B",  # 8B: stable, no language bleed
    "max_tokens":        3500,   # 4090 (8192 ctx): input ~4097t → 3500 output fits safely
                                 # Cloud L40S (16384 ctx): change to 6000
    "temperature":       0.6,
    "top_p":             0.95,
    "presence_penalty":  0.1,    # discourages repetition loops
    "frequency_penalty": 0.05,   # mild frequency penalty reduces looping further
    # Do NOT use Cosmos-Reason2-2B — produces language bleed (Chinese/Arabic mid-response)
    # and repetition loops that corrupt training data. Confirmed broken on RTX 4090.
}

# ── Question templates ─────────────────────────────────────────────────────────
# Designed to force Cosmos to use ALL THREE input modalities in sequence:
#   BEV map (spatial layout) → front camera (visual scene) → GT JSON (exact numbers)
#
# Each question uses a [BEV] → [CAM] → [GT] → [DECISION] structure.
# This produces reasoning chains where Cosmos describes what it SEES in each image
# before confirming with numbers — which is exactly the reasoning pattern Qwen
# must learn: look at BEV spatially, look at camera visually, reason physically.

QUESTION_TEMPLATES = {
    "spatial": (
        "Look at the BEV map and front camera image, then answer: "
        "What is the closest object directly in the vehicle's path?\n"
        "In your reasoning:\n"
        "1. [BEV] Describe what you see directly ahead of the ego vehicle in the BEV map — "
        "what objects are in the forward zone, how close do they appear, what shape/colour?\n"
        "2. [CAM] Describe what you can see ahead in the front camera — is there an object "
        "visible on the road, in a lane, on the pavement?\n"
        "3. [GT] From GROUND TRUTH DATA, find the object with the smallest distance_m where "
        "bearing_deg is between -22.5 and +22.5 (directly ahead; positive=left, negative=right "
        "in nuScenes ego frame). If none, use nearest with bearing between -90 and +90. "
        "State its exact class, distance_m, and bearing_deg.\n"
        "4. [DECISION] Is this object a hazard? Should the vehicle slow, stop, or maintain speed?"
    ),

    "safety": (
        "Look at the BEV map and front camera image, then answer: "
        "Which object poses the greatest immediate safety risk and what should the driver do?\n"
        "In your reasoning:\n"
        "1. [BEV] Scan the BEV map — which objects are closest to ego, which appear to be "
        "in motion, which are directly in the vehicle's path?\n"
        "2. [CAM] What does the front camera show — any visible threats, pedestrians, "
        "vehicles cutting in, objects blocking the lane ahead?\n"
        "3. [GT] For every moving object (velocity_ms > 0.5), compute TTC:\n"
        "   - Ahead (bearing -45 to +45): TTC = distance_m / (ego_speed + velocity_ms)\n"
        "   - Behind (|bearing| > 135): TTC = distance_m / (velocity_ms - ego_speed) "
        "only if velocity_ms > ego_speed\n"
        "   - Side: use proximity. Ego speed from EGO STATE. Show the TTC calculation explicitly.\n"
        "4. [DECISION] Name the highest-risk object, its TTC, and the exact required action "
        "(brake, yield, swerve, or maintain speed)."
    ),

    "velocity": (
        "Look at the BEV map and front camera image, then answer: "
        "Which objects in this scene are actively moving, and do any pose a collision risk?\n"
        "In your reasoning:\n"
        "1. [BEV] Look at the BEV map — which objects appear to be dynamic actors vs static "
        "obstacles? In condition D, describe any cyan Doppler velocity arrows you can see: "
        "which objects have them, what direction do they point, how long are they?\n"
        "2. [CAM] In the front camera, can you see objects in motion — "
        "vehicles, cyclists, pedestrians?\n"
        "3. [GT] From GROUND TRUTH DATA, list every object with velocity_ms > 0.5. "
        "For each: class, exact velocity_ms, approximate position relative to ego "
        "(derive from bearing_deg: positive=left, negative=right), and whether moving "
        "toward or away from ego. State the total count. Do not skip any.\n"
        "4. [DECISION] Are any moving objects on a collision course? What should the driver do?"
    ),

    "occlusion": (
        "Look at the BEV map and front camera image, then answer: "
        "Are there objects the LiDAR has detected that the camera cannot fully see, "
        "and what risk do they pose?\n"
        "In your reasoning:\n"
        "1. [BEV] The BEV shows ALL LiDAR-detected objects regardless of camera visibility. "
        "Describe any objects in the BEV that appear to be behind or around other objects — "
        "anything that might be hidden from the camera's view.\n"
        "2. [CAM] Look at the front camera — what areas of the scene are occluded by other "
        "vehicles, structures, or the edge of the field of view?\n"
        "3. [GT] From GROUND TRUTH DATA, list every object where visibility is "
        "'mostly_occluded' or 'partially_visible'. For each: class, distance_m, "
        "approximate zone (derive from bearing_deg), and the specific risk it poses.\n"
        "4. [DECISION] How should the driver account for these hidden objects? "
        "What defensive driving action is appropriate?"
    ),

    "gap": (
        "Look at the BEV map and front camera image, then answer: "
        "Is it safe to make a lane change to the right?\n"
        "In your reasoning:\n"
        "1. [BEV] Look at the right side of the ego vehicle in the BEV map. "
        "Describe what you see — is the right lane clear, are there close objects, "
        "how much visual space exists between ego and any right-side objects?\n"
        "2. [CAM] In the front camera, can you see any objects to the front-right "
        "that would affect a rightward lane change?\n"
        "3. [GT] From GROUND TRUTH DATA, identify all objects to the right of ego "
        "(negative bearing in nuScenes frame): "
        "front-right: -67.5 to -22.5; right: -112.5 to -67.5; rear-right: -157.5 to -112.5. "
        "For the nearest right-side object compute: "
        "lateral_gap = distance_m * sin(|bearing_deg|) - width_m/2 - 1.0 (ego half-width). "
        "Also compute TTC = distance_m / (ego_speed + velocity_ms) if moving. "
        "Safe threshold: lateral_gap > 1.5m AND TTC > 3s.\n"
        "4. [DECISION] Is the lane change safe? State the exact gap and TTC values."
    ),

    "physics": (
        "Look at the BEV map and front camera image, then answer: "
        "Can the vehicle stop in time for the nearest obstacle ahead, "
        "or is emergency braking required?\n"
        "In your reasoning:\n"
        "1. [BEV] Describe the nearest object directly ahead in the BEV map. "
        "How close does it appear? Is it stationary or moving?\n"
        "2. [CAM] In the front camera, describe what is directly ahead on the road. "
        "Does there appear to be enough space to stop?\n"
        "3. [GT] From GROUND TRUTH DATA, find the nearest object with bearing_deg "
        "between -22.5 and +22.5 (directly ahead). If none, use bearing between -45 and +45. "
        "From EGO STATE extract ego_speed_ms and stopping_distance_m. "
        "Verify: stopping_distance = speed² / (2 × 4.0 m/s²). "
        "Show all three numbers: ego speed, stopping distance, object distance.\n"
        "4. [DECISION] Can the vehicle stop in time? State clearly whether "
        "emergency braking is required and why."
    ),

    "zone": (
        "Look at the BEV map and front camera image, then answer: "
        "Which is more hazardous right now — the front-left zone or the front-right zone, "
        "and which specific object is the threat?\n"
        "In your reasoning:\n"
        "1. [BEV] Compare the front-left and front-right areas of the BEV map. "
        "Describe what objects you can see in each zone — positions, sizes, and "
        "in condition D any Doppler arrows.\n"
        "2. [CAM] In the front camera, can you see objects to the left or right of "
        "centre in the forward view?\n"
        "3. [GT] From GROUND TRUTH DATA, derive zone from bearing_deg: "
        "front-left: +22.5 to +67.5; front-right: -67.5 to -22.5. "
        "For each object in these zones: class, distance_m, velocity_ms, visibility. "
        "Compute TTC = distance_m / (ego_speed + velocity_ms) for moving objects; "
        "use distance_m as proxy for stationary. Ego speed from EGO STATE.\n"
        "4. [DECISION] Which zone has lower minimum TTC (higher risk)? "
        "Name the specific object and state the required action."
    ),

    "radar": (
        "Look at the BEV map — you will see cyan arrows on some objects. "
        "These are radar Doppler velocity vectors. Answer: what do the radar measurements "
        "reveal about moving objects that the camera alone cannot determine?\n"
        "In your reasoning:\n"
        "1. [BEV] Describe the cyan Doppler arrows visible in the BEV map. "
        "Which objects have them? What direction do they point relative to ego? "
        "Which are long (fast-moving) vs short (slow)? Any pointing toward ego?\n"
        "2. [CAM] For the objects with radar vectors in the BEV, can you see them "
        "in the front camera? Do they look like they are moving?\n"
        "3. [GT] From GROUND TRUTH DATA, find every object where "
        "radar_velocity_confirmed is true. For each: class, distance_m, "
        "GT velocity_ms, radar_velocity_ms, radar_quality. "
        "Compute agreement: |radar_velocity_ms - velocity_ms| < 0.5 m/s = good. "
        "Note: radar_quality='reliable' = trustworthy; "
        "'radial_ambiguous' = pedestrian/cyclist (radial component only).\n"
        "4. [DECISION] For the fastest-closing radar-confirmed object: compute its TTC "
        "using radar_velocity_ms. Would a camera-only system have known this velocity "
        "without multiple frames? What action is required?"
    ),

    # ── Planning ───────────────────────────────────────────────────────────────
    # What concrete action should the vehicle take RIGHT NOW?
    # GT-verifiable: recommended action (BRAKE/MAINTAIN etc) can be checked
    # against rule-based oracle from TTC + stopping distance.
    # Works entirely from existing detections.json + ego_speed_ms.
    "planning": (
        "Look at the BEV map and front camera image, then answer: "
        "What is the single most important driving action the ego vehicle should "
        "take RIGHT NOW, and what specific physical evidence justifies it?\n"
        "In your reasoning:\n"
        "1. [BEV] Survey the full BEV map. Identify the single highest-priority "
        "object — the one that most constrains what the vehicle can safely do next. "
        "Describe its position, apparent motion, and proximity to ego.\n"
        "2. [CAM] Confirm this priority object in the front camera, or explain why "
        "it may not be visible (occluded, outside FOV, behind ego). "
        "What does the camera add about road context — intersection, lane, signal?\n"
        "3. [GT] From GROUND TRUTH DATA compute for every object within 30m "
        "(bearing -90 to +90):\n"
        "   TTC = distance_m / max(ego_speed_ms + velocity_ms, 0.1)\n"
        "   stopping_dist = ego_speed_ms^2 / (2 x 4.0)\n"
        "   emergency_dist = ego_speed_ms^2 / (2 x 8.0)\n"
        "   Find the object with lowest TTC. Show all calculations explicitly.\n"
        "   Ego speed from EGO STATE.\n"
        "4. [DECISION] State exactly ONE action and justify with numbers:\n"
        "   EMERGENCY_BRAKE — TTC < 2.0s OR distance < emergency_dist\n"
        "   BRAKE           — TTC < 4.0s OR distance < stopping_dist\n"
        "   YIELD           — TTC 4-6s, moving object nearby\n"
        "   MAINTAIN        — TTC > 6.0s, no immediate hazard\n"
        "   STOP            — intersection or crosswalk with pedestrians present\n"
        "   Format your decision as: ACTION: [action]. REASON: [object] at "
        "[distance]m, TTC=[value]s, [justification]."
    ),

    # ── Counterfactual ─────────────────────────────────────────────────────────
    # What happens if the vehicle does NOTHING for 4 seconds?
    # Forces causal reasoning about consequences, not just scene description.
    # This is what separates planning from perception — the model must reason
    # about outcomes, not just identify objects.
    # Works entirely from existing detections.json + ego_speed_ms.
    "counterfactual": (
        "Look at the BEV map and front camera image, then answer: "
        "What would happen if the ego vehicle maintained its current speed and "
        "heading for the next 4 seconds with NO braking or steering?\n"
        "In your reasoning:\n"
        "1. [BEV] Identify every object in the forward path (bearing -45 to +45). "
        "Describe which objects the vehicle would geometrically reach first "
        "if it continued straight at current speed.\n"
        "2. [CAM] What does the front camera show directly ahead — is the road "
        "clear, is there a stopped vehicle, pedestrian, or intersection? "
        "Would these be unavoidable at current speed?\n"
        "3. [GT] Using ego_speed_ms from EGO STATE, compute projected position "
        "at t=1, t=2, t=3, t=4 seconds: projected_dist = ego_speed_ms x t. "
        "For each object ahead (bearing -45 to +45) check whether "
        "projected_dist reaches distance_m within 4 seconds. "
        "Show each check explicitly: 'At t=Xs vehicle reaches Ym. "
        "[Object] is at Zm — [REACHED / not reached].'\n"
        "4. [DECISION] Describe the specific outcome if no action is taken: "
        "name the object hit or nearly missed, time to impact, and severity "
        "(COLLISION / NEAR-MISS / SAFE PASSAGE). "
        "Then state the minimum action that prevents the worst outcome and "
        "quantify it: e.g. braking at 4 m/s^2 gives X extra metres."
    ),
}


# ── DriveLM Overlap Detection ──────────────────────────────────────────────────

def check_drivelm_overlap(
    pipeline_root: Path,
    nuscenes_dataroot: str = "data/nuscenes",
    drivelm_json: str = "DriveLM/v1_1_train_nus.json",
) -> dict:
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        print("ERROR: nuscenes-devkit not installed. Run: pip install nuscenes-devkit")
        return {}

    print("Loading nuScenes mini...")
    nusc = NuScenes(version="v1.0-mini", dataroot=nuscenes_dataroot, verbose=False)

    scene_tokens: dict[str, list[str]] = {}
    all_tokens: set[str] = set()
    for scene in nusc.scene:
        name = scene["name"]
        tokens = []
        tok = scene["first_sample_token"]
        while tok:
            tokens.append(tok)
            all_tokens.add(tok)
            tok = nusc.get("sample", tok)["next"]
        scene_tokens[name] = tokens

    print(f"Total mini sample tokens: {len(all_tokens)}")

    drivelm_path = pipeline_root.parent / drivelm_json
    if not drivelm_path.exists():
        print(f"WARNING: DriveLM not found at {drivelm_path}")
        return {}

    print(f"Loading DriveLM from {drivelm_path}...")
    with open(drivelm_path) as f:
        drivelm = json.load(f)

    drivelm_tokens = set(drivelm.keys())
    overlap_by_scene: dict[str, list[str]] = {}
    total_overlap = 0
    for scene_name, tokens in scene_tokens.items():
        overlapping = [t for t in tokens if t in drivelm_tokens]
        if overlapping:
            overlap_by_scene[scene_name] = overlapping
            total_overlap += len(overlapping)

    print(f"Overlap: {total_overlap} tokens across {len(overlap_by_scene)} scenes")

    out_path = pipeline_root / "02_cosmos_integration" / "drivelm_overlap.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(overlap_by_scene, f, indent=2)
    print(f"Saved → {out_path}")
    return overlap_by_scene


# ── Image Loading ──────────────────────────────────────────────────────────────

def load_image_b64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    return "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"


# ── Cosmos API Call ────────────────────────────────────────────────────────────

def call_cosmos(
    bev_image_b64: str,
    scene_description: str,
    question: str,
    detections_json: str = "",
    extra_images: Optional[list[dict]] = None,
    api_mode: str = "local",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """
    Call Cosmos Reason 2-8B with:
      - BEV map image (always first)
      - Radar BEV image if condition D (second, if provided in extra_images)
      - Camera images (CAM_FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT)
      - scene_description.txt text
      - Full detections.json as structured GT data
      - Question

    extra_images: list of {"b64": str, "media_type": str, "label": str}
    """
    if dry_run:
        return {
            "reasoning": "[DRY RUN - no API call made]",
            "answer": "[DRY RUN]",
            "raw_response": "[DRY RUN]",
            "tokens_used": 0,
        }

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    if api_mode == "nim":
        key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise ValueError("Set NVIDIA_API_KEY env var or pass --api-key.")
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=key,
        )
    elif api_mode == "local":
        url = base_url or os.environ.get("COSMOS_BASE_URL", "http://localhost:8000/v1")
        client = OpenAI(base_url=url, api_key="not-needed")
    else:
        raise ValueError(f"Unknown api_mode: {api_mode}. Use 'nim' or 'local'")

    # ── Build user message content ──────────────────────────────────────────
    # Image order: BEV map → radar BEV (if present) → camera images
    # Each image preceded by a text label so the model knows what it's seeing.

    user_content = []

    # BEV map (always first)
    user_content.append({
        "type": "text",
        "text": "[IMAGE 1: BEV MAP — bird's-eye view from LiDAR, ego vehicle at centre]",
    })
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{bev_image_b64}"},
    })

    # Extra images: radar BEV and/or camera images
    if extra_images:
        for idx, img in enumerate(extra_images, start=2):
            user_content.append({
                "type": "text",
                "text": f"[IMAGE {idx}: {img['label']}]",
            })
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img['media_type']};base64,{img['b64']}"
                },
            })

    # Text block: scene description + GT JSON + question
    # ego_speed_ms is extracted from scene_description and added explicitly
    # so safety/physics questions can reference it without parsing prose
    import re as _re
    ego_speed_match = _re.search(r"ego speed[^:]*:\s*([\d.]+)", scene_description, _re.IGNORECASE)
    ego_speed_str = f"\nEGO STATE: ego_speed_ms={ego_speed_match.group(1)}" if ego_speed_match else ""

    gt_block = (
        f"\n\nGROUND TRUTH DATA (detections.json — use these exact values):{ego_speed_str}\n"
        f"{detections_json}"
        if detections_json else ""
    )

    user_content.append({
        "type": "text",
        "text": (
            f"SCENE SENSOR DATA:\n{scene_description}"
            f"{gt_block}\n\n"
            f"QUESTION: {question}\n\n"
            f"IMPORTANT: Respond in English only. "
            f"Use the exact numerical values from GROUND TRUTH DATA above. "
            f"ego_speed_ms is in EGO STATE above."
        ),
    })

    # Retry up to 3 times on transient errors (server restart, timeout, etc.)
    max_retries = 3
    last_error  = None
    response    = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": COSMOS_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                **COSMOS_PARAMS,
            )
            break  # success
        except Exception as e:
            last_error = e
            err_str = str(e)
            # Context overflow — not retryable, raise immediately
            if "input_tokens" in err_str or "context length" in err_str or "decoder prompt" in err_str:
                raise
            # Transient error — wait and retry
            wait = 10 * (attempt + 1)
            print(f"\n    [attempt {attempt+1}/{max_retries}] error: {err_str[:120]} — retrying in {wait}s")
            time.sleep(wait)

    if response is None:
        raise RuntimeError(f"All {max_retries} attempts failed. Last error: {last_error}")

    raw = response.choices[0].message.content or ""

    # Guard: if model returned empty or only whitespace, flag it clearly
    # so the QA pair is marked as failed rather than silently storing empty strings
    if not raw.strip():
        return {
            "reasoning":    "[ERROR: model returned empty response]",
            "answer":       "[ERROR: empty response]",
            "raw_response": raw,
            "tokens_used":  response.usage.total_tokens if response.usage else 0,
        }

    # ── Parse <think> and <answer> blocks ──────────────────────────────────
    reasoning = ""
    answer = raw

    if "<think>" in raw and "</think>" in raw:
        think_start = raw.index("<think>") + len("<think>")
        think_end   = raw.index("</think>")
        reasoning   = raw[think_start:think_end].strip()
        after_think = raw[think_end + len("</think>"):]
        if "<answer>" in after_think and "</answer>" in after_think:
            ans_start = after_think.index("<answer>") + len("<answer>")
            ans_end   = after_think.index("</answer>")
            answer    = after_think[ans_start:ans_end].strip()
        else:
            # Fallback: everything after </think>, strip any leftover tags
            answer = after_think.strip().lstrip("<answer>").rstrip("</answer>").strip()
    elif "<answer>" in raw and "</answer>" in raw:
        # No <think> block — just extract answer
        ans_start = raw.index("<answer>") + len("<answer>")
        ans_end   = raw.index("</answer>")
        answer    = raw[ans_start:ans_end].strip()

    return {
        "reasoning":    reasoning,
        "answer":       answer,
        "raw_response": raw,
        "tokens_used":  response.usage.total_tokens if response.usage else 0,
    }


# ── GT Verification Fields ─────────────────────────────────────────────────────

def extract_gt_fields(detections: list[dict], question_type: str) -> dict:
    gt = {"gt_verifiable": False, "gt_value": None, "gt_field": None}

    if not detections:
        return gt

    by_dist = sorted(detections, key=lambda d: d["distance_m"])

    if question_type == "spatial":
        # Directly ahead zone: bearing_deg within ±22.5° (matches ZONE_LABELS in scene_utils.py)
        ahead = [d for d in by_dist if abs(d.get("bearing_deg", 90)) <= 22.5]
        if ahead:
            obj = ahead[0]
            gt = {
                "gt_verifiable": True,
                "gt_value":      obj["distance_m"],
                "gt_field":      "distance_m",
                "gt_object_class": obj["class"],
                "gt_bearing_deg":  obj.get("bearing_deg"),
            }

    elif question_type == "safety":
        closing = [
            d for d in by_dist
            if d.get("velocity_ms", 0) > 0.5 and d["distance_m"] < 30
        ]
        if closing:
            obj = closing[0]
            ttc = obj["distance_m"] / max(obj["velocity_ms"], 0.1)
            gt = {
                "gt_verifiable": True,
                "gt_value":      round(ttc, 2),
                "gt_field":      "ttc_s",
                "gt_object_class": obj["class"],
                "gt_distance_m":   obj["distance_m"],
            }

    elif question_type == "velocity":
        moving = [d for d in detections if d.get("velocity_ms", 0) > 0.5]
        if moving:
            fastest = max(moving, key=lambda d: d["velocity_ms"])
            gt = {
                "gt_verifiable":  True,
                "gt_value":       fastest["velocity_ms"],
                "gt_field":       "velocity_ms",
                "gt_object_class": fastest["class"],
                "gt_moving_count": len(moving),
            }

    elif question_type == "physics":
        ahead = [d for d in by_dist if abs(d.get("bearing_deg", 90)) <= 22.5]
        if ahead:
            obj = ahead[0]
            gt = {
                "gt_verifiable":   True,
                "gt_value":        obj["distance_m"],
                "gt_field":        "nearest_ahead_m",
                "gt_object_class": obj["class"],
            }

    elif question_type == "planning":
        # GT action: rule-based oracle from TTC + stopping distance.
        # This gives an evaluable ground truth action for the planning question —
        # we can check if the model recommended the correct action class.
        # ego_speed stored separately; use median estimate if not available.
        EGO_SPEED_FALLBACK = 4.0  # m/s — typical urban speed
        ahead_30 = [
            d for d in by_dist
            if abs(d.get("bearing_deg", 90)) <= 90
            and d["distance_m"] <= 30
        ]
        if ahead_30:
            # Find minimum TTC object
            def ttc(d):
                return d["distance_m"] / max(EGO_SPEED_FALLBACK + d.get("velocity_ms", 0), 0.1)
            nearest_ttc_obj = min(ahead_30, key=ttc)
            min_ttc = round(ttc(nearest_ttc_obj), 2)
            stop_dist = round(EGO_SPEED_FALLBACK**2 / (2 * 4.0), 2)
            emg_dist  = round(EGO_SPEED_FALLBACK**2 / (2 * 8.0), 2)

            # Rule-based oracle action
            dist = nearest_ttc_obj["distance_m"]
            if min_ttc < 2.0 or dist < emg_dist:
                gt_action = "EMERGENCY_BRAKE"
            elif min_ttc < 4.0 or dist < stop_dist:
                gt_action = "BRAKE"
            elif min_ttc < 6.0:
                gt_action = "YIELD"
            else:
                gt_action = "MAINTAIN"

            gt = {
                "gt_verifiable":   True,
                "gt_value":        min_ttc,
                "gt_field":        "min_ttc_s",
                "gt_object_class": nearest_ttc_obj["class"],
                "gt_action":       gt_action,
                "gt_distance_m":   dist,
            }

    elif question_type == "counterfactual":
        # GT outcome: does straight travel at current speed hit anything in 4s?
        # If yes: COLLISION with that object. If no: SAFE PASSAGE.
        EGO_SPEED_FALLBACK = 4.0
        ahead_path = [
            d for d in by_dist
            if abs(d.get("bearing_deg", 90)) <= 45
        ]
        if ahead_path:
            nearest_ahead = ahead_path[0]
            dist = nearest_ahead["distance_m"]
            t_impact = dist / max(EGO_SPEED_FALLBACK, 0.1)
            outcome = "COLLISION" if t_impact <= 4.0 else "SAFE_PASSAGE"
            gt = {
                "gt_verifiable":   True,
                "gt_value":        round(t_impact, 2),
                "gt_field":        "t_impact_s",
                "gt_object_class": nearest_ahead["class"],
                "gt_outcome":      outcome,
                "gt_distance_m":   dist,
            }
        else:
            # No objects directly ahead — safe passage
            gt = {
                "gt_verifiable": True,
                "gt_value":      None,
                "gt_field":      "t_impact_s",
                "gt_outcome":    "SAFE_PASSAGE",
            }

    return gt


# ── Scene QA Generation ────────────────────────────────────────────────────────

def generate_scene_qa(
    scene_name: str,
    condition: str,
    pipeline_root: Path,
    api_mode: str = "local",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    dry_run: bool = False,
    skip_existing: bool = False,  # False = always regenerate (8B run)
) -> list[dict]:
    condition_dir = CONDITION_DIRS[condition]
    scene_dir     = pipeline_root / "outputs" / condition_dir / scene_name

    if not scene_dir.exists():
        print(f"  WARNING: {scene_dir} not found, skipping")
        return []

    out_dir = pipeline_root / "02_cosmos_integration" / "cosmos_qa" / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_out = out_dir / f"qa_pairs_condition{condition}.json"

    if skip_existing and qa_out.exists():
        print(f"  Skipping {scene_name} condition {condition} — already exists")
        with open(qa_out) as f:
            return json.load(f)

    # ── Load BEV map ───────────────────────────────────────────────────────
    bev_path = scene_dir / "bev_map.png"
    if not bev_path.exists():
        print(f"  WARNING: No bev_map.png in {scene_dir}, skipping")
        return []

    print(f"  Loading BEV map:  {bev_path}")
    bev_b64 = load_image_b64(bev_path)

    # ── Load scene description ─────────────────────────────────────────────
    desc_path        = scene_dir / "scene_description.txt"
    scene_description = desc_path.read_text() if desc_path.exists() else ""

    # ── Load detections JSON (full GT) ─────────────────────────────────────
    det_path   = scene_dir / "detections.json"
    detections = []
    detections_json_str = ""
    if det_path.exists():
        with open(det_path) as f:
            detections = json.load(f)
        # Trim to only fields needed for QA questions.
        # Dropping: x, y, z, yaw_rad, ann_token, num_radar_pts, radar_vx, radar_vy,
        #           radar_rcs, source, confidence, height_m
        # Keeping: all fields referenced in any question template.
        # Keep only the 6 fields actually referenced in question templates.
        # Filter to <=50m — objects beyond that are irrelevant to all question types
        # and are still visible on the BEV map for spatial context.
        # Result: ~15-20 objects x 6 fields = ~350-450 tokens (was 8500+ tokens for 66 objects).
        # Fields dropped: x, y, z, yaw_rad, vx, vy, ann_token, source, confidence,
        #                 height_m, num_lidar_pts, num_radar_pts, radar_vx, radar_vy, radar_rcs
        KEEP_FIELDS = {
            "class", "distance_m", "bearing_deg",
            "velocity_ms", "visibility", "width_m",
            "radar_velocity_ms", "radar_velocity_confirmed", "radar_quality",
        }
        trimmed = [
            {k: v for k, v in d.items() if k in KEEP_FIELDS}
            for d in detections
            if d.get("distance_m", 999) <= 50.0
        ]
        # Sort nearest-first so model sees the most relevant objects immediately
        trimmed = sorted(trimmed, key=lambda x: x.get("distance_m", 999))
        detections_json_str = json.dumps(trimmed, separators=(",", ":"))

    # ── Build extra images list ────────────────────────────────────────────
    # Order: radar BEV (condition D only) → camera images (all that exist)
    extra_images: list[dict] = []

    # Note: condition D bev_map.png already contains radar Doppler overlay
    # (rendered by 02_generate_gt_radar_scenes). No separate radar image needed.

    for cam_filename in CAMERA_FILENAMES:
        cam_path = scene_dir / cam_filename
        if cam_path.exists():
            cam_label = cam_filename.replace(".jpg", "").replace(".jpeg", "")
            extra_images.append({
                "b64":        load_image_b64(cam_path),
                "media_type": get_image_media_type(cam_path),
                "label":      f"CAMERA IMAGE — {cam_label}",
            })

    n_images = 1 + len(extra_images)
    print(f"  Total images per call: {n_images} "
          f"(BEV + {len(extra_images)} extra)")

    # ── Generate QA pairs ──────────────────────────────────────────────────
    qa_pairs    = []
    total_tokens = 0

    # ── Parallel question generation ──────────────────────────────────────
    # Send all questions for this scene simultaneously using a thread pool.
    # vLLM handles concurrent requests with continuous batching — this gives
    # ~1.5-2x real speedup vs sequential because prefill passes are batched
    # and decode is overlapped across sequences.
    # ThreadPoolExecutor is safe here because call_cosmos is I/O bound (HTTP).
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Build list of (q_type, question) pairs to run
    active_questions = [
        (q_type, question)
        for q_type, question in QUESTION_TEMPLATES.items()
        if not (q_type == "radar" and condition not in ("D",))
    ]
    skipped = [q for q, _ in QUESTION_TEMPLATES.items()
               if q == "radar" and condition not in ("D",)]
    for q in skipped:
        print(f"    [radar] SKIPPED (condition D only)")

    def run_one_question(q_type_question):
        q_type, question = q_type_question
        t0 = time.time()
        try:
            result = call_cosmos(
                bev_image_b64=bev_b64,
                scene_description=scene_description,
                question=question,
                detections_json=detections_json_str,
                extra_images=extra_images,
                api_mode=api_mode,
                base_url=base_url,
                api_key=api_key,
                dry_run=dry_run,
            )
        except Exception as e:
            err_msg = str(e)
            print(f"\n    [{q_type_question[0]}] EXCEPTION: {err_msg[:200]}")
            result = {
                "reasoning":    f"[ERROR: {err_msg}]",
                "answer":       "[ERROR]",
                "raw_response": "",
                "tokens_used":  0,
            }
        elapsed = time.time() - t0
        return q_type, question, result, elapsed

    # Max workers = number of questions per scene (9 or 10).
    # vLLM continuous batching handles all concurrent requests efficiently.
    max_workers = len(active_questions)
    results_map = {}
    t_batch_start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one_question, qt): qt[0]
            for qt in active_questions
        }
        for future in as_completed(futures):
            q_type, question, result, elapsed = future.result()
            reasoning = result["reasoning"]
            answer    = result["answer"]
            is_error  = answer.startswith("[ERROR") or not answer.strip()
            has_bev   = "[BEV]" in reasoning
            has_cam   = "[CAM]" in reasoning
            has_gt    = "[GT]" in reasoning
            has_dec   = "[DECISION]" in reasoning
            quality   = "✅" if (has_bev and has_cam and has_gt and has_dec and not is_error) else "⚠️ "
            missing   = [t for t, ok in [("[BEV]",has_bev),("[CAM]",has_cam),
                          ("[GT]",has_gt),("[DECISION]",has_dec)] if not ok]
            if missing and not is_error:
                print(f"    [{q_type}] {quality} missing: {missing} ({elapsed:.1f}s)")
            else:
                print(f"    [{q_type}] done ({elapsed:.1f}s, {result['tokens_used']} tok) {quality}")
            results_map[q_type] = (question, result, elapsed)
            total_tokens += result["tokens_used"]

    t_batch_total = time.time() - t_batch_start
    print(f"  All {len(active_questions)} questions done in {t_batch_total:.1f}s "
          f"(sequential would be ~{sum(r[2] for r in results_map.values()):.1f}s)")

    # Reassemble qa_pairs in original question order
    for q_type, _ in QUESTION_TEMPLATES.items():
        if q_type not in results_map:
            continue
        question, result, elapsed = results_map[q_type]
        reasoning = result["reasoning"]
        answer    = result["answer"]
        is_error  = answer.startswith("[ERROR") or not answer.strip()
        has_bev   = "[BEV]" in reasoning
        has_cam   = "[CAM]" in reasoning
        has_gt    = "[GT]" in reasoning
        has_dec   = "[DECISION]" in reasoning

        gt_fields = extract_gt_fields(detections, q_type)

        qa_pair = {
            "scene":         scene_name,
            "condition":     condition,
            "question_type": q_type,
            "question":      question,
            "reasoning":     reasoning,
            "answer":        answer,
            "quality_ok":    bool(has_bev and has_cam and has_gt and has_dec and not is_error),
            **gt_fields,
        }
        qa_pairs.append(qa_pair)

    # ── Save outputs ───────────────────────────────────────────────────────
    with open(qa_out, "w") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(qa_pairs)} QA pairs → {qa_out} ({total_tokens} total tokens)")

    sharegpt_out = out_dir / f"qa_pairs_condition{condition}_sharegpt.jsonl"
    save_sharegpt(qa_pairs, scene_dir, sharegpt_out, condition)

    return qa_pairs


# ── ShareGPT Format ────────────────────────────────────────────────────────────

def save_sharegpt(
    qa_pairs: list[dict],
    scene_dir: Path,
    out_path: Path,
    condition: str,
) -> None:
    """
    Save QA pairs in ShareGPT JSONL format for LLaMA-Factory.
    Images list: BEV map first, then all camera images that exist on disk.
    For condition D, radar BEV is second.
    """
    # Build ordered image path list (same order as was passed to the model)
    image_paths = [str(scene_dir / "bev_map.png")]

    # condition D bev_map.png already has radar overlay — no separate file needed
    for cam_filename in CAMERA_FILENAMES:
        p = scene_dir / cam_filename
        if p.exists():
            image_paths.append(str(p))

    n_images = len(image_paths)

    lines = []
    for qa in qa_pairs:
        # User turn: one <image> token per image, then the question
        image_tokens = "".join(["<image>\n"] * n_images)
        user_content = f"{image_tokens}{qa['question']}"

        # Assistant turn: full CoT + answer
        if qa["reasoning"]:
            assistant_content = (
                f"<think>\n{qa['reasoning']}\n</think>\n\n"
                f"<answer>\n{qa['answer']}\n</answer>"
            )
        else:
            assistant_content = qa["answer"]

        record = {
            "messages": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": image_paths,
            "_meta": {
                "scene":         qa["scene"],
                "condition":     qa["condition"],
                "question_type": qa["question_type"],
                "gt_verifiable": qa.get("gt_verifiable", False),
                "gt_value":      qa.get("gt_value"),
                "gt_field":      qa.get("gt_field"),
            },
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Saved ShareGPT JSONL → {out_path}  ({n_images} images/record)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Cosmos Reason 2-8B QA pairs for MoRAL pipeline"
    )
    parser.add_argument("--scene",     default=None,
        help="Single scene to process (default: all 9 clean scenes)")
    parser.add_argument("--condition", choices=["A", "B", "D"], default="B",
        help="Data condition: B=gt_annotations, D=gt_with_radar (default: B)")
    parser.add_argument("--api-mode",  choices=["nim", "local"], default="local",
        help="API mode: nim=NVIDIA cloud, local=vLLM server (default: local)")
    parser.add_argument("--base-url",  default=None,
        help="vLLM base URL (default: http://localhost:8000/v1)")
    parser.add_argument("--api-key",   default=None,
        help="NVIDIA API key (or set NVIDIA_API_KEY env var)")
    parser.add_argument("--dry-run",   action="store_true",
        help="Test file I/O without making API calls")
    parser.add_argument("--no-skip", action="store_true", default=False,
        help="Force regenerate all scenes even if output already exists")
    parser.add_argument("--scene-list", default=None,
        help="Path to txt file with one scene name per line (for trainval runs)")
    parser.add_argument("--check-drivelm-overlap", action="store_true",
        help="Check DriveLM overlap (no API calls)")
    parser.add_argument("--pipeline-root", default=None,
        help="Path to moral_pipeline/ directory (auto-detected if not set)")
    parser.add_argument("--nuscenes-dataroot", default="data/nuscenes")
    args = parser.parse_args()

    # Resolve pipeline root
    if args.pipeline_root:
        pipeline_root = Path(args.pipeline_root)
    else:
        script_dir = Path(__file__).parent
        if script_dir.name == "02_cosmos_integration":
            pipeline_root = script_dir.parent
        else:
            pipeline_root = Path.cwd() / "moral_pipeline"

    print(f"Pipeline root:  {pipeline_root}")
    print(f"Condition:      {args.condition}")
    print(f"Model:          nvidia/Cosmos-Reason2-8B")
    print(f"API mode:       {args.api_mode}")

    skip_existing = not args.no_skip  # default: skip existing (safe for resume)
    print(f"Skip existing:  {skip_existing} (use --no-skip to force regenerate)")

    if args.check_drivelm_overlap:
        check_drivelm_overlap(pipeline_root, args.nuscenes_dataroot)
        return

    # Build scene list — priority: --scene > --scene-list > CLEAN_SCENES
    if args.scene:
        scenes = [args.scene]
    elif args.scene_list:
        scene_list_path = Path(args.scene_list)
        if not scene_list_path.exists():
            print(f"ERROR: scene list file not found: {scene_list_path}")
            sys.exit(1)
        scenes = [s.strip() for s in scene_list_path.read_text().splitlines() if s.strip()]
        print(f"Scene list:     {scene_list_path} ({len(scenes)} scenes)")
    else:
        scenes = CLEAN_SCENES
    print(f"Scenes:         {len(scenes)}")
    if args.dry_run:
        print("DRY RUN — no API calls")

    all_qa_pairs = []

    # ── Output paths ──────────────────────────────────────────────────────
    # crash-safe: per-scene files written inside generate_scene_qa()
    # incremental concat: written after every scene so crashes don't lose work
    # final outputs mirror what LLaMA-Factory expects directly
    cosmos_qa_dir = pipeline_root / "02_cosmos_integration" / "cosmos_qa"
    cosmos_qa_dir.mkdir(parents=True, exist_ok=True)

    concat_out   = cosmos_qa_dir / f"all_condition{args.condition}_sharegpt.jsonl"
    train_out    = cosmos_qa_dir / f"train_condition{args.condition}_sharegpt.jsonl"
    val_out      = cosmos_qa_dir / f"val_condition{args.condition}_sharegpt.jsonl"
    quality_out  = cosmos_qa_dir / f"quality_condition{args.condition}.json"

    # If resuming — load already-written concat file to avoid double-writing
    existing_scenes_done = set()
    if concat_out.exists() and skip_existing:
        with open(concat_out) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    sc = rec.get("_meta", {}).get("scene")
                    if sc:
                        existing_scenes_done.add(sc)
                except Exception:
                    pass
        if existing_scenes_done:
            print(f"Resume: {len(existing_scenes_done)} scenes already in concat file")

    # Open concat file in append mode — safe for resume
    concat_f = open(concat_out, "a", encoding="utf-8")

    scenes_done = 0
    scenes_failed = 0

    for scene_name in scenes:
        print(f"\n── {scene_name} ─────────────────────────────────────────")

        qa_pairs = generate_scene_qa(
            scene_name=scene_name,
            condition=args.condition,
            pipeline_root=pipeline_root,
            api_mode=args.api_mode,
            base_url=args.base_url,
            api_key=args.api_key,
            dry_run=args.dry_run,
            skip_existing=skip_existing,
        )

        if not qa_pairs:
            scenes_failed += 1
            continue

        all_qa_pairs.extend(qa_pairs)
        scenes_done += 1

        # ── Write to concat file immediately after each scene ──────────
        # This means a crash loses at most ONE scene, not all of them.
        # Per-scene sharegpt already written inside generate_scene_qa().
        if scene_name not in existing_scenes_done:
            scene_dir = (
                pipeline_root / "outputs"
                / CONDITION_DIRS[args.condition]
                / scene_name
            )
            image_paths = [str(scene_dir / "bev_map.png")]
            for cam_filename in CAMERA_FILENAMES:
                p = scene_dir / cam_filename
                if p.exists():
                    image_paths.append(str(p))
            n_images   = len(image_paths)
            img_tokens = "".join(["<image>\n"] * n_images)

            for qa in qa_pairs:
                record = {
                    "messages": [
                        {
                            "role":    "user",
                            "content": f"{img_tokens}{qa['question']}",
                        },
                        {
                            "role":    "assistant",
                            "content": (
                                f"<think>\n{qa['reasoning']}\n</think>\n\n"
                                f"<answer>\n{qa['answer']}\n</answer>"
                                if qa.get("reasoning") else qa.get("answer", "")
                            ),
                        },
                    ],
                    "images": image_paths,
                    "_meta": {
                        "scene":         qa["scene"],
                        "condition":     qa["condition"],
                        "question_type": qa["question_type"],
                        "quality_ok":    qa.get("quality_ok", False),
                        "gt_verifiable": qa.get("gt_verifiable", False),
                        "gt_value":      qa.get("gt_value"),
                        "gt_field":      qa.get("gt_field"),
                        "gt_action":     qa.get("gt_action"),
                        "gt_outcome":    qa.get("gt_outcome"),
                    },
                }
                concat_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            concat_f.flush()  # flush after each scene — survive power loss

        print(f"  Progress: {scenes_done}/{len(scenes)} scenes done")

    concat_f.close()

    # ── Train / val split (90/10 by scene, not by QA pair) ────────────────
    # Split by scene so same scene never appears in both train and val.
    # This is the correct split — splitting by QA pair would leak scene context.
    import random
    random.seed(42)  # reproducible split

    # Re-read the full concat file to do the split cleanly
    all_records = []
    scene_to_records: dict = {}
    with open(concat_out) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sc  = rec.get("_meta", {}).get("scene", "unknown")
                scene_to_records.setdefault(sc, []).append(rec)
                all_records.append(rec)
            except Exception:
                pass

    all_scene_names = list(scene_to_records.keys())
    random.shuffle(all_scene_names)
    n_val   = max(1, len(all_scene_names) // 10)   # 10% val
    val_scenes   = set(all_scene_names[:n_val])
    train_scenes = set(all_scene_names[n_val:])

    with open(train_out, "w", encoding="utf-8") as tf, \
         open(val_out,   "w", encoding="utf-8") as vf:
        for rec in all_records:
            sc = rec.get("_meta", {}).get("scene", "unknown")
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if sc in val_scenes:
                vf.write(line)
            else:
                tf.write(line)

    n_train = sum(len(v) for k,v in scene_to_records.items() if k in train_scenes)
    n_val_r = sum(len(v) for k,v in scene_to_records.items() if k in val_scenes)

    print(f"\n{'='*60}")
    print(f"DONE — condition {args.condition}")
    print(f"  Scenes processed:  {scenes_done} ok, {scenes_failed} failed")
    print(f"  Total QA pairs:    {len(all_records)}")
    print(f"  Train:             {n_train} pairs ({len(train_scenes)} scenes) → {train_out.name}")
    print(f"  Val:               {n_val_r} pairs ({len(val_scenes)} scenes)   → {val_out.name}")
    print(f"  Full concat:       {len(all_records)} pairs → {concat_out.name}")
    print(f"{'='*60}")
    print(f"\nOutput directory: {cosmos_qa_dir}")
    print(f"  {concat_out.name}")
    print(f"  {train_out.name}   ← use this for LLaMA-Factory training")
    print(f"  {val_out.name}     ← use this for LLaMA-Factory eval_dataset")

    # ── Quality summary ────────────────────────────────────────────────────
    quality_ok = [r for r in all_records if r.get("_meta",{}).get("quality_ok", False)]
    errored    = [r for r in all_records
                  if r.get("messages",[{}])[1].get("content","").startswith("<answer>\n[ERROR")]

    print(f"\nQUALITY:")
    print(f"  Full quality (✅): {len(quality_ok)}/{len(all_records)} "
          f"({100*len(quality_ok)//max(len(all_records),1)}%)")
    print(f"  Errored:           {len(errored)}")

    by_type: dict = {}
    for r in all_records:
        qt = r.get("_meta",{}).get("question_type","unknown")
        qok = r.get("_meta",{}).get("quality_ok", False)
        by_type.setdefault(qt, {"total":0, "ok":0})
        by_type[qt]["total"] += 1
        if qok:
            by_type[qt]["ok"] += 1
    print("\nBy question type:")
    for qt, counts in sorted(by_type.items()):
        pct = 100*counts["ok"]//max(counts["total"],1)
        print(f"  {qt:14s}: {counts['total']:3d} total, {counts['ok']:3d} ok ({pct}%)")

    # Save quality report JSON
    report = {
        "condition":        args.condition,
        "scenes_processed": scenes_done,
        "scenes_failed":    scenes_failed,
        "total_qa_pairs":   len(all_records),
        "quality_ok":       len(quality_ok),
        "errored":          len(errored),
        "train_pairs":      n_train,
        "val_pairs":        n_val_r,
        "train_scenes":     sorted(train_scenes),
        "val_scenes":       sorted(val_scenes),
        "by_question_type": by_type,
    }
    quality_out.write_text(json.dumps(report, indent=2))
    print(f"\nQuality report → {quality_out}")


if __name__ == "__main__":
    main()
