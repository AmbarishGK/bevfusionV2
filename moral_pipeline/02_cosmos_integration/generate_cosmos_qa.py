"""
moral_pipeline/02_cosmos_integration/generate_cosmos_qa.py
============================================================
MoRAL Pipeline — Cosmos Reason 2 QA Generation Script  [v2 — 8B rewrite]

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

AWS setup (g5.2xlarge — A10G 24GB):
  vllm serve nvidia/Cosmos-Reason2-8B \\
    --max-model-len 16384 \\
    --reasoning-parser qwen3 \\
    --port 8000

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

# Camera images — front 3 only to stay within Cosmos 8B 8192-token context limit.
# BEV map already encodes all rear objects. Front cameras cover the safety-critical zone.
# Budget: BEV(~1024t) + 3 front cams(~3072t) + trimmed GT JSON(~800t) + desc(~600t) = ~5500t input
# leaving ~2700 tokens for CoT — sufficient for thorough reasoning.
CAMERA_FILENAMES = [
    "CAM_FRONT.jpg",
    "CAM_FRONT_LEFT.jpg",
    "CAM_FRONT_RIGHT.jpg",
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
    "  1. A BEV (bird's-eye-view) map rendered from LiDAR — shows all detected objects "
    "as coloured shapes around the ego vehicle (white box, centre). "
    "In condition D, cyan arrows on objects show radar-confirmed Doppler velocities.\n"
    "  2. A front-facing camera image — shows the road scene as the driver sees it.\n"
    "  3. GROUND TRUTH DATA — structured sensor annotations with exact distances, "
    "speeds, bearings, and visibility for every detected object.\n\n"
    "STRICT REASONING ORDER — you must follow this for every answer:\n"
    "  Step 1 — READ THE BEV MAP: Describe what you see spatially. Where are the objects "
    "relative to ego? What is the spatial layout — crowded ahead, clear to the right, "
    "objects merging from the left? If condition D, describe the cyan Doppler vectors "
    "you can see on objects — their direction and approximate magnitude.\n"
    "  Step 2 — READ THE FRONT CAMERA: Describe what is visible in the camera image. "
    "What does the road look like? What objects can you see directly, and what might be "
    "partially obscured? Connect what you see in the camera to what the BEV shows.\n"
    "  Step 3 — CONFIRM WITH GT DATA: Use the exact numerical values from GROUND TRUTH "
    "DATA to confirm and quantify what you observed in Steps 1 and 2. "
    "Do all relevant calculations here (TTC, stopping distance, lateral gap). "
    "Never invent numbers — every figure must come from the GT data.\n"
    "  Step 4 — DRIVING DECISION: State a clear, specific action the vehicle should take "
    "and why, grounded in the numbers from Step 3.\n\n"
    "Always respond in this exact format:\n"
    "<think>\n"
    "[BEV] What you observe in the BEV map image.\n"
    "[CAM] What you observe in the front camera image.\n"
    "[GT]  Exact values from GT data + calculations.\n"
    "[DECISION] Recommended action and physical justification.\n"
    "</think>\n\n"
    "<answer>\n"
    "Concise answer: situation, key number(s), recommended action.\n"
    "</answer>"
)

FORMAT_INSTRUCTION = ""  # Format is fully specified in system prompt

# ── Cosmos 8B parameters ───────────────────────────────────────────────────────
COSMOS_PARAMS = {
    "model": "nvidia/Cosmos-Reason2-8B",
    "max_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.95,
    "presence_penalty": 0.0,
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

    # Text block: scene description + full GT JSON + question
    gt_block = (
        f"\n\nGROUND TRUTH DATA (detections.json — use these exact values):\n"
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
            f"Use the exact numerical values from GROUND TRUTH DATA above."
        ),
    })

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": COSMOS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        **COSMOS_PARAMS,
    )

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
        KEEP_FIELDS = {
            "class", "distance_m", "bearing_deg",
            "velocity_ms", "vx", "vy",
            "visibility", "num_lidar_pts",
            "radar_velocity_ms", "radar_velocity_confirmed", "radar_quality",
            "width_m", "length_m",
        }
        trimmed = [{k: v for k, v in d.items() if k in KEEP_FIELDS} for d in detections]
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

    for q_type, question in QUESTION_TEMPLATES.items():
        if q_type == "radar" and condition not in ("D",):
            print(f"    [{q_type}] SKIPPED (radar only for condition D)")
            continue

        print(f"    [{q_type}] Querying Cosmos 8B...", end=" ", flush=True)
        t0 = time.time()

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

        elapsed       = time.time() - t0
        total_tokens += result["tokens_used"]
        print(f"done ({elapsed:.1f}s, {result['tokens_used']} tokens)")

        gt_fields = extract_gt_fields(detections, q_type)

        qa_pair = {
            "scene":         scene_name,
            "condition":     condition,
            "question_type": q_type,
            "question":      question,
            "reasoning":     result["reasoning"],
            "answer":        result["answer"],
            **gt_fields,
        }
        qa_pairs.append(qa_pair)

        if not dry_run:
            time.sleep(0.5)

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
    parser.add_argument("--skip-existing", action="store_true",
        help="Skip scenes already processed (default: regenerate all)")
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

    if args.check_drivelm_overlap:
        check_drivelm_overlap(pipeline_root, args.nuscenes_dataroot)
        return

    scenes = [args.scene] if args.scene else CLEAN_SCENES
    print(f"Scenes:         {len(scenes)}")
    if args.dry_run:
        print("DRY RUN — no API calls")

    all_qa_pairs = []

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
            skip_existing=args.skip_existing,
        )
        all_qa_pairs.extend(qa_pairs)

    # ── Concatenated training file ────────────────────────────────────────
    concat_out = (
        pipeline_root
        / "02_cosmos_integration"
        / f"all_qa_condition{args.condition}_sharegpt.jsonl"
    )
    concat_out.parent.mkdir(parents=True, exist_ok=True)

    with open(concat_out, "w") as f:
        for qa in all_qa_pairs:
            scene_dir = (
                pipeline_root / "outputs"
                / CONDITION_DIRS[args.condition]
                / qa["scene"]
            )
            # Rebuild image list for this scene
            image_paths = [str(scene_dir / "bev_map.png")]
            # condition D bev_map.png already has radar overlay
            for cam_filename in CAMERA_FILENAMES:
                p = scene_dir / cam_filename
                if p.exists():
                    image_paths.append(str(p))

            n_images = len(image_paths)
            image_tokens = "".join(["<image>\n"] * n_images)

            record = {
                "messages": [
                    {
                        "role":    "user",
                        "content": f"{image_tokens}{qa['question']}",
                    },
                    {
                        "role":    "assistant",
                        "content": (
                            f"<think>\n{qa['reasoning']}\n</think>\n\n"
                            f"<answer>\n{qa['answer']}\n</answer>"
                            if qa["reasoning"] else qa["answer"]
                        ),
                    },
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
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"DONE: {len(all_qa_pairs)} total QA pairs")
    print(f"Training file → {concat_out}")
    print(f"{'='*60}")

    verifiable = [qa for qa in all_qa_pairs if qa.get("gt_verifiable")]
    print(f"GT-verifiable: {len(verifiable)}/{len(all_qa_pairs)}")
    by_type: dict[str, int] = {}
    for qa in all_qa_pairs:
        qt = qa["question_type"]
        by_type[qt] = by_type.get(qt, 0) + 1
    print("By type:")
    for qt, count in sorted(by_type.items()):
        print(f"  {qt}: {count}")


if __name__ == "__main__":
    main()
