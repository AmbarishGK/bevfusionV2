"""
moral_pipeline/02_cosmos_integration/generate_cosmos_qa.py
============================================================
MoRAL Pipeline — Cosmos Reason 2 QA Generation Script

PURPOSE:
  For each clean scene, sends BEV map + scene_description.txt to Cosmos Reason 2
  and generates structured QA pairs with chain-of-thought reasoning.
  Saves output as ShareGPT-format JSONL for LLaMA-Factory fine-tuning.

USAGE:
  # Smoke test on one scene (no API cost except ~10 questions):
  python generate_cosmos_qa.py --scene scene-0061 --condition B --dry-run

  # Generate for all clean scenes, condition B (BEV + GT):
  python generate_cosmos_qa.py --condition B

  # Generate for all clean scenes, condition D (BEV + radar):
  python generate_cosmos_qa.py --condition D

  # Check DriveLM overlap first (no API calls):
  python generate_cosmos_qa.py --check-drivelm-overlap

ACCESS MODES (set via --api-mode):
  nim    : NVIDIA NIM cloud API (fastest to start, ~$0.001/question)
           Get key at: https://build.nvidia.com
           Set env:    export NVIDIA_API_KEY=nvapi-xxxx

  local  : Local vLLM server (AWS g4dn.2xlarge or better)
           Start server: vllm serve nvidia/Cosmos-Reason2-8B --port 8000
           Set env:      export COSMOS_BASE_URL=http://localhost:8000/v1

OUTPUT:
  moral_pipeline/02_cosmos_integration/cosmos_qa/
    scene-XXXX/
      qa_pairs.json          — structured QA with GT verification fields
      qa_pairs_sharegpt.jsonl — training format for LLaMA-Factory

  moral_pipeline/02_cosmos_integration/
    all_qa_sharegpt.jsonl    — concatenated training file (all scenes)
    drivelm_overlap.json     — which sample tokens are in DriveLM (eval set)

SCENE STATUS (update this list after visual checks):
  CLEAN (use for training):  0061, 0553, 0655, 0757, 0796, 0916, 1077, 1094, 1100
  DROPPED (annotation gaps): 0103
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

# All 9 confirmed clean scenes (scene-0103 dropped — annotation gaps confirmed)
CLEAN_SCENES = [
    "scene-0061",
    "scene-0553",
    "scene-0655",
    "scene-0757",   # ✅ verified 2026-02-24: construction zone, clear ahead
    "scene-0796",   # ✅ verified 2026-02-24: daytime Singapore, pedestrians annotated
    "scene-0916",
    "scene-1077",   # ✅ verified 2026-02-24: night scene, all vehicles annotated
    "scene-1094",
    "scene-1100",
]

# Condition folder names
CONDITION_DIRS = {
    "A": "00_camera_only",
    "B": "01_gt_annotations",
    "D": "02_gt_with_radar",
}

# Question templates — 8 categories × 1 question each = 8 API calls per scene
# These are grounded in what detections.json actually contains.
# Cosmos sees: BEV map image + scene_description.txt + this question
QUESTION_TEMPLATES = {
    "spatial": (
        "Looking at the BEV map and scene description, what is the nearest object "
        "directly ahead of the ego vehicle? State the object class, distance in metres, "
        "and bearing. If nothing is directly ahead, state the nearest object in any forward zone."
    ),
    "safety": (
        "Based on the sensor data provided, identify the single most critical safety risk "
        "in this scene. State which object it is, its distance, estimated time-to-collision "
        "if applicable, and why it represents the highest risk."
    ),
    "velocity": (
        "Which objects in this scene are actively moving (speed > 0.5 m/s)? "
        "For each moving object, state its class, speed in m/s, direction of travel "
        "relative to the ego vehicle, and whether the motion is toward or away from ego."
    ),
    "occlusion": (
        "The scene description includes visibility ratings from LiDAR. "
        "Which objects are partially or mostly occluded? What risk does each occluded object "
        "pose, and how should the driver account for objects that cameras may miss but "
        "LiDAR has detected?"
    ),
    "gap": (
        "Examining the objects to the right of the ego vehicle, is there sufficient gap "
        "to safely change lanes or pass? State the gap size in metres for each relevant "
        "object and your conclusion about whether a lane change is safe."
    ),
    "physics": (
        "Given the ego vehicle speed stated in the scene description and the current "
        "stopping distance, can the ego vehicle safely stop before reaching the nearest "
        "object ahead? Show your calculation and state whether emergency braking is required."
    ),
    "zone": (
        "Provide a complete inventory of all objects in the front-left and front-right "
        "zones. For each object, state class, distance, speed, and visibility. "
        "Which zone presents greater risk and why?"
    ),
    "radar": (
        "The scene description includes radar Doppler velocity data for some objects. "
        "Which objects have radar-confirmed velocity readings? For each, state whether "
        "the radar reading agrees with the GT velocity estimate, and what the Doppler "
        "data tells us that cameras alone could not reveal."
    ),
}

# Cosmos Reason 2 prompt structure (from NVIDIA Cosmos Cookbook)
COSMOS_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the question in the following format: "
    "<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."
)

FORMAT_INSTRUCTION = ""   # Now empty — format is in system prompt


# Cosmos sampling parameters (from cookbook)
COSMOS_PARAMS = {
    "model": "nvidia/cosmos-reason2-8b",
    "max_tokens": 2048,
    "temperature": 0.6,
    "top_p": 0.95,
    "presence_penalty": 0.0,
}


# ── DriveLM Overlap Detection ──────────────────────────────────────────────────

def check_drivelm_overlap(
    pipeline_root: Path,
    nuscenes_dataroot: str = "data/nuscenes",
    drivelm_json: str = "DriveLM/v1_1_train_nus.json",
) -> dict:
    """
    Find which nuScenes mini sample tokens appear in DriveLM.
    These samples should be held out as evaluation set (never used for training QA).

    Returns dict: {scene_name: [sample_token, ...], ...}
    """
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        print("ERROR: nuscenes-devkit not installed. Run: pip install nuscenes-devkit")
        return {}

    print("Loading nuScenes mini...")
    nusc = NuScenes(version="v1.0-mini", dataroot=nuscenes_dataroot, verbose=False)

    # Collect all sample tokens per scene
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

    # Load DriveLM
    drivelm_path = pipeline_root.parent / drivelm_json
    if not drivelm_path.exists():
        print(f"WARNING: DriveLM not found at {drivelm_path}")
        print("Clone it: git clone https://huggingface.co/datasets/OpenDriveLab/DriveLM")
        return {}

    print(f"Loading DriveLM from {drivelm_path}...")
    with open(drivelm_path) as f:
        drivelm = json.load(f)

    drivelm_tokens = set(drivelm.keys())
    print(f"DriveLM tokens: {len(drivelm_tokens)}")

    # Find overlap
    overlap_by_scene: dict[str, list[str]] = {}
    total_overlap = 0
    for scene_name, tokens in scene_tokens.items():
        overlapping = [t for t in tokens if t in drivelm_tokens]
        if overlapping:
            overlap_by_scene[scene_name] = overlapping
            total_overlap += len(overlapping)

    print(f"\nOverlap: {total_overlap} sample tokens across {len(overlap_by_scene)} scenes")
    if overlap_by_scene:
        print("Scenes with DriveLM overlap (USE AS EVAL, not training):")
        for scene, tokens in overlap_by_scene.items():
            print(f"  {scene}: {len(tokens)} tokens")
    else:
        print("No overlap found — all mini scenes can be used for training QA.")
        print("Use DriveLM questions as question templates only.")

    # Save result
    out_path = pipeline_root / "02_cosmos_integration" / "drivelm_overlap.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(overlap_by_scene, f, indent=2)
    print(f"\nSaved overlap map → {out_path}")

    return overlap_by_scene


# ── Image Loading ──────────────────────────────────────────────────────────────

def load_image_b64(image_path: Path) -> str:
    """Load image and return base64 string for API."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


# ── Cosmos API Call ────────────────────────────────────────────────────────────

def call_cosmos(
    bev_image_b64: str,
    scene_description: str,
    question: str,
    api_mode: str = "nim",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """
    Call Cosmos Reason 2 with BEV image + scene description + question.

    Prompt structure follows NVIDIA Cosmos Cookbook requirements:
    - System: minimal ("You are a helpful assistant.")
    - User: [image FIRST, then text] + format instruction appended to user turn
    - Media before text in user message payload
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

    # Set up client
    if api_mode == "nim":
        key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise ValueError(
                "Set NVIDIA_API_KEY env var or pass --api-key. "
                "Get key at: https://build.nvidia.com"
            )
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=key,
        )
    elif api_mode == "local":
        url = base_url or os.environ.get("COSMOS_BASE_URL", "http://localhost:8000/v1")
        client = OpenAI(base_url=url, api_key="not-needed")
    elif api_mode == "local-hf":
        # Use local Qwen3-VL-7B backend — bypass OpenAI client entirely
        sys.path.insert(0, str(Path(__file__).parent))
        from local_vlm_backend import call_cosmos_local
        return call_cosmos_local(
            bev_image_b64=bev_image_b64,
            scene_description=scene_description,
            question=question,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown api_mode: {api_mode}. Use 'nim', 'local', or 'local-hf'")

    # Build user message: IMAGE FIRST (cookbook requirement), then text
    user_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{bev_image_b64}",
            },
        },
        {
            "type": "text",
            "text": (
                f"SCENE SENSOR DATA:\n{scene_description}\n\n"
                f"QUESTION: {question}"
                f"{FORMAT_INSTRUCTION}"
            ),
        },
    ]

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": COSMOS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        **COSMOS_PARAMS,
    )

    raw = response.choices[0].message.content

    # Parse <think>...</think> + final answer
    reasoning = ""
    answer = raw
    if "<think>" in raw and "</think>" in raw:
        think_start = raw.index("<think>") + len("<think>")
        think_end = raw.index("</think>")
        reasoning = raw[think_start:think_end].strip()
        # Extract <answer>...</answer> block
        if "<answer>" in raw and "</answer>" in raw:
            ans_start = raw.index("<answer>") + len("<answer>")
            ans_end = raw.index("</answer>")
            answer = raw[ans_start:ans_end].strip()
        else:
            answer = raw[think_end + len("<tool_call>"):].strip()

    return {
        "reasoning": reasoning,
        "answer": answer,
        "raw_response": raw,
        "tokens_used": response.usage.total_tokens if response.usage else 0,
    }


# ── GT Verification Fields ─────────────────────────────────────────────────────

def extract_gt_fields(detections: list[dict], question_type: str) -> dict:
    """
    Extract ground-truth verifiable fields from detections.json for a given question type.
    These allow automatic scoring of Cosmos answers later.
    """
    gt = {"gt_verifiable": False, "gt_value": None, "gt_field": None}

    if not detections:
        return gt

    # Sort by distance
    by_dist = sorted(detections, key=lambda d: d["distance_m"])

    if question_type == "spatial":
        # Find nearest object in directly-ahead zone (bearing within ±45°)
        ahead = [
            d for d in by_dist
            if abs(d.get("bearing_deg", 90)) <= 45
        ]
        if ahead:
            obj = ahead[0]
            gt = {
                "gt_verifiable": True,
                "gt_value": obj["distance_m"],
                "gt_field": "distance_m",
                "gt_object_class": obj["class"],
                "gt_bearing_deg": obj.get("bearing_deg"),
            }

    elif question_type == "safety":
        # Find object with lowest TTC (distance / relative closing speed)
        # Simple approximation: nearest fast-moving object toward ego
        closing = [
            d for d in by_dist
            if d.get("velocity_ms", 0) > 0.5 and d["distance_m"] < 30
        ]
        if closing:
            obj = closing[0]
            ttc = obj["distance_m"] / max(obj["velocity_ms"], 0.1)
            gt = {
                "gt_verifiable": True,
                "gt_value": round(ttc, 2),
                "gt_field": "ttc_s",
                "gt_object_class": obj["class"],
                "gt_distance_m": obj["distance_m"],
            }

    elif question_type == "velocity":
        # Find all moving objects
        moving = [d for d in detections if d.get("velocity_ms", 0) > 0.5]
        if moving:
            fastest = max(moving, key=lambda d: d["velocity_ms"])
            gt = {
                "gt_verifiable": True,
                "gt_value": fastest["velocity_ms"],
                "gt_field": "velocity_ms",
                "gt_object_class": fastest["class"],
                "gt_moving_count": len(moving),
            }

    elif question_type == "physics":
        # Nearest ahead + ego stopping distance (from scene_desc parsed separately)
        ahead = [d for d in by_dist if abs(d.get("bearing_deg", 90)) <= 45]
        if ahead:
            obj = ahead[0]
            gt = {
                "gt_verifiable": True,
                "gt_value": obj["distance_m"],
                "gt_field": "nearest_ahead_m",
                "gt_object_class": obj["class"],
            }

    return gt


# ── Scene QA Generation ────────────────────────────────────────────────────────

def generate_scene_qa(
    scene_name: str,
    condition: str,
    pipeline_root: Path,
    api_mode: str = "nim",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> list[dict]:
    """
    Generate QA pairs for one scene under one condition.
    Returns list of QA pair dicts.
    """
    condition_dir = CONDITION_DIRS[condition]
    scene_dir = pipeline_root / "outputs" / condition_dir / scene_name

    if not scene_dir.exists():
        print(f"  WARNING: {scene_dir} not found, skipping")
        return []

    # Output path
    out_dir = pipeline_root / "02_cosmos_integration" / "cosmos_qa" / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_out = out_dir / f"qa_pairs_condition{condition}.json"

    if skip_existing and qa_out.exists():
        print(f"  Skipping {scene_name} condition {condition} — already exists")
        with open(qa_out) as f:
            return json.load(f)

    # Load inputs
    bev_path = scene_dir / "bev_map.png"
    desc_path = scene_dir / "scene_description.txt"
    det_path = scene_dir / "detections.json"

    if not bev_path.exists():
        print(f"  WARNING: No bev_map.png in {scene_dir}, skipping")
        return []

    print(f"  Loading BEV map from {bev_path}")
    bev_b64 = load_image_b64(bev_path)

    scene_description = ""
    if desc_path.exists():
        scene_description = desc_path.read_text()

    detections = []
    if det_path.exists():
        with open(det_path) as f:
            detections = json.load(f)

    qa_pairs = []
    total_tokens = 0

    for q_type, question in QUESTION_TEMPLATES.items():
        print(f"    [{q_type}] Querying Cosmos...", end=" ", flush=True)
        t0 = time.time()

        # Skip radar question for non-radar conditions
        if q_type == "radar" and condition not in ("D",):
            print("SKIPPED (radar question only for condition D)")
            continue

        result = call_cosmos(
            bev_image_b64=bev_b64,
            scene_description=scene_description,
            question=question,
            api_mode=api_mode,
            base_url=base_url,
            api_key=api_key,
            dry_run=dry_run,
        )

        elapsed = time.time() - t0
        total_tokens += result["tokens_used"]
        print(f"done ({elapsed:.1f}s, {result['tokens_used']} tokens)")

        # Build QA pair record
        gt_fields = extract_gt_fields(detections, q_type)

        qa_pair = {
            "scene": scene_name,
            "condition": condition,
            "question_type": q_type,
            "question": question,
            "reasoning": result["reasoning"],
            "answer": result["answer"],
            **gt_fields,
        }
        qa_pairs.append(qa_pair)

        # Rate limiting — avoid hammering the API
        if not dry_run:
            time.sleep(0.5)

    # Save QA pairs
    with open(qa_out, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"  Saved {len(qa_pairs)} QA pairs → {qa_out} ({total_tokens} total tokens)")

    # Also save in ShareGPT format for LLaMA-Factory
    sharegpt_out = out_dir / f"qa_pairs_condition{condition}_sharegpt.jsonl"
    save_sharegpt(qa_pairs, scene_dir, sharegpt_out)

    return qa_pairs


# ── ShareGPT Format ────────────────────────────────────────────────────────────

def save_sharegpt(
    qa_pairs: list[dict],
    scene_dir: Path,
    out_path: Path,
) -> None:
    """
    Save QA pairs in ShareGPT JSONL format for LLaMA-Factory fine-tuning.

    Format: one JSON object per line
    {
      "messages": [
        {"role": "user",      "content": "<image>\n<image>\nQUESTION"},
        {"role": "assistant", "content": "<think>REASONING</think>ANSWER"}
      ],
      "images": ["path/to/bev_map.png", "path/to/CAM_FRONT.jpg"]
    }
    """
    bev_path = str(scene_dir / "bev_map.png")
    cam_path = str(scene_dir / "CAM_FRONT.jpg")

    lines = []
    for qa in qa_pairs:
        # Training input: BEV map + front camera + question (no scene_description at inference)
        user_content = f"<image>\n<image>\n{qa['question']}"

        # Training output: full chain-of-thought + answer
        if qa["reasoning"]:
            assistant_content = f"<think>\n{qa['reasoning']}\n</think>\n{qa['answer']}"
        else:
            assistant_content = qa["answer"]

        record = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": [bev_path, cam_path],
            # Metadata (ignored by training, useful for filtering)
            "_meta": {
                "scene": qa["scene"],
                "condition": qa["condition"],
                "question_type": qa["question_type"],
                "gt_verifiable": qa.get("gt_verifiable", False),
                "gt_value": qa.get("gt_value"),
                "gt_field": qa.get("gt_field"),
            },
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Saved ShareGPT JSONL → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Cosmos Reason 2 QA pairs for MoRAL pipeline"
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Single scene to process (e.g. scene-0061). Default: all clean scenes.",
    )
    parser.add_argument(
        "--condition",
        choices=["A", "B", "D"],
        default="B",
        help="Data condition: A=camera_only, B=gt_annotations, D=gt_with_radar (default: B)",
    )
    parser.add_argument(
        "--api-mode",
        choices=["nim", "local", "local-hf"],
        default="local-hf",
        help="API access: nim=NVIDIA cloud, local=vLLM server (default: nim)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for local vLLM server (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="NVIDIA API key (or set NVIDIA_API_KEY env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline without making API calls (tests file I/O only)",
    )
    parser.add_argument(
        "--check-drivelm-overlap",
        action="store_true",
        help="Check which scenes overlap with DriveLM (no API calls)",
    )
    parser.add_argument(
        "--pipeline-root",
        default=None,
        help="Path to moral_pipeline/ directory. Auto-detected if not set.",
    )
    parser.add_argument(
        "--nuscenes-dataroot",
        default="data/nuscenes",
        help="Path to nuScenes data directory (default: data/nuscenes)",
    )
    args = parser.parse_args()

    # Resolve pipeline root
    if args.pipeline_root:
        pipeline_root = Path(args.pipeline_root)
    else:
        # Auto-detect: script lives in moral_pipeline/02_cosmos_integration/
        # so pipeline_root = 2 levels up from this file
        script_dir = Path(__file__).parent
        if script_dir.name == "02_cosmos_integration":
            pipeline_root = script_dir.parent
        else:
            # Fallback: assume CWD is bevfusionV2/
            pipeline_root = Path.cwd() / "moral_pipeline"

    print(f"Pipeline root: {pipeline_root}")

    # DriveLM overlap check
    if args.check_drivelm_overlap:
        check_drivelm_overlap(
            pipeline_root=pipeline_root,
            nuscenes_dataroot=args.nuscenes_dataroot,
        )
        return

    # Which scenes to process
    scenes = [args.scene] if args.scene else CLEAN_SCENES
    print(f"Processing {len(scenes)} scenes, condition {args.condition}")
    if args.dry_run:
        print("DRY RUN MODE — no API calls will be made")

    all_qa_pairs = []

    for scene_name in scenes:
        print(f"\n── {scene_name} ──────────────────────────────")
        qa_pairs = generate_scene_qa(
            scene_name=scene_name,
            condition=args.condition,
            pipeline_root=pipeline_root,
            api_mode=args.api_mode,
            base_url=args.base_url,
            api_key=args.api_key,
            dry_run=args.dry_run,
        )
        all_qa_pairs.extend(qa_pairs)

    # Concatenate all ShareGPT records into one training file
    concat_out = (
        pipeline_root
        / "02_cosmos_integration"
        / f"all_qa_condition{args.condition}_sharegpt.jsonl"
    )
    concat_out.parent.mkdir(parents=True, exist_ok=True)

    with open(concat_out, "w") as f:
        for qa in all_qa_pairs:
            scene_dir = (
                pipeline_root
                / "outputs"
                / CONDITION_DIRS[args.condition]
                / qa["scene"]
            )
            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<image>\n<image>\n{qa['question']}",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"<think>\n{qa['reasoning']}\n</think>\n{qa['answer']}"
                            if qa["reasoning"]
                            else qa["answer"]
                        ),
                    },
                ],
                "images": [
                    str(scene_dir / "bev_map.png"),
                    str(scene_dir / "CAM_FRONT.jpg"),
                ],
                "_meta": {
                    "scene": qa["scene"],
                    "condition": qa["condition"],
                    "question_type": qa["question_type"],
                    "gt_verifiable": qa.get("gt_verifiable", False),
                    "gt_value": qa.get("gt_value"),
                    "gt_field": qa.get("gt_field"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"DONE: {len(all_qa_pairs)} total QA pairs")
    print(f"Training file → {concat_out}")
    print(f"{'='*60}")

    # Print summary stats
    verifiable = [qa for qa in all_qa_pairs if qa.get("gt_verifiable")]
    print(f"GT-verifiable QA pairs: {len(verifiable)}/{len(all_qa_pairs)}")
    by_type: dict[str, int] = {}
    for qa in all_qa_pairs:
        qt = qa["question_type"]
        by_type[qt] = by_type.get(qt, 0) + 1
    print("QA pairs by type:")
    for qt, count in sorted(by_type.items()):
        print(f"  {qt}: {count}")


if __name__ == "__main__":
    main()
