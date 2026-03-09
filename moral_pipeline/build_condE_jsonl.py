#!/usr/bin/env python3
"""
MoRAL — condE JSONL Builder
============================
Builds training/val JSONL where each sample contains:
  - Raw LiDAR+Radar BEV image  (outputs/03_clean_bev/scene-XXXX/bev_map.png)
  - CAM_FRONT image            (outputs/03_clean_bev/scene-XXXX/CAM_FRONT.jpg)
  - detections.json as structured text (full rich fields)
  - Two-stage reasoning prompt: describe BEV → correlate with detections → answer

Goal: Model learns cross-modal grounding — correlating visual BEV patterns
with structured detection data. At eval time, detections can be removed
to test how well the model internalized spatial understanding from BEV alone.

Usage:
    python build_condE_jsonl.py \
        --input_train 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
        --input_val   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
        --det_root    outputs/02_gt_with_radar \
        --bev_root    outputs/03_clean_bev \
        --output_dir  02_cosmos_integration/hf_data

Output:
    clean_conditionE_train.jsonl
    clean_conditionE_val.jsonl
"""

import os, json, argparse
from pathlib import Path

# ── Detection text formatter ───────────────────────────────────────────────────

def format_detection_entry(d, idx):
    """Format a single detection into rich text the model can learn from."""
    lines = []
    cls = d.get('class', 'unknown')
    dist = d.get('distance_m', '?')
    bearing = d.get('bearing_deg', '?')
    vel = d.get('velocity_ms', '?')
    vx = d.get('vx', '?')
    vy = d.get('vy', '?')
    w = d.get('width_m', '?')
    l = d.get('length_m', '?')
    h = d.get('height_m', '?')
    vis = d.get('visibility', 'unknown')
    n_lidar = d.get('num_lidar_pts', 0)
    n_radar = d.get('num_radar_pts', 0)

    # Radar-specific
    r_vel = d.get('radar_velocity_ms')
    r_quality = d.get('radar_quality', 'unconfirmed')
    r_confirmed = d.get('radar_velocity_confirmed', False)
    r_rcs = d.get('radar_rcs')

    lines.append(f"[Object {idx+1}] {cls.upper()}")
    lines.append(f"  Position : {dist:.1f}m away, bearing {bearing:.1f}° from ego")
    lines.append(f"  Size     : {w:.2f}m wide × {l:.2f}m long × {h:.2f}m tall")
    lines.append(f"  Velocity : {vel:.2f} m/s  (vx={vx:.2f}, vy={vy:.2f})")
    lines.append(f"  LiDAR pts: {n_lidar}  |  Radar pts: {n_radar}")
    lines.append(f"  Visibility: {vis}")

    if r_vel is not None:
        lines.append(f"  Radar velocity: {r_vel:.2f} m/s (quality: {r_quality}, confirmed: {r_confirmed})")
    else:
        lines.append(f"  Radar velocity: not confirmed (quality: {r_quality})")

    if r_rcs is not None:
        lines.append(f"  Radar RCS: {r_rcs:.1f} dBsm")

    return '\n'.join(lines)


def format_detections_rich(dets, ego_speed=None):
    """Format full detections.json into rich structured text."""
    lines = ["=== SCENE DETECTIONS ==="]

    if ego_speed is not None:
        lines.append(f"Ego vehicle speed: {ego_speed:.2f} m/s\n")

    if not dets:
        lines.append("No detections available.")
        return '\n'.join(lines)

    # Sort by distance
    dets_sorted = sorted(dets, key=lambda d: d.get('distance_m', 999))

    for idx, d in enumerate(dets_sorted):
        lines.append(format_detection_entry(d, idx))
        lines.append("")  # blank line between objects

    lines.append("=== END DETECTIONS ===")
    return '\n'.join(lines)


# ── Two-stage reasoning prompt ─────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert autonomous driving perception assistant. "
    "You reason about driving scenes using Bird's Eye View (BEV) sensor maps, "
    "front camera images, and structured detection data. "
    "When given a BEV map, first describe what you observe visually "
    "(point cloud density, object shapes, movement indicators). "
    "Then correlate your visual observations with the detection data. "
    "Finally, use both to answer the question accurately."
)

TWO_STAGE_PREFIX = (
    "You are given:\n"
    "1. A Bird's Eye View (BEV) map showing raw LiDAR + Radar point clouds\n"
    "2. A front camera image (CAM_FRONT)\n"
    "3. Structured detection data from sensors\n\n"
    "STEP 1 — Describe the BEV map:\n"
    "Look at the BEV image carefully. Identify dense point cloud clusters "
    "(likely objects), sparse regions (open road), and any movement indicators "
    "(arrow overlays showing velocity). Note approximate positions and directions.\n\n"
    "STEP 2 — Correlate with detections:\n"
    "Match each detection entry to what you see in the BEV. For example, "
    "a car at 10.3m bearing -116.8° should appear as a dense cluster "
    "in the rear-left quadrant of the BEV. Confirm or note discrepancies.\n\n"
    "STEP 3 — Answer the question:\n"
    "Use both your visual understanding and the detection data to give "
    "a precise, grounded answer.\n\n"
)


def build_prompt_text(det_text, question_text):
    """Combine two-stage prefix + detections + question."""
    return (
        TWO_STAGE_PREFIX
        + det_text
        + "\n\n"
        + "QUESTION: " + question_text.strip()
    )


# ── JSONL builder ──────────────────────────────────────────────────────────────

def process_record(record, det_root, bev_root):
    """
    Transform a condD record into a condE record with:
    - bev_lidar_radar.png (raw LiDAR+Radar BEV)
    - CAM_FRONT.jpg
    - rich detections text in prompt
    - two-stage reasoning instruction
    """
    meta = record.get('_meta', {})
    scene = meta.get('scene', record.get('scene', ''))

    if not scene:
        return None

    # BEV image path — use bev_lidar_radar.png (LiDAR+Radar)
    bev_path = os.path.join(bev_root, scene, 'bev_lidar_radar.png')
    # fallback to bev_map.png
    if not os.path.exists(bev_path):
        bev_path = os.path.join(bev_root, scene, 'bev_map.png')
    if not os.path.exists(bev_path):
        return None

    # CAM_FRONT path
    cam_path = os.path.join(bev_root, scene, 'CAM_FRONT.jpg')
    if not os.path.exists(cam_path):
        cam_path = os.path.join(bev_root, scene, 'CAM_FRONT.png')

    # detections.json — prefer gt_with_radar (has radar fields)
    det_path = os.path.join(det_root, scene, 'detections.json')
    if not os.path.exists(det_path):
        # fallback to gt_annotations
        det_path = os.path.join(
            det_root.replace('02_gt_with_radar', '01_gt_annotations'),
            scene, 'detections.json'
        )

    dets = []
    if os.path.exists(det_path):
        with open(det_path) as f:
            dets = json.load(f)

    # Format detection text
    ego_speed = meta.get('ego_speed_ms')
    det_text = format_detections_rich(dets, ego_speed=ego_speed)

    # Extract original question from messages
    orig_content = record['messages'][0]['content']
    texts = [p for p in orig_content if p.get('type') == 'text']
    question_text = texts[0]['text'] if texts else ''

    # Build prompt
    full_prompt = build_prompt_text(det_text, question_text)

    # Build new messages — same structure as condD train JSONL
    new_content = []

    # Image 1: BEV map
    new_content.append({'type': 'image', 'image': bev_path})

    # Image 2: CAM_FRONT (if exists)
    if os.path.exists(cam_path):
        new_content.append({'type': 'image', 'image': cam_path})

    # Text: two-stage prompt
    new_content.append({'type': 'text', 'text': full_prompt})

    # Assistant answer stays the same as condD
    assistant_answer = record['messages'][1]['content'] if len(record['messages']) > 1 else ''

    new_record = {
        '_meta': {
            **meta,
            'condition': 'E',
            'input_level': 'clean_radar_det',
            'bev_path': bev_path,
            'cam_path': cam_path if os.path.exists(cam_path) else '',
            'det_path': det_path if os.path.exists(det_path) else '',
            'n_detections': len(dets),
        },
        'messages': [
            {'role': 'user', 'content': new_content},
            {'role': 'assistant', 'content': assistant_answer},
        ]
    }

    return new_record


def build_jsonl(input_path, output_path, det_root, bev_root):
    with open(input_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    print(f"Processing {len(records)} records from {input_path}")

    out_records = []
    skipped = 0
    no_det = 0

    for rec in records:
        new_rec = process_record(rec, det_root, bev_root)
        if new_rec is None:
            skipped += 1
            continue
        if new_rec['_meta']['n_detections'] == 0:
            no_det += 1
        out_records.append(new_rec)

    print(f"  Written : {len(out_records)}")
    print(f"  Skipped : {skipped} (missing BEV)")
    print(f"  No dets : {no_det} (detections.json missing/empty)")

    with open(output_path, 'w') as f:
        for rec in out_records:
            f.write(json.dumps(rec) + '\n')

    print(f"  Saved → {output_path}")
    return len(out_records)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train', default='02_cosmos_integration/hf_data/clean_conditionD_train.jsonl')
    parser.add_argument('--input_val',   default='02_cosmos_integration/hf_data/clean_conditionD_val.jsonl')
    parser.add_argument('--det_root',    default='outputs/02_gt_with_radar',
                        help='Root dir containing per-scene detections.json with radar fields')
    parser.add_argument('--bev_root',    default='outputs/03_clean_bev',
                        help='Root dir containing bev_lidar_radar.png and CAM_FRONT.jpg per scene')
    parser.add_argument('--output_dir',  default='02_cosmos_integration/hf_data')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_out = os.path.join(args.output_dir, 'clean_conditionE_train.jsonl')
    val_out   = os.path.join(args.output_dir, 'clean_conditionE_val.jsonl')

    print("\n=== Building condE train JSONL ===")
    n_train = build_jsonl(args.input_train, train_out, args.det_root, args.bev_root)

    print("\n=== Building condE val JSONL ===")
    n_val = build_jsonl(args.input_val, val_out, args.det_root, args.bev_root)

    print(f"\n✅ Done. Train: {n_train}, Val: {n_val}")
    print(f"\nNext steps:")
    print(f"  # Dry run")
    print(f"  python train_cosmos2b.py \\")
    print(f"    --train_file {train_out} \\")
    print(f"    --val_file   {val_out} \\")
    print(f"    --output_dir saves/cosmos2b_condE_finetuned \\")
    print(f"    --dry_run")
    print(f"")
    print(f"  # Full training")
    print(f"  nohup python train_cosmos2b.py \\")
    print(f"    --train_file {train_out} \\")
    print(f"    --val_file   {val_out} \\")
    print(f"    --output_dir saves/cosmos2b_condE_finetuned \\")
    print(f"    --epochs 3 --lr 2e-4 --lora_rank 16 \\")
    print(f"    > logs/finetune_condE.log 2>&1 &")
    print(f"")
    print(f"  # Eval — 3 inference conditions on same condE model:")
    print(f"  # 1. BEV only (no camera, no detections)")
    print(f"  python evaluate_zeroshot.py --model saves/cosmos2b_condE_finetuned \\")
    print(f"    --input_level clean_radar_only --condition E ...")
    print(f"  # 2. BEV + camera")
    print(f"  python evaluate_zeroshot.py --model saves/cosmos2b_condE_finetuned \\")
    print(f"    --input_level clean_radar --condition E ...")
    print(f"  # 3. BEV + detections (no camera)")
    print(f"  python evaluate_zeroshot.py --model saves/cosmos2b_condE_finetuned \\")
    print(f"    --input_level clean_radar_det --condition E ...")


if __name__ == '__main__':
    main()
