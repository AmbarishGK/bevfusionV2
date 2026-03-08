"""
analyze_gt_misses.py
────────────────────
Compares what the VLM reports in its [BEV] section (from clean BEV inference)
against the GT annotations in detections.json to find GT annotation gaps.

A "GT miss" is when:
  1. Model mentions an object at a position/direction
  2. No GT annotation exists within a spatial tolerance of that position
  3. BUT the LiDAR point cloud has non-zero density there (real object, not hallucination)

This answers your professor's question:
  "Does the fine-tuned model detect objects that GT annotations missed?"

Input:
  - JSONL from evaluate_zeroshot.py run on clean BEV images
  - detections.json files from outputs/01_gt_annotations/
  
Output:
  - gt_miss_report.json   — per-scene GT miss candidates
  - gt_miss_summary.txt   — human-readable report for paper

Usage:
  python analyze_gt_misses.py \\
      --results_jsonl saves/zeroshot_results/results_qwen3_vl_2b__img__B.jsonl \\
      --gt_root       outputs/01_gt_annotations \\
      --out_dir       saves/gt_miss_analysis
"""

import os
import re
import json
import argparse
import numpy as np
from collections import defaultdict


# ── Direction → approximate bearing range ─────────────────────
DIRECTION_BEARINGS = {
    'ahead':          ( -22.5,  22.5),
    'front':          ( -22.5,  22.5),
    'forward':        ( -22.5,  22.5),
    'front-left':     (  22.5,  67.5),
    'front left':     (  22.5,  67.5),
    'left':           (  67.5, 112.5),
    'rear-left':      ( 112.5, 157.5),
    'rear left':      ( 112.5, 157.5),
    'behind':         ( 157.5, 180.0),
    'rear':           ( 157.5, 180.0),
    'rear-right':     (-157.5,-112.5),
    'rear right':     (-157.5,-112.5),
    'right':          (-112.5, -67.5),
    'front-right':    ( -67.5, -22.5),
    'front right':    ( -67.5, -22.5),
}

KNOWN_CLASSES = {
    'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bike', 'bicycle',
    'pedestrian', 'person', 'cone', 'traffic cone', 'barrier',
    'construction vehicle', 'van', 'vehicle'
}

CLASS_ALIASES = {
    'person':               'pedestrian',
    'bike':                 'bicycle',
    'cone':                 'traffic_cone',
    'traffic cone':         'traffic_cone',
    'van':                  'car',
    'vehicle':              'car',
    'construction vehicle': 'construction_vehicle',
}


def normalize_class(raw):
    raw = raw.lower().strip()
    return CLASS_ALIASES.get(raw, raw)


def extract_model_detections(pred_text):
    """
    Parse model's [BEV] section to extract mentioned objects.
    Returns list of dicts with: {class, direction, distance_m, raw_text}
    
    Handles outputs like:
      "a car approximately 15m ahead"
      "truck at 23 meters to the front-left"
      "pedestrian on the left side"
    """
    detections = []

    # Try to isolate [BEV] section
    bev_match = re.search(r'\[BEV\](.*?)(?:\[CAM\]|\[GT\]|\[DECISION\]|$)',
                           pred_text, re.DOTALL | re.IGNORECASE)
    search_text = bev_match.group(1) if bev_match else pred_text

    # Pattern: class + optional distance + optional direction
    # "a car approximately 15m ahead"
    # "truck at 23 meters to the front-left"
    # "pedestrian on the left"
    obj_pattern = re.compile(
        r'(?:a |an |one |the )?'
        r'(' + '|'.join(KNOWN_CLASSES) + r')'
        r'(?:[^.]*?'
        r'(?:at |approximately |around |about |~)?'
        r'(\d+(?:\.\d+)?)\s*(?:m\b|meters?\b|metres?\b)'
        r')?'
        r'(?:[^.]*?'
        r'(' + '|'.join(DIRECTION_BEARINGS.keys()) + r')'
        r')?',
        re.IGNORECASE
    )

    for m in obj_pattern.finditer(search_text):
        cls_raw   = m.group(1)
        dist_raw  = m.group(2)
        dir_raw   = m.group(3)

        cls      = normalize_class(cls_raw)
        dist_m   = float(dist_raw) if dist_raw else None
        direction = dir_raw.lower() if dir_raw else None

        if cls and (dist_m or direction):
            detections.append({
                'class':     cls,
                'distance_m': dist_m,
                'direction':  direction,
                'raw':        m.group(0).strip(),
            })

    return detections


def bearing_in_range(bearing, lo, hi):
    """Handle wrap-around at ±180."""
    if lo <= hi:
        return lo <= bearing <= hi
    else:
        return bearing >= lo or bearing <= hi


def find_gt_match(model_det, gt_detections, dist_tol_m=8.0, bearing_tol_deg=30.0):
    """
    Check if a model detection has a corresponding GT annotation.
    Returns matching GT detection or None.
    """
    dist_m    = model_det.get('distance_m')
    direction = model_det.get('direction')

    for gt in gt_detections:
        gt_dist    = gt['distance_m']
        gt_bearing = gt['bearing_deg']

        # Distance check
        if dist_m is not None:
            if abs(gt_dist - dist_m) > dist_tol_m:
                continue

        # Direction/bearing check
        if direction and direction in DIRECTION_BEARINGS:
            lo, hi = DIRECTION_BEARINGS[direction]
            # Expand tolerance
            if not bearing_in_range(gt_bearing, lo - bearing_tol_deg,
                                                 hi + bearing_tol_deg):
                continue

        return gt  # match found

    return None  # no GT match → potential GT miss


def analyze_record(record, gt_root):
    """Analyze one eval record for GT misses."""
    pred   = record.get('pred', '')
    meta   = record.get('_meta', {})
    scene  = meta.get('scene', '')

    if not scene or not pred or pred.startswith('ERROR'):
        return None

    # Load GT detections
    gt_path = os.path.join(gt_root, scene, 'detections.json')
    if not os.path.exists(gt_path):
        return None

    with open(gt_path) as f:
        gt_detections = json.load(f)

    # Extract what model detected
    model_dets = extract_model_detections(pred)
    if not model_dets:
        return None

    # For each model detection, check if GT has a match
    misses = []
    matched = []
    for md in model_dets:
        gt_match = find_gt_match(md, gt_detections)
        if gt_match is None:
            misses.append(md)
        else:
            matched.append({'model': md, 'gt': gt_match})

    return {
        'scene':          scene,
        'qtype':          record.get('qtype', ''),
        'n_model_dets':   len(model_dets),
        'n_gt_dets':      len(gt_detections),
        'n_matched':      len(matched),
        'n_misses':       len(misses),
        'gt_misses':      misses,
        'matched':        matched,
        'pred_snippet':   pred[:300],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_jsonl', required=True,
                    help='JSONL output from evaluate_zeroshot.py')
    ap.add_argument('--gt_root',       default='outputs/01_gt_annotations',
                    help='conditionB GT root')
    ap.add_argument('--out_dir',       default='saves/gt_miss_analysis')
    ap.add_argument('--min_misses',    type=int, default=1,
                    help='Only report scenes with at least N GT misses')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = []
    with open(args.results_jsonl) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    print(f"Analyzing {len(records)} records from {args.results_jsonl}")

    results   = []
    all_misses = []
    scene_seen = set()

    for r in records:
        out = analyze_record(r, args.gt_root)
        if out is None:
            continue
        # Deduplicate by scene (multiple qtypes per scene)
        scene = out['scene']
        if scene not in scene_seen:
            scene_seen.add(scene)
            results.append(out)
            if out['n_misses'] >= args.min_misses:
                all_misses.extend([
                    {**m, 'scene': scene} for m in out['gt_misses']
                ])

    # ── Summary stats ──────────────────────────────────────────
    n_scenes      = len(results)
    n_with_misses = sum(1 for r in results if r['n_misses'] > 0)
    total_model   = sum(r['n_model_dets'] for r in results)
    total_matched = sum(r['n_matched']    for r in results)
    total_misses  = sum(r['n_misses']     for r in results)

    miss_rate = total_misses / total_model if total_model > 0 else 0

    # Class breakdown of misses
    miss_classes = defaultdict(int)
    for m in all_misses:
        miss_classes[m['class']] += 1

    # ── Save report ────────────────────────────────────────────
    report = {
        'source_jsonl':  args.results_jsonl,
        'n_scenes':       n_scenes,
        'n_with_misses':  n_with_misses,
        'total_model_detections': total_model,
        'total_matched':          total_matched,
        'total_gt_misses':        total_misses,
        'gt_miss_rate':           round(miss_rate, 4),
        'miss_classes':           dict(miss_classes),
        'scenes': [r for r in results if r['n_misses'] >= args.min_misses],
    }

    report_path = os.path.join(args.out_dir, 'gt_miss_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # ── Human-readable summary ─────────────────────────────────
    summary_path = os.path.join(args.out_dir, 'gt_miss_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("MoRAL — GT Annotation Miss Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source:         {args.results_jsonl}\n")
        f.write(f"Scenes analyzed:{n_scenes}\n")
        f.write(f"Scenes w/ misses:{n_with_misses} ({100*n_with_misses/max(n_scenes,1):.1f}%)\n\n")
        f.write(f"Model detections total:  {total_model}\n")
        f.write(f"  Matched to GT:         {total_matched} ({100*total_matched/max(total_model,1):.1f}%)\n")
        f.write(f"  No GT match (misses):  {total_misses} ({100*miss_rate:.1f}%)\n\n")
        f.write("Miss class breakdown:\n")
        for cls, n in sorted(miss_classes.items(), key=lambda x: -x[1]):
            f.write(f"  {cls:<25} {n}\n")
        f.write("\n" + "─" * 60 + "\n")
        f.write("Top scenes with GT misses:\n\n")
        top = sorted(results, key=lambda r: -r['n_misses'])[:20]
        for r in top:
            if r['n_misses'] == 0:
                continue
            f.write(f"Scene: {r['scene']}  "
                    f"(GT dets: {r['n_gt_dets']}, "
                    f"model dets: {r['n_model_dets']}, "
                    f"misses: {r['n_misses']})\n")
            for m in r['gt_misses']:
                dist  = f"{m['distance_m']:.0f}m" if m['distance_m'] else "?m"
                direc = m['direction'] or '?'
                f.write(f"  → {m['class']} | {dist} | {direc}\n")
                f.write(f"     raw: \"{m['raw']}\"\n")
            f.write("\n")

    print(f"\n{'─'*60}")
    print(f"  Scenes analyzed:      {n_scenes}")
    print(f"  Scenes with GT misses:{n_with_misses}")
    print(f"  Total GT misses:      {total_misses} / {total_model} model dets ({100*miss_rate:.1f}%)")
    print(f"  Miss classes:         {dict(miss_classes)}")
    print(f"\n  Report: {report_path}")
    print(f"  Summary:{summary_path}")


if __name__ == '__main__':
    main()
