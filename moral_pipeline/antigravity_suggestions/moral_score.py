#!/usr/bin/env python3
"""
MoRAL-Score Compute + Novel Object Detection + HIL Sample Generator
===================================================================

Computes the full MoRAL-Score metric (GSA, RSQ, ACT, SEN, NOV) on result files,
detects novel objects not in GT, and exports samples for HIL evaluation.

USAGE:
    # Compute MoRAL-Score (auto metrics only, no API needed):
    python moral_score.py \
        --results_dir /path/to/saves/zeroshot_results \
        --detections_dir /path/to/detections \
        --output_dir saves/moral_score_results

    # With LLM judge for SEN scoring:
    python moral_score.py \
        --results_dir /path/to/saves/zeroshot_results \
        --provider anthropic --model claude-sonnet-4-20250514

    # Generate HIL evaluation samples:
    python moral_score.py \
        --results_dir /path/to/saves/zeroshot_results \
                      /path/to/saves/finetuned_results \
        --hil --hil_per_qtype 2

    # Compare multiple runs:
    python moral_score.py \
        --results_dir /path/to/saves/zeroshot_results \
                      /path/to/saves/finetuned_results \
        --compare
"""

import argparse
import json
import os
import re
import sys
import csv
import time
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# S₁: GSA — Grounded Spatial Accuracy
# ═══════════════════════════════════════════════════════════════════════════════

# Per-qtype thresholds: (exact, good, partial)
GSA_THRESHOLDS = {
    "spatial":        {"field": "distance_m",           "exact": 0.10, "good": 0.20, "partial": 0.50, "type": "pct"},
    "safety":         {"field": "ttc_s",                "exact": 0.15, "good": 0.30, "partial": 0.50, "type": "pct"},
    "velocity":       {"field": "velocity_ms",          "exact": 0.15, "good": 0.30, "partial": 0.50, "type": "pct"},
    "physics":        {"field": "nearest_ahead_m",      "exact": 0.10, "good": 0.20, "partial": 0.50, "type": "pct"},
    "counterfactual": {"field": "t_impact_s",           "exact": 0.20, "good": 0.40, "partial": 0.60, "type": "pct"},
    "planning":       {"field": "min_ttc_s",            "exact": 0.15, "good": 0.30, "partial": 0.50, "type": "pct"},
    "ethical":        {"field": "min_dilemma_ttc_s",    "exact": 0.20, "good": 0.40, "partial": None, "type": "pct"},
    "multi_conflict": {"field": "top_risk_score",       "exact": 0.15, "good": 0.30, "partial": 0.50, "type": "pct"},
    "trajectory":     {"field": "objects_entering_path","exact": 0,    "good": 1,    "partial": 2,    "type": "abs"},
    "near_miss":      {"field": "near_miss_count",      "exact": 0,    "good": 1,    "partial": 2,    "type": "abs"},
    "sensor_limit":   {"field": "gt_estimated_count",   "exact": 0,    "good": 1,    "partial": 2,    "type": "abs"},
}

# Qualitative qtypes (no numeric GT — scored by LLM or HIL only)
QUALITATIVE_QTYPES = {"occlusion", "gap", "zone"}

def extract_number_from_text(text: str, gt_field: str) -> Optional[float]:
    """Extract a numeric value from model output relevant to the GT field."""
    # Try field-specific patterns first
    field_hints = {
        "distance_m": [r"(\d+\.?\d*)\s*(?:m\b|meter)", r"(?:distance|far|away|ahead|behind).*?(\d+\.?\d*)",
                       r"(\d+\.?\d*)\s*m\b"],
        "ttc_s": [r"(?:TTC|time.to.collision|reach|impact).*?(\d+\.?\d*)\s*(?:s|sec|second)",
                  r"(\d+\.?\d*)\s*(?:s|sec|second)"],
        "velocity_ms": [r"(\d+\.?\d*)\s*(?:m/?s|meters?\s*per\s*second)",
                        r"(?:speed|velocity|moving).*?(\d+\.?\d*)"],
        "nearest_ahead_m": [r"(\d+\.?\d*)\s*(?:m\b|meter)", r"(?:nearest|closest|ahead).*?(\d+\.?\d*)"],
        "objects_entering_path": [r"(\d+)\s*(?:object|vehicle|car|pedestrian)",
                                  r"(?:enter|cross|intersect).*?(\d+)"],
        "near_miss_count": [r"(\d+)\s*(?:near.miss|close.call)", r"(?:near.miss|close).*?(\d+)"],
        "gt_estimated_count": [r"(\d+)\s*(?:object|detection|vehicle)"],
    }

    patterns = field_hints.get(gt_field, [])
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue

    # Fallback: find reasonable numbers
    numbers = re.findall(r"(\d+\.?\d+)", text)
    for n in numbers:
        val = float(n)
        if 0.1 < val < 500:
            return val

    return None


def compute_gsa(sample: dict) -> Optional[float]:
    """Compute GSA sub-score for a single sample. Returns 0-100 or None."""
    qtype = sample.get("qtype", "")
    gt_value = sample.get("gt_value")
    pred_text = sample.get("pred", "")

    if qtype in QUALITATIVE_QTYPES:
        return None  # Must be scored by LLM/HIL

    if gt_value is None:
        return None

    config = GSA_THRESHOLDS.get(qtype)
    if not config:
        return None

    pred_val = extract_number_from_text(pred_text, config["field"])
    if pred_val is None:
        return 0.0  # Model didn't produce a number

    if config["type"] == "pct":
        # Percentage-based threshold
        denom = max(0.01, abs(gt_value))
        error = abs(pred_val - gt_value) / denom
        if error <= config["exact"]:
            return 100.0
        elif error <= config["good"]:
            return 75.0
        elif config.get("partial") and error <= config["partial"]:
            return 40.0
        else:
            return 0.0
    else:
        # Absolute threshold (integer counts)
        error = abs(pred_val - gt_value)
        if error <= config["exact"]:
            return 100.0
        elif error <= config["good"]:
            return 75.0
        elif config.get("partial") and error <= config["partial"]:
            return 40.0
        else:
            return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# S₂: RSQ — Reasoning Structure Quality
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rsq(sample: dict) -> float:
    """Compute RSQ sub-score. Returns 0-100."""
    pred = sample.get("pred", "")
    score = 0

    # Check for structured sections
    has_bev = bool(re.search(r'\[BEV\]', pred))
    has_cam = bool(re.search(r'\[CAM\]', pred))
    has_gt = bool(re.search(r'\[GT\]', pred))
    has_decision = bool(re.search(r'\[DECISION\]', pred))

    if has_bev:
        score += 25
    if has_cam:
        score += 25
    if has_gt:
        score += 25
        # Bonus: GT section has concrete numbers
        gt_section = ""
        if has_decision:
            parts = re.split(r'\[DECISION\]', re.split(r'\[GT\]', pred)[-1])
            gt_section = parts[0] if parts else ""
        else:
            gt_section = re.split(r'\[GT\]', pred)[-1]
        if re.search(r'\d+\.?\d*\s*m', gt_section):
            score += 5
    if has_decision:
        score += 25
        # Bonus: decision references TTC
        dec_section = re.split(r'\[DECISION\]', pred)[-1] if has_decision else ""
        if re.search(r'TTC|time.to.collision', dec_section, re.IGNORECASE):
            score += 5

    return min(100.0, float(score))


# ═══════════════════════════════════════════════════════════════════════════════
# S₃: ACT — Action Decision Accuracy
# ═══════════════════════════════════════════════════════════════════════════════

ACTION_FAMILIES = {
    "BRAKE":      {"BRAKE", "EMERGENCY_BRAKE", "HARD_BRAKE", "STOP"},
    "MAINTAIN":   {"MAINTAIN", "CONTINUE", "CRUISE"},
    "ACCELERATE": {"ACCELERATE", "SPEED_UP"},
    "YIELD":      {"YIELD", "WAIT", "SLOW_DOWN"},
    "EVADE":      {"SWERVE", "LANE_CHANGE", "MERGE"},
}

# Broader: brake-like vs go-like
ACTION_DIRECTIONS = {
    "DECELERATE": {"BRAKE", "EMERGENCY_BRAKE", "HARD_BRAKE", "STOP", "YIELD", "WAIT", "SLOW_DOWN"},
    "PROCEED":    {"MAINTAIN", "CONTINUE", "CRUISE", "ACCELERATE", "SPEED_UP"},
    "MANEUVER":   {"SWERVE", "LANE_CHANGE", "MERGE"},
}

def compute_act(sample: dict) -> Optional[float]:
    """Compute ACT sub-score. Returns 0-100 or None if no GT action."""
    gt = sample.get("gt_action", "")
    pred = sample.get("pred_action", "")

    if not gt:
        return None  # No GT action for this qtype

    if not pred:
        return 0.0

    gt_upper = gt.upper().strip()
    pred_upper = pred.upper().strip()

    # Exact match
    if gt_upper == pred_upper:
        return 100.0

    # Family match
    gt_fam = None
    pred_fam = None
    for fam, members in ACTION_FAMILIES.items():
        if gt_upper in members:
            gt_fam = fam
        if pred_upper in members:
            pred_fam = fam
    if gt_fam and gt_fam == pred_fam:
        return 75.0

    # Direction match
    gt_dir = None
    pred_dir = None
    for direction, members in ACTION_DIRECTIONS.items():
        if gt_upper in members:
            gt_dir = direction
        if pred_upper in members:
            pred_dir = direction
    if gt_dir and gt_dir == pred_dir:
        return 40.0

    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# S₅: NOV — Novel Detection Capability
# ═══════════════════════════════════════════════════════════════════════════════

OBJECT_CLASSES = {
    'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle',
    'pedestrian', 'traffic_cone', 'barrier', 'construction_vehicle',
    'vehicle', 'cyclist', 'motorbike', 'van', 'person',
}

# Synonyms → canonical
CLASS_SYNONYMS = {
    'vehicle': 'car', 'van': 'car', 'sedan': 'car', 'suv': 'car',
    'person': 'pedestrian', 'walker': 'pedestrian', 'man': 'pedestrian',
    'woman': 'pedestrian', 'child': 'pedestrian',
    'cyclist': 'bicycle', 'biker': 'bicycle', 'bike': 'bicycle',
    'motorbike': 'motorcycle', 'scooter': 'motorcycle',
    'cone': 'traffic_cone',
}

def canonicalize_class(cls: str) -> str:
    """Normalize class name to canonical form."""
    cls = cls.lower().strip()
    return CLASS_SYNONYMS.get(cls, cls)


def extract_mentioned_objects(pred_text: str) -> list:
    """Extract objects mentioned in model output with approximate distances."""
    mentions = []
    seen = set()

    for cls in OBJECT_CLASSES:
        # Pattern: "a <class> at/approximately <N> meters"
        patterns = [
            rf'({cls})\s+(?:at|approximately|about|roughly|within|is)\s+(\d+\.?\d*)\s*(?:m\b|meter)',
            rf'({cls}).*?(\d+\.?\d*)\s*(?:m\b|meter)',
            rf'(?:nearest|closest)\s+({cls}).*?(\d+\.?\d*)',
        ]
        for pat in patterns:
            for m in re.finditer(pat, pred_text, re.IGNORECASE):
                canon = canonicalize_class(m.group(1))
                dist = float(m.group(2))
                key = f"{canon}_{dist:.0f}"
                if key not in seen and 0.5 < dist < 200:
                    seen.add(key)
                    mentions.append({
                        'class': canon,
                        'distance_m': dist,
                        'text_span': m.group(0)[:100],
                    })

    return mentions


def find_novel_detections(mentioned: list, gt_detections: list,
                          match_radius: float = 5.0) -> dict:
    """
    Compare model-mentioned objects against GT detections.
    Returns dict with 'matched', 'novel', 'gt_missed'.
    """
    matched = []
    novel = []
    gt_matched_flags = [False] * len(gt_detections)

    for mention in mentioned:
        best_match = None
        best_dist = float('inf')

        for i, gt in enumerate(gt_detections):
            gt_cls = canonicalize_class(gt.get('category', gt.get('class', '')))
            if canonicalize_class(mention['class']) != gt_cls:
                continue
            d = abs(mention['distance_m'] - gt.get('distance_m', float('inf')))
            if d < match_radius and d < best_dist:
                best_match = i
                best_dist = d

        if best_match is not None:
            matched.append({**mention, 'gt_index': best_match, 'match_error': best_dist})
            gt_matched_flags[best_match] = True
        else:
            novel.append(mention)

    gt_missed = [gt for i, gt in enumerate(gt_detections) if not gt_matched_flags[i]]

    return {
        'matched': matched,
        'novel': novel,
        'gt_missed': gt_missed,
        'n_matched': len(matched),
        'n_novel': len(novel),
        'n_gt_missed': len(gt_missed),
    }


def compute_nov(sample: dict, gt_detections: list = None) -> dict:
    """
    Compute NOV sub-score.
    Returns dict with score (0-100) and novel detection details.
    """
    pred_text = sample.get("pred", "")
    mentioned = extract_mentioned_objects(pred_text)

    result = {
        'mentioned_objects': mentioned,
        'n_mentioned': len(mentioned),
        'novel_detections': [],
        'novel_score': 50.0,  # Neutral default
    }

    if not mentioned:
        return result

    if gt_detections:
        detection_result = find_novel_detections(mentioned, gt_detections)
        result.update(detection_result)

        # Score: +20 per novel, capped at 100
        n_novel = detection_result['n_novel']
        if n_novel > 0:
            result['novel_score'] = min(100.0, 50.0 + n_novel * 20.0)
            result['novel_detections'] = detection_result['novel']
        else:
            result['novel_score'] = 50.0  # Neutral — everything matched GT
    else:
        # Without GT detections, we can still count mentioned objects
        result['novel_score'] = 50.0  # Can't verify without GT

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Combined MoRAL-Score
# ═══════════════════════════════════════════════════════════════════════════════

WEIGHTS = {
    "gsa": 0.25,
    "rsq": 0.20,
    "act": 0.20,
    "sen": 0.20,
    "nov": 0.15,
}

def compute_moral_score(sample: dict, gt_detections: list = None,
                        sen_score: float = None) -> dict:
    """
    Compute full MoRAL-Score for a single sample.
    
    Args:
        sample: Result dict from evaluation JSONL
        gt_detections: Optional list of GT detection dicts for NOV scoring
        sen_score: Optional SEN score from LLM judge (0-100)
    
    Returns:
        Dict with all sub-scores and combined MoRAL-Score
    """
    gsa = compute_gsa(sample)
    rsq = compute_rsq(sample)
    act = compute_act(sample)
    nov_result = compute_nov(sample, gt_detections)
    nov = nov_result['novel_score']

    # SEN defaults to 50 if no LLM judge
    sen = sen_score if sen_score is not None else 50.0

    # Combine with weights (skip None scores)
    scores = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for name, val, weight in [("gsa", gsa, WEIGHTS["gsa"]),
                               ("rsq", rsq, WEIGHTS["rsq"]),
                               ("act", act, WEIGHTS["act"]),
                               ("sen", sen, WEIGHTS["sen"]),
                               ("nov", nov, WEIGHTS["nov"])]:
        scores[name] = val
        if val is not None:
            weighted_sum += val * weight
            total_weight += weight

    moral_score = weighted_sum / max(0.01, total_weight)

    return {
        "moral_score": round(moral_score, 1),
        "gsa": gsa,
        "rsq": rsq,
        "act": act,
        "sen": sen,
        "nov": nov,
        "nov_details": {
            "n_mentioned": nov_result['n_mentioned'],
            "n_novel": len(nov_result.get('novel_detections', [])),
            "novel_objects": nov_result.get('novel_detections', []),
        },
        "weights_used": total_weight,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_moral_scores(scored_samples: list) -> dict:
    """Aggregate per-sample MoRAL-Scores into summary statistics."""
    summary = {"total": len(scored_samples)}

    for key in ["moral_score", "gsa", "rsq", "act", "sen", "nov"]:
        vals = [s["scores"][key] for s in scored_samples
                if s["scores"].get(key) is not None]
        if vals:
            summary[key] = {
                "mean": round(sum(vals) / len(vals), 1),
                "median": round(sorted(vals)[len(vals) // 2], 1),
                "min": round(min(vals), 1),
                "max": round(max(vals), 1),
                "n": len(vals),
            }

    # Per question type
    by_qtype = defaultdict(list)
    for s in scored_samples:
        by_qtype[s.get("qtype", "unknown")].append(s)

    summary["by_qtype"] = {}
    for qtype, samples in sorted(by_qtype.items()):
        ms_vals = [s["scores"]["moral_score"] for s in samples
                   if s["scores"].get("moral_score") is not None]
        summary["by_qtype"][qtype] = {
            "n": len(samples),
            "moral_score_mean": round(sum(ms_vals) / len(ms_vals), 1) if ms_vals else None,
        }
        for key in ["gsa", "rsq", "act"]:
            vals = [s["scores"][key] for s in samples if s["scores"].get(key) is not None]
            if vals:
                summary["by_qtype"][qtype][f"{key}_mean"] = round(sum(vals) / len(vals), 1)

    # Novel detection summary
    all_novel = []
    for s in scored_samples:
        novels = s["scores"].get("nov_details", {}).get("novel_objects", [])
        for n in novels:
            all_novel.append({**n, "scene": s.get("scene"), "qtype": s.get("qtype")})

    summary["novel_detections"] = {
        "total_novel": len(all_novel),
        "unique_classes": list(set(n["class"] for n in all_novel)),
        "examples": all_novel[:10],
    }

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# HIL Sample Export
# ═══════════════════════════════════════════════════════════════════════════════

def generate_hil_samples(scored_samples: list, n_per_qtype: int = 2,
                         output_path: Path = None) -> list:
    """
    Generate stratified HIL evaluation samples.
    Exports CSV + JSON for human evaluators.
    """
    by_qtype = defaultdict(list)
    for s in scored_samples:
        by_qtype[s.get("qtype", "unknown")].append(s)

    selected = []
    for qtype, samples in sorted(by_qtype.items()):
        # Pick n_per_qtype with diverse scenes
        random.seed(42)
        n = min(n_per_qtype, len(samples))
        picked = random.sample(samples, n)
        selected.extend(picked)

    random.shuffle(selected)

    # Assign blind IDs (so human doesn't know which model)
    for i, s in enumerate(selected):
        s["hil_id"] = f"EVAL_{i+1:03d}"

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

        # JSON for programmatic use
        hil_json = output_path / "hil_samples.json"
        with open(hil_json, "w") as f:
            json.dump(selected, f, indent=2, default=str)

        # CSV for spreadsheet
        hil_csv = output_path / "hil_evaluation_sheet.csv"
        with open(hil_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "HIL_ID", "Scene", "Question_Type",
                "Model_Output_Preview",
                "GT_Value", "GT_Field", "GT_Action",
                "Auto_MoRAL_Score", "Auto_GSA", "Auto_RSQ",
                "Human_Spatial_Accuracy_1to5",
                "Human_Reasoning_Quality_1to5",
                "Human_Action_Correctness_1to5",
                "Human_Sensor_Grounding_1to5",
                "Human_Physical_Plausibility_1to5",
                "Human_Novel_Observations_1to5",
                "Novel_Object_Not_In_GT_YesNo",
                "Novel_Object_Description",
                "Human_Notes",
            ])
            for s in selected:
                scores = s.get("scores", {})
                writer.writerow([
                    s.get("hil_id", ""),
                    s.get("scene", ""),
                    s.get("qtype", ""),
                    s.get("pred", "")[:200],
                    s.get("gt_value", ""),
                    s.get("gt_field", ""),
                    s.get("gt_action", ""),
                    scores.get("moral_score", ""),
                    scores.get("gsa", ""),
                    scores.get("rsq", ""),
                    "", "", "", "", "", "",  # Human fills these in
                    "", "", "",
                ])

        # Full text sheet for reading
        hil_txt = output_path / "hil_full_evaluation.txt"
        with open(hil_txt, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MoRAL Human-in-the-Loop Evaluation Sheet\n")
            f.write(f"Total samples: {len(selected)}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n")
            f.write("=" * 80 + "\n\n")
            f.write("Instructions: For each sample, rate dimensions 1-5.\n")
            f.write("1=Poor, 2=Below average, 3=Acceptable, 4=Good, 5=Excellent\n")
            f.write("Mark N/A if dimension doesn't apply.\n\n")

            for s in selected:
                scores = s.get("scores", {})
                f.write("─" * 80 + "\n")
                f.write(f"ID: {s.get('hil_id')}  |  Scene: {s.get('scene')}  |  "
                        f"Type: {s.get('qtype')}\n")
                f.write(f"Auto MoRAL-Score: {scores.get('moral_score', '?')}/100  |  "
                        f"GSA: {scores.get('gsa', 'N/A')}  |  RSQ: {scores.get('rsq', '?')}\n")
                f.write("─" * 80 + "\n\n")

                f.write("GROUND TRUTH:\n")
                if s.get("gt_value") is not None:
                    f.write(f"  Value: {s['gt_value']} ({s.get('gt_field', '?')})\n")
                if s.get("gt_action"):
                    f.write(f"  Action: {s['gt_action']}\n")
                if s.get("gt_ttc") is not None:
                    f.write(f"  TTC: {s['gt_ttc']}s\n")
                f.write("\n")

                f.write("MODEL OUTPUT:\n")
                f.write(f"  {s.get('pred', '(no output)')}\n\n")

                f.write("RATE (1-5 or N/A):\n")
                f.write("  [ ] Spatial Accuracy:      ___\n")
                f.write("  [ ] Reasoning Quality:     ___\n")
                f.write("  [ ] Action Correctness:    ___\n")
                f.write("  [ ] Sensor Grounding:      ___\n")
                f.write("  [ ] Physical Plausibility: ___\n")
                f.write("  [ ] Novel Observations:    ___\n\n")
                f.write("  Novel object NOT in GT?  [ ] Yes  [ ] No\n")
                f.write("  If yes, describe: ________________________________\n\n")
                f.write("  Notes: ____________________________________________\n\n\n")

        print(f"  HIL exports:")
        print(f"    JSON:  {hil_json}")
        print(f"    CSV:   {hil_csv} (open in Google Sheets/Excel)")
        print(f"    TXT:   {hil_txt} (printable evaluation sheet)")

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="MoRAL-Score: Compute multi-dimensional VLM reasoning metric",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", nargs="+", default=None)
    parser.add_argument("--results_file", default=None)
    parser.add_argument("--detections_dir", default=None,
                        help="Dir with detections.json per scene (for NOV scoring)")
    parser.add_argument("--output_dir", default="saves/moral_score_results")

    parser.add_argument("--hil", action="store_true", help="Generate HIL samples")
    parser.add_argument("--hil_per_qtype", type=int, default=2,
                        help="Samples per question type for HIL")

    parser.add_argument("--compare", action="store_true",
                        help="Generate comparison table")
    parser.add_argument("--max_samples", type=int, default=None)

    return parser.parse_args()


def find_files(args) -> list:
    files = []
    if args.results_file:
        files.append(Path(args.results_file))
    if args.results_dir:
        for d in args.results_dir:
            p = Path(d)
            if p.is_dir():
                files.extend(sorted(p.glob("results_*.jsonl")))
            elif p.is_file():
                files.append(p)
    return files


def load_detections(det_dir: str, scene: str) -> list:
    """Load GT detections for a scene if available."""
    if not det_dir:
        return []
    det_path = Path(det_dir) / scene / "detections.json"
    if det_path.exists():
        with open(det_path) as f:
            return json.load(f)
    return []


def process_file(filepath: Path, args) -> tuple:
    """Process a single results JSONL and compute MoRAL-Scores."""
    run_id = filepath.stem
    if run_id.startswith("results_"):
        run_id = run_id[len("results_"):]

    print(f"\n{'═' * 60}")
    print(f"  MoRAL-Score: {run_id}")
    print(f"{'═' * 60}")

    samples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"  Loaded {len(samples)} samples")

    scored = []
    for sample in samples:
        gt_dets = load_detections(args.detections_dir, sample.get("scene", ""))
        scores = compute_moral_score(sample, gt_detections=gt_dets)
        sample["scores"] = scores
        scored.append(sample)

    summary = aggregate_moral_scores(scored)
    summary["run_id"] = run_id
    summary["source_file"] = str(filepath)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / f"scored_{run_id}.jsonl"
    with open(out_jsonl, "w") as f:
        for s in scored:
            f.write(json.dumps(s, default=str) + "\n")

    out_summary = out_dir / f"moral_score_summary_{run_id}.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print
    print(f"\n  ── MoRAL-Score Summary: {run_id} ──")
    for key in ["moral_score", "gsa", "rsq", "act", "nov"]:
        s = summary.get(key, {})
        if isinstance(s, dict) and s.get("mean") is not None:
            print(f"    {key.upper():<14} {s['mean']:>5.1f}/100  "
                  f"(min={s['min']:.0f}, max={s['max']:.0f}, n={s['n']})")

    print(f"\n    By question type:")
    for qtype, qs in sorted(summary.get("by_qtype", {}).items()):
        ms = qs.get("moral_score_mean", "?")
        gsa = qs.get("gsa_mean", "—")
        rsq = qs.get("rsq_mean", "—")
        print(f"      {qtype:<18} MoRAL={ms:>5}  GSA={gsa!s:>5}  RSQ={rsq!s:>5}  n={qs['n']}")

    nd = summary.get("novel_detections", {})
    if nd.get("total_novel", 0) > 0:
        print(f"\n    🔍 Novel detections (not in GT): {nd['total_novel']}")
        print(f"       Classes: {', '.join(nd.get('unique_classes', []))}")
        for ex in nd.get("examples", [])[:5]:
            print(f"       • {ex.get('class')} at {ex.get('distance_m')}m "
                  f"[{ex.get('scene')}] — \"{ex.get('text_span', '')[:60]}\"")

    print(f"\n  Saved: {out_jsonl}")
    print(f"  Saved: {out_summary}")

    return run_id, summary, scored


def generate_comparison(all_summaries: dict, output_dir: Path):
    """Generate markdown comparison table."""
    lines = []
    lines.append("# MoRAL-Score Comparison Table\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n")

    runs = sorted(all_summaries.keys())

    # Main table
    header = "| Metric | " + " | ".join(r[:30] for r in runs) + " |"
    sep = "|--------|" + "|".join(["-------"] * len(runs)) + "|"
    lines.append(header)
    lines.append(sep)

    for metric in ["moral_score", "gsa", "rsq", "act", "sen", "nov"]:
        row = f"| **{metric.upper()}** |"
        for run in runs:
            s = all_summaries[run].get(metric, {})
            if isinstance(s, dict) and s.get("mean") is not None:
                row += f" {s['mean']:.1f} |"
            else:
                row += " — |"
        lines.append(row)

    # Per-qtype
    all_qtypes = set()
    for s in all_summaries.values():
        all_qtypes.update(s.get("by_qtype", {}).keys())

    lines.append("\n### Per Question Type — MoRAL-Score\n")
    header2 = "| QType | " + " | ".join(r[:30] for r in runs) + " |"
    lines.append(header2)
    lines.append(sep)
    for qtype in sorted(all_qtypes):
        row = f"| {qtype} |"
        for run in runs:
            qt = all_summaries[run].get("by_qtype", {}).get(qtype, {})
            ms = qt.get("moral_score_mean")
            if ms is not None:
                row += f" {ms:.1f} |"
            else:
                row += " — |"
        lines.append(row)

    table_path = output_dir / "moral_score_comparison.md"
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  📊 Comparison table: {table_path}")


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print(" MoRAL-Score Evaluator")
    print("=" * 60)

    files = find_files(args)
    if not files:
        print("❌ No results files found")
        sys.exit(1)

    print(f"  Found {len(files)} results file(s)")

    all_summaries = {}
    all_scored = []

    for filepath in files:
        run_id, summary, scored = process_file(filepath, args)
        all_summaries[run_id] = summary
        all_scored.extend(scored)

    out_dir = Path(args.output_dir)

    if args.compare and len(all_summaries) > 1:
        generate_comparison(all_summaries, out_dir)

    if args.hil:
        print(f"\n{'═' * 60}")
        print(f"  Generating HIL evaluation samples")
        print(f"{'═' * 60}")
        generate_hil_samples(all_scored, n_per_qtype=args.hil_per_qtype,
                             output_path=out_dir / "hil")

    print(f"\n{'═' * 60}")
    print(f" ✅ Done — {len(all_summaries)} run(s) scored")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
