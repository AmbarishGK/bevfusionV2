#!/usr/bin/env python3
"""
moral_score.py — MoRAL Evaluation Pipeline
Reads all result JSONL files and produces:
  - Per-experiment SVA / ASA / SER / MoRAL-Score
  - SGC sensor contribution heatmap
  - Per-qtype breakdown
  - 2B vs 8B, zero-shot vs fine-tuned comparisons
  - Mode collapse (unique prediction %) audit
  - Tables 1, 2, 3 → saves/moral_score_results/all_tables.md

Field names confirmed from actual results files:
  idx, scene, qtype, gt_action, pred_action, gt_ttc, pred_ttc,
  gt_value, gt_field, bleu, quality_tags, pred

Usage:
  python3 moral_score.py                          # run all experiments
  python3 moral_score.py --experiment img__B      # single experiment
  python3 moral_score.py --tables-only            # skip scoring, regenerate tables
  python3 moral_score.py --verbose                # print per-experiment details
"""

import re
import json
import glob
import argparse
import os
from pathlib import Path
from collections import defaultdict

# ─── Paths ──────────────────────────────────────────────────────────────────
PIPELINE_ROOT = Path(os.environ.get("MORAL_ROOT", Path.home() / "workspace/amb_ws/bevfusionV2/moral_pipeline"))
ZS_DIR   = PIPELINE_ROOT / "saves/zeroshot_results"
ZS_8B    = PIPELINE_ROOT / "saves/zeroshot_results_8B"
FT_DIR   = PIPELINE_ROOT / "saves/finetuned_results"
FT_AWS   = PIPELINE_ROOT / "saves/finetuned_results_aws"
FT_8B    = PIPELINE_ROOT / "saves/finetuned_results_8b"
OUT_DIR  = PIPELINE_ROOT / "saves/moral_score_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Question-type routing ───────────────────────────────────────────────────
SVA_QTYPES = {"spatial", "velocity", "physics", "near_miss", "counterfactual"}
ASA_QTYPES = {"safety", "gap", "planning", "zone", "ethical"}
SER_SAFETY_CRITICAL = {"safety", "gap", "planning"}   # must have N >= 30 for valid SER

# ─── Action family matching ──────────────────────────────────────────────────
ACTION_FAMILIES = {
    "EMERGENCY_BRAKE": {"EMERGENCY_BRAKE", "BRAKE", "STOP"},
    "BRAKE":           {"BRAKE", "EMERGENCY_BRAKE", "STOP"},
    "YIELD":           {"YIELD", "BRAKE"},
    "MAINTAIN":        {"MAINTAIN"},
    "STOP":            {"STOP", "BRAKE", "EMERGENCY_BRAKE"},
}

# SER asymmetric cost weights: (gt_action → pred_action) = cost
SER_COSTS = {
    ("STOP",            "MAINTAIN"): 1.0,
    ("STOP",            "YIELD"):    0.5,
    ("EMERGENCY_BRAKE", "MAINTAIN"): 1.0,
    ("EMERGENCY_BRAKE", "YIELD"):    0.5,
    ("BRAKE",           "MAINTAIN"): 0.7,
    ("BRAKE",           "YIELD"):    0.3,
    ("YIELD",           "MAINTAIN"): 0.5,
}

# ─── Composite weights ───────────────────────────────────────────────────────
W_SVA = 0.35
W_ASA = 0.30
W_RQS = 0.20
W_SER = 0.15

# ─── Degraded modalities (SER computed but NOT in composite) ─────────────────
DEGRADED_MODALITIES = {"lidar_only", "radar_only", "cam_only", "bev_only",
                       "clean_lidar_only", "clean_radar_only"}
FULL_SENSOR_MODALITIES = {"clean_lidar", "clean_radar", "img", "img+det"}

# ─── Regex patterns ──────────────────────────────────────────────────────────
NUM_PATTERN   = re.compile(r"([0-9]+\.?[0-9]*)\s*(?:m(?:eters?|etres?)?|km/h|m/s|s(?:econds?)?)?", re.IGNORECASE)
ACTION_PATTERN = re.compile(r"\b(EMERGENCY_BRAKE|BRAKE|YIELD|MAINTAIN|STOP)\b", re.IGNORECASE)
TAG_PATTERN    = re.compile(r"\[(BEV|CAM|GT|DECISION)\]")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def action_family_match(pred: str, gt: str) -> bool:
    """True if pred falls within the allowed family for gt."""
    if not pred or not gt:
        return False
    pred_up = pred.strip().upper()
    gt_up   = gt.strip().upper()
    family  = ACTION_FAMILIES.get(gt_up, {gt_up})
    return pred_up in family


def ser_cost(gt_action: str, pred_action: str) -> float:
    """Return SER cost for a safety-critical sample. 0 if action is safe/correct."""
    if not gt_action or not pred_action:
        return 0.0
    gt_up   = gt_action.strip().upper()
    pred_up = pred_action.strip().upper()
    # Only penalise conservative→permissive errors
    return SER_COSTS.get((gt_up, pred_up), 0.0)


def extract_numeric(text: str) -> float | None:
    """Extract first numeric value from prediction text."""
    if not text:
        return None
    m = NUM_PATTERN.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def sva_match(pred_text: str, gt_value: float | None, tolerance: float = 0.20) -> bool:
    """True if extracted numeric prediction is within ±tolerance of gt_value."""
    if gt_value is None:
        return False
    pred_val = extract_numeric(pred_text)
    if pred_val is None:
        return False
    if gt_value == 0:
        return pred_val == 0
    return abs(pred_val - gt_value) / abs(gt_value) <= tolerance


def quality_tag_coverage(tags_list: list) -> dict:
    """Return dict of which RSQ tags were present."""
    tags = set(tags_list) if tags_list else set()
    return {
        "BEV":      "[BEV]"      in tags,
        "CAM":      "[CAM]"      in tags,
        "GT":       "[GT]"       in tags,
        "DECISION": "[DECISION]" in tags,
        "all_four": all(t in tags for t in ["[BEV]", "[CAM]", "[GT]", "[DECISION]"]),
    }


def unique_prediction_rate(records: list) -> float:
    """Mode collapse indicator: fraction of unique pred_action values."""
    actions = [r.get("pred_action", "") for r in records if r.get("pred_action")]
    if not actions:
        return 0.0
    return len(set(actions)) / max(len(actions), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Core scoring
# ═══════════════════════════════════════════════════════════════════════════════

def score_experiment(records: list, modality: str = "", verbose: bool = False) -> dict:
    """
    Compute all Layer 1+2 metrics for a list of JSONL records.
    Returns a results dict.
    """
    sva_num = sva_den = 0
    asa_num = asa_den = 0
    ser_full_sum = ser_full_den = 0
    ser_deg_sum  = ser_deg_den  = 0
    bleu_scores  = []
    tag_all_four = tag_total = 0
    qtype_sva    = defaultdict(lambda: [0, 0])   # [correct, total]
    qtype_asa    = defaultdict(lambda: [0, 0])

    is_degraded = any(m in modality for m in DEGRADED_MODALITIES)
    is_full     = any(m in modality for m in FULL_SENSOR_MODALITIES) and not is_degraded

    for r in records:
        qtype    = (r.get("qtype") or "").lower().strip()
        pred     = r.get("pred", "") or ""
        gt_val   = r.get("gt_value")
        gt_act   = r.get("gt_action", "") or ""
        pred_act = r.get("pred_action", "") or ""
        bleu     = r.get("bleu")
        tags     = r.get("quality_tags") or []

        # BLEU
        if bleu is not None:
            bleu_scores.append(float(bleu))

        # Quality tags
        cov = quality_tag_coverage(tags)
        tag_total += 1
        if cov["all_four"]:
            tag_all_four += 1

        # SVA
        if qtype in SVA_QTYPES:
            hit = sva_match(pred, gt_val)
            sva_num += int(hit)
            sva_den += 1
            qtype_sva[qtype][0] += int(hit)
            qtype_sva[qtype][1] += 1

        # ASA
        if qtype in ASA_QTYPES:
            hit = action_family_match(pred_act, gt_act)
            asa_num += int(hit)
            asa_den += 1
            qtype_asa[qtype][0] += int(hit)
            qtype_asa[qtype][1] += 1

        # SER (safety error rate)
        if qtype in SER_SAFETY_CRITICAL and gt_act:
            cost = ser_cost(gt_act, pred_act)
            if is_full:
                ser_full_sum += cost
                ser_full_den += 1
            else:
                ser_deg_sum += cost
                ser_deg_den += 1

    # Compute metrics
    sva  = sva_num / sva_den if sva_den else None
    asa  = asa_num / asa_den if asa_den else None
    bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else None
    tag_pct = tag_all_four / tag_total if tag_total else None

    ser_full  = (ser_full_sum / ser_full_den)  if ser_full_den >= 30  else None
    ser_full_flag = "*" if (0 < ser_full_den < 30) else ""
    ser_deg   = (ser_deg_sum  / ser_deg_den)   if ser_deg_den  >= 30  else None

    ser_inv_full = (1 - ser_full) if ser_full is not None else None

    # RQS placeholder (requires llm_judge.py output — filled by merge step)
    rqs = None

    # MoRAL-Score composite (only for full-sensor, non-degraded)
    moral_score = None
    if not is_degraded and sva is not None and asa is not None and ser_inv_full is not None:
        rqs_val = rqs if rqs is not None else 0.0   # treat missing RQS as 0
        moral_score = W_SVA * sva + W_ASA * asa + W_RQS * rqs_val + W_SER * ser_inv_full

    unique_pct = unique_prediction_rate(records)

    return {
        "n_samples":     len(records),
        "sva":           round(sva, 4)  if sva  is not None else None,
        "sva_n":         sva_den,
        "asa":           round(asa, 4)  if asa  is not None else None,
        "asa_n":         asa_den,
        "bleu":          round(bleu, 4) if bleu is not None else None,
        "tag_pct":       round(tag_pct, 4) if tag_pct is not None else None,
        "ser_full":      round(ser_full, 4)     if ser_full  is not None else None,
        "ser_full_flag": ser_full_flag,
        "ser_full_n":    ser_full_den,
        "ser_degraded":  round(ser_deg, 4)      if ser_deg   is not None else None,
        "ser_deg_n":     ser_deg_den,
        "ser_inv_full":  round(ser_inv_full, 4) if ser_inv_full is not None else None,
        "rqs":           rqs,
        "moral_score":   round(moral_score, 4)  if moral_score is not None else None,
        "unique_pct":    round(unique_pct, 4),
        "is_degraded":   is_degraded,
        "qtype_sva":     {k: {"correct": v[0], "total": v[1],
                              "pct": round(v[0]/v[1], 4) if v[1] else None}
                          for k, v in qtype_sva.items()},
        "qtype_asa":     {k: {"correct": v[0], "total": v[1],
                              "pct": round(v[0]/v[1], 4) if v[1] else None}
                          for k, v in qtype_asa.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# File discovery
# ═══════════════════════════════════════════════════════════════════════════════

def parse_filename(path: Path) -> dict:
    """
    Extract metadata from result filename.
    Handles patterns:
      results_cosmos_reason2_2b__{modality}__{condition}.jsonl          (zero-shot 2B)
      results_cosmos_reason2_8b__{modality}__{condition}.jsonl          (zero-shot 8B)
      results_cosmos2b_cond{X}_finetuned__{modality}__{condition}.jsonl (fine-tuned 2B)
      results_moral_cosmos2b_cond{X}__{modality}__{condition}.jsonl     (fine-tuned 2B alt)
      results_cosmos2b_cond{D}_8b_finetuned__{modality}__{condition}    (fine-tuned 8B)
    """
    stem = path.stem  # filename without .jsonl
    meta = {"path": path, "raw_name": stem}

    # Model size
    if "8b" in stem.lower():
        meta["model_size"] = "8B"
    else:
        meta["model_size"] = "2B"

    # Zero-shot vs fine-tuned
    if "finetuned" in stem or ("moral_cosmos" in stem and "zeroshot" not in stem):
        meta["eval_type"] = "finetuned"
    else:
        meta["eval_type"] = "zeroshot"

    # Training condition (for fine-tuned)
    cond_match = re.search(r"cond([BDE])", stem, re.IGNORECASE)
    meta["train_cond"] = cond_match.group(1).upper() if cond_match else None

    # Extract modality and eval condition from the __ separated parts
    parts = stem.split("__")
    if len(parts) >= 3:
        meta["modality"]   = parts[-2]
        raw_cond = parts[-1].split("_cleanbev")[0]   # strip _cleanbev suffix
        meta["eval_cond"]  = raw_cond.upper()
    elif len(parts) == 2:
        meta["modality"]  = parts[-1]
        meta["eval_cond"] = None
    else:
        meta["modality"]  = None
        meta["eval_cond"] = None

    return meta


def discover_all_experiments() -> list[dict]:
    """Find all result JSONL files across all save directories."""
    experiments = []
    dirs = [ZS_DIR, ZS_8B, FT_DIR, FT_AWS, FT_8B]

    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("results_*.jsonl")):
            meta = parse_filename(f)
            meta["save_dir"] = d.name
            experiments.append(meta)

    return experiments


# ═══════════════════════════════════════════════════════════════════════════════
# Load and score
# ═══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def run_all(filter_exp: str = None, verbose: bool = False) -> dict:
    """Score all discovered experiments. Returns {exp_key: score_dict}."""
    experiments = discover_all_experiments()
    if not experiments:
        print(f"[WARN] No result files found. Check paths:\n  {ZS_DIR}\n  {FT_DIR}")
        return {}

    results = {}
    for meta in experiments:
        exp_key = f"{meta['eval_type']}__{meta['model_size']}__{meta.get('train_cond','ZS')}__{meta['modality']}__{meta['eval_cond']}"
        if filter_exp and filter_exp not in exp_key:
            continue

        print(f"  Scoring: {meta['path'].name} ...", end="", flush=True)
        records = load_jsonl(meta["path"])
        scores  = score_experiment(records, modality=meta.get("modality", ""), verbose=verbose)
        scores.update(meta)
        scores["exp_key"] = exp_key
        results[exp_key]  = scores

        sva_str = f"{scores['sva']*100:.1f}%" if scores["sva"] is not None else "N/A"
        asa_str = f"{scores['asa']*100:.1f}%" if scores["asa"] is not None else "N/A"
        print(f" SVA={sva_str}  ASA={asa_str}  n={scores['n_samples']}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SGC — Sensor Grounding Consistency
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sgc(results: dict) -> dict:
    """
    SGC(model, sensor) = (SVA_with_sensor - SVA_without_sensor) / SVA_without_sensor
    Computes for each model group across modality pairs.
    """
    sgc = {}

    # Group results by (eval_type, model_size, train_cond, eval_cond)
    groups = defaultdict(dict)
    for key, r in results.items():
        group = (r["eval_type"], r["model_size"], r.get("train_cond"), r.get("eval_cond"))
        modality = r.get("modality")
        if modality and r.get("sva") is not None:
            groups[group][modality] = r["sva"]

    # Compute SGC for radar contribution: clean_radar vs clean_lidar
    for group, mods in groups.items():
        if "clean_radar" in mods and "clean_lidar" in mods:
            baseline = mods["clean_lidar"]
            sensor_val = mods["clean_radar"]
            if baseline > 0:
                sgc[group] = {"radar_sgc": round((sensor_val - baseline) / baseline, 4),
                               "radar_sva": sensor_val,
                               "lidar_sva": baseline}

        if "img" in mods and "cam_only" in mods:
            baseline = mods["cam_only"]
            if baseline > 0:
                sgc.setdefault(group, {})["bev_sgc"] = round(
                    (mods["img"] - baseline) / baseline, 4)

    return sgc


# ═══════════════════════════════════════════════════════════════════════════════
# Table generation
# ═══════════════════════════════════════════════════════════════════════════════

def pct(v, flag=""):
    if v is None:
        return "—"
    return f"{v*100:.1f}%{flag}"

def fmt(v):
    if v is None:
        return "—"
    return f"{v:.4f}"


def build_table1(results: dict) -> str:
    """Table 1 — Main Results: Model × Condition × Modality → metrics."""
    rows = []
    rows.append("## Table 1 — Main Results\n")
    rows.append("| Model | Train | Modality | Eval Cond | N | SVA | ASA | BLEU | Tags% | MoRAL-Score | Unique% |")
    rows.append("|-------|-------|----------|-----------|---|-----|-----|------|-------|-------------|---------|")

    # Sort: zeroshot first, then finetuned; by model size, condition, modality
    def sort_key(item):
        r = item[1]
        return (r["eval_type"], r["model_size"], r.get("train_cond") or "ZS",
                r.get("eval_cond") or "", r.get("modality") or "")

    for _, r in sorted(results.items(), key=sort_key):
        model_label = f"Cosmos-{r['model_size']}"
        train_label = r.get("train_cond") or ("ZS" if r["eval_type"] == "zeroshot" else "?")
        ev_type     = "ZS" if r["eval_type"] == "zeroshot" else f"FT-{train_label}"
        rows.append(
            f"| {model_label} | {ev_type} | {r.get('modality','?')} | {r.get('eval_cond','?')} "
            f"| {r['n_samples']} "
            f"| {pct(r['sva'])} | {pct(r['asa'])} | {fmt(r['bleu'])} "
            f"| {pct(r['tag_pct'])} | {fmt(r['moral_score'])} | {pct(r['unique_pct'])} |"
        )
    return "\n".join(rows)


def build_table2(results: dict, sgc: dict) -> str:
    """Table 2 — Sensor Ablation: cam_only vs BEV-only vs BEV+radar."""
    rows = []
    rows.append("\n## Table 2 — Sensor Ablation\n")
    rows.append("| Model | Train | Eval Cond | cam_only SVA | lidar_only SVA | clean_lidar SVA | clean_radar SVA | SGC(radar) | SER_degraded |")
    rows.append("|-------|-------|-----------|-------------|----------------|-----------------|-----------------|------------|--------------|")

    # Group by (eval_type, model_size, train_cond, eval_cond)
    groups = defaultdict(dict)
    for _, r in results.items():
        group = (r["eval_type"], r["model_size"], r.get("train_cond"), r.get("eval_cond"))
        mod = r.get("modality")
        if mod:
            groups[group][mod] = r

    for group, mods in sorted(groups.items()):
        eval_type, model_size, train_cond, eval_cond = group
        ev_label = "ZS" if eval_type == "zeroshot" else f"FT-{train_cond or '?'}"
        model_label = f"Cosmos-{model_size}"

        def sva_for(m):
            return pct(mods[m]["sva"]) if m in mods else "—"
        def ser_deg(m):
            if m not in mods:
                return "—"
            v = mods[m].get("ser_degraded")
            flag = mods[m].get("ser_full_flag", "")
            return pct(v, flag)

        sgc_val = sgc.get(group, {}).get("radar_sgc")
        sgc_str = f"{sgc_val*100:+.1f}%" if sgc_val is not None else "—"

        rows.append(
            f"| {model_label} | {ev_label} | {eval_cond or '?'} "
            f"| {sva_for('cam_only')} | {sva_for('clean_lidar_only')} "
            f"| {sva_for('clean_lidar')} | {sva_for('clean_radar')} "
            f"| {sgc_str} | {ser_deg('clean_lidar_only')} |"
        )
    return "\n".join(rows)


def build_table3(results: dict) -> str:
    """Table 3 — Mode Collapse & Unique Prediction audit (proxy for NOV score until novel_detection_probe.py runs)."""
    rows = []
    rows.append("\n## Table 3 — Mode Collapse & Prediction Diversity\n")
    rows.append("*Note: Full NOV score requires novel_detection_probe.py. This table reports unique pred_action % as a mode-collapse proxy.*\n")
    rows.append("| Model | Train | Modality | Eval Cond | N | Unique% | Collapse? |")
    rows.append("|-------|-------|----------|-----------|---|---------|-----------|")

    for _, r in sorted(results.items(), key=lambda x: x[1].get("unique_pct", 1.0)):
        model_label = f"Cosmos-{r['model_size']}"
        train_label = r.get("train_cond") or ("ZS" if r["eval_type"] == "zeroshot" else "?")
        ev_type = "ZS" if r["eval_type"] == "zeroshot" else f"FT-{train_label}"
        upct = r.get("unique_pct")
        collapse = "⚠️ YES" if (upct is not None and upct < 0.65) else "no"
        rows.append(
            f"| {model_label} | {ev_type} | {r.get('modality','?')} | {r.get('eval_cond','?')} "
            f"| {r['n_samples']} | {pct(upct)} | {collapse} |"
        )
    return "\n".join(rows)


def build_summary_stats(results: dict, sgc: dict) -> str:
    """Print key findings summary for quick human review."""
    lines = ["\n## Key Findings Summary\n"]

    # Best MoRAL-Score
    scored = [(k, r) for k, r in results.items() if r.get("moral_score") is not None]
    if scored:
        best_k, best_r = max(scored, key=lambda x: x[1]["moral_score"])
        lines.append(f"**Best MoRAL-Score**: {best_r['moral_score']:.4f} "
                     f"({best_r['eval_type']} {best_r['model_size']} | "
                     f"{best_r.get('modality')} | cond{best_r.get('eval_cond')})")

    # Radar SGC
    for group, vals in sgc.items():
        if "radar_sgc" in vals:
            ev, sz, tc, ec = group
            lines.append(f"**Radar SGC** ({sz} {ev} cond{ec}): "
                         f"{vals['radar_sgc']*100:+.1f}% spatial gain "
                         f"(lidar={vals['lidar_sva']*100:.1f}% → radar={vals['radar_sva']*100:.1f}%)")

    # Mode collapse warnings
    collapses = [(r.get("modality"), r.get("eval_cond"), r.get("unique_pct"))
                 for r in results.values()
                 if r.get("unique_pct") is not None and r["unique_pct"] < 0.65]
    if collapses:
        lines.append(f"\n**⚠️ Mode collapse detected** ({len(collapses)} experiments, unique% < 65%):")
        for mod, cond, upct in sorted(collapses, key=lambda x: x[2]):
            lines.append(f"  - {mod} / {cond}: {upct*100:.1f}%")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Save outputs
# ═══════════════════════════════════════════════════════════════════════════════

def save_outputs(results: dict, sgc: dict):
    # Full JSON results
    out_json = OUT_DIR / "moral_scores_all.json"
    serialisable = {k: {kk: (str(vv) if isinstance(vv, Path) else vv)
                         for kk, vv in v.items()}
                    for k, v in results.items()}
    out_json.write_text(json.dumps(serialisable, indent=2))
    print(f"\n[✓] Scores written → {out_json}")

    # SGC
    sgc_json = OUT_DIR / "sgc_heatmap.json"
    sgc_serial = {str(k): v for k, v in sgc.items()}
    sgc_json.write_text(json.dumps(sgc_serial, indent=2))
    print(f"[✓] SGC heatmap  → {sgc_json}")

    # Tables markdown
    tables_md = OUT_DIR / "all_tables.md"
    content = "# MoRAL Evaluation Tables\n"
    content += f"*Generated from {len(results)} experiments*\n\n"
    content += build_table1(results) + "\n"
    content += build_table2(results, sgc) + "\n"
    content += build_table3(results) + "\n"
    content += build_summary_stats(results, sgc) + "\n"
    tables_md.write_text(content)
    print(f"[✓] Tables       → {tables_md}")


# ═══════════════════════════════════════════════════════════════════════════════
# RQS merge (for when llm_judge.py has run)
# ═══════════════════════════════════════════════════════════════════════════════

def merge_rqs(results: dict, judge_dir: Path) -> dict:
    """
    Merge RQS scores from llm_judge output files into results dict.
    Judge files expected: saves/judge_results/rqs_{exp_key}.json
    Each file: {"rqs_mean": 0.72, "per_sample": [...]}
    """
    if not judge_dir.exists():
        print(f"[INFO] No judge_results dir found at {judge_dir} — skipping RQS merge")
        return results

    merged = 0
    for key in list(results.keys()):
        judge_file = judge_dir / f"rqs_{key}.json"
        if judge_file.exists():
            jdata = json.loads(judge_file.read_text())
            rqs_val = jdata.get("rqs_mean")
            results[key]["rqs"] = rqs_val
            # Recompute MoRAL-Score with RQS
            r = results[key]
            if (not r["is_degraded"] and r.get("sva") and r.get("asa")
                    and r.get("ser_inv_full") and rqs_val is not None):
                results[key]["moral_score"] = round(
                    W_SVA * r["sva"] + W_ASA * r["asa"]
                    + W_RQS * rqs_val + W_SER * r["ser_inv_full"], 4)
            merged += 1

    print(f"[✓] Merged RQS for {merged} experiments")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MoRAL-Score evaluation pipeline")
    parser.add_argument("--experiment",   help="Filter to single experiment key substring")
    parser.add_argument("--tables-only",  action="store_true",
                        help="Load existing moral_scores_all.json and regenerate tables only")
    parser.add_argument("--merge-rqs",    action="store_true",
                        help="Merge llm_judge RQS scores before computing composite")
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args()

    if args.tables_only:
        existing = OUT_DIR / "moral_scores_all.json"
        if not existing.exists():
            print(f"[ERROR] No existing scores at {existing}. Run without --tables-only first.")
            return
        results = json.loads(existing.read_text())
        # Re-parse Path fields
        sgc = compute_sgc(results)
        save_outputs(results, sgc)
        return

    print(f"\n{'='*60}")
    print("MoRAL-Score Evaluation Pipeline")
    print(f"Pipeline root: {PIPELINE_ROOT}")
    print(f"{'='*60}\n")

    results = run_all(filter_exp=args.experiment, verbose=args.verbose)

    if not results:
        print("[WARN] No experiments scored.")
        return

    if args.merge_rqs:
        results = merge_rqs(results, PIPELINE_ROOT / "saves/judge_results")

    sgc = compute_sgc(results)
    save_outputs(results, sgc)

    print(f"\n[✓] Done. {len(results)} experiments scored.")
    print(f"    Tables → {OUT_DIR}/all_tables.md")


if __name__ == "__main__":
    main()
