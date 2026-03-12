#!/usr/bin/env python3
"""
MoRAL LLM-as-Judge — Evaluate VLM outputs using Claude or OpenAI GPT
=====================================================================

Uses a structured rubric to grade each prediction on multiple dimensions,
then aggregates scores across question types and model configurations.

USAGE:
    # Grade results using Claude (recommended — best at spatial reasoning):
    python llm_judge.py \
        --results_dir saves/zeroshot_results/ \
        --provider anthropic \
        --model claude-sonnet-4-20250514 \
        --output_dir saves/judge_results/

    # Grade results using GPT-4o:
    python llm_judge.py \
        --results_dir saves/zeroshot_results/ \
        --provider openai \
        --model gpt-4o \
        --output_dir saves/judge_results/

    # Grade a single results file:
    python llm_judge.py \
        --results_file saves/zeroshot_results/results_clean_radar_D.jsonl \
        --provider anthropic \
        --model claude-sonnet-4-20250514

    # Compare multiple runs (generates comparison table):
    python llm_judge.py \
        --results_dir saves/zeroshot_results/ saves/finetuned_results/ \
        --provider anthropic \
        --model claude-sonnet-4-20250514 \
        --compare

    # Dry run (test with first 5 samples, no API calls):
    python llm_judge.py \
        --results_file saves/zeroshot_results/results_clean_radar_D.jsonl \
        --dry_run

ENVIRONMENT:
    export ANTHROPIC_API_KEY="sk-ant-..."
    export OPENAI_API_KEY="sk-..."

OUTPUT:
    For each input results file, produces:
    - judge_<run_id>.jsonl       — per-sample grades
    - judge_summary_<run_id>.json — aggregate scores per qtype + overall
    - comparison_table.md        — if --compare, cross-run comparison
"""

import argparse
import json
import os
import sys
import time
import re
import math
from pathlib import Path
from collections import defaultdict
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# RUBRIC — what the LLM judge evaluates per sample
# ═══════════════════════════════════════════════════════════════════════════════

RUBRIC = """You are an expert evaluator for autonomous driving VLM (Vision-Language Model) outputs.
You will grade a model's response to a driving scene question on multiple dimensions.

## CONTEXT
- The model was given a BEV (Bird's Eye View) map and/or camera image of a driving scene
- It was asked a question about the scene (spatial, safety, velocity, etc.)
- You have the ground truth values to compare against

## GRADING DIMENSIONS (score each 1-5)

### 1. REASONING_STRUCTURE (1-5)
Does the response follow structured reasoning?
- 5: Clear [BEV]→[CAM]→[GT]→[DECISION] structure with each section adding value
- 4: Has structure but some sections are weak or redundant
- 3: Partially structured, some analysis present
- 2: Mostly unstructured, stream-of-consciousness
- 1: No structure, just a direct answer or hallucination

### 2. SPATIAL_ACCURACY (1-5)
Are distances, positions, and zones correct?
- 5: All spatial claims within 20% of ground truth
- 4: Most spatial claims correct, one minor error
- 3: Some spatial claims correct, some significantly off
- 2: Major spatial errors (>50% off)
- 1: Completely wrong spatial reasoning or hallucinated positions

### 3. TEMPORAL_ACCURACY (1-5)
Are TTC, velocities, and time predictions correct?
- 5: TTC/velocity within 20% of ground truth
- 4: TTC/velocity within 30% of ground truth
- 3: Right order of magnitude
- 2: Wrong but in the right direction (e.g., says "fast" for a fast object)
- 1: Completely wrong or hallucinated

### 4. ACTION_CORRECTNESS (1-5)
Is the recommended action correct?
- 5: Exact match to ground truth action with correct reasoning
- 4: Correct action family (e.g., BRAKE vs EMERGENCY_BRAKE)
- 3: Reasonable action but not optimal
- 2: Wrong action but shows awareness of the risk
- 1: Dangerous action or no action when one is needed

### 5. SENSOR_GROUNDING (1-5)
Does the model correctly reference what it sees in the BEV/camera?
- 5: Accurately describes BEV objects, colors, arrows, range rings
- 4: Mostly accurate with minor visual errors
- 3: Some correct references mixed with invented details
- 2: Mostly fabricated visual descriptions
- 1: Pure hallucination — describes things not in the image

### 6. PHYSICAL_PLAUSIBILITY (1-5)
Are the physics, speeds, and distances physically plausible?
- 5: All physics correct (stopping distances, closing speeds, etc.)
- 4: Minor physics errors but reasonable
- 3: Some implausible claims
- 2: Major physics errors (e.g., car stopping in 1m from 60km/h)
- 1: Completely implausible

## SCORING RULES
- If a dimension is NOT applicable (e.g., no TTC in a gap question), score it as null
- Be strict on SPATIAL_ACCURACY — this is the key thesis metric
- Hallucinated objects or distances that don't match the scene should severely penalize SENSOR_GROUNDING
- The BLEU score is provided for reference but should NOT influence your grading

## OUTPUT FORMAT
Respond with ONLY a JSON object, no other text:
```json
{
    "reasoning_structure": <1-5 or null>,
    "spatial_accuracy": <1-5 or null>,
    "temporal_accuracy": <1-5 or null>,
    "action_correctness": <1-5 or null>,
    "sensor_grounding": <1-5 or null>,
    "physical_plausibility": <1-5 or null>,
    "overall_quality": <1-5>,
    "brief_justification": "<2-3 sentences explaining the grade>"
}
```"""


def build_judge_prompt(sample: dict) -> str:
    """Build the prompt for the LLM judge given a single evaluation result."""
    # Build ground truth context
    gt_parts = []
    if sample.get("gt_value") is not None:
        gt_parts.append(f"Ground truth value: {sample['gt_value']} ({sample.get('gt_field', 'unknown')})")
    if sample.get("gt_action"):
        gt_parts.append(f"Ground truth action: {sample['gt_action']}")
    if sample.get("gt_ttc") is not None:
        gt_parts.append(f"Ground truth TTC: {sample['gt_ttc']}s")
    gt_context = "\n".join(gt_parts) if gt_parts else "No quantitative ground truth available for this question."

    # Quality tags
    tags = sample.get("quality_tags", {})
    tag_str = ", ".join(f"{k}={'✅' if v else '❌'}" for k, v in tags.items())

    prompt = f"""## EVALUATION SAMPLE

**Scene:** {sample.get('scene', 'unknown')}
**Question Type:** {sample.get('qtype', 'unknown')}
**BLEU Score:** {sample.get('bleu', 0):.4f}
**Quality Tags Present:** {tag_str}

### Ground Truth
{gt_context}

### Model Prediction
{sample.get('pred', '(no prediction)')}

### Predicted Action: {sample.get('pred_action', 'none')}
### Predicted TTC: {sample.get('pred_ttc', 'none')}

Grade this response according to the rubric."""

    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# API CLIENTS — Claude and OpenAI with retry
# ═══════════════════════════════════════════════════════════════════════════════

def call_anthropic(system_prompt: str, user_prompt: str, model: str,
                   max_retries: int = 5) -> dict:
    """Call Claude API with exponential backoff retry."""
    try:
        import anthropic
    except ImportError:
        print("❌ Install anthropic: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
            )
            text = response.content[0].text.strip()
            # Extract JSON from response
            return _parse_judge_response(text)

        except anthropic.RateLimitError as e:
            wait = 30 * (2 ** (attempt - 1))
            print(f"    ⚠️  Rate limited (attempt {attempt}/{max_retries}). Waiting {wait}s...")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code in (500, 502, 503, 529):
                wait = 15 * (2 ** (attempt - 1))
                print(f"    ⚠️  Server error {e.status_code} (attempt {attempt}/{max_retries}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < max_retries:
                wait = 10 * (2 ** (attempt - 1))
                print(f"    ⚠️  Error: {e} (attempt {attempt}/{max_retries}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

    return {"error": f"Failed after {max_retries} attempts"}


def call_openai(system_prompt: str, user_prompt: str, model: str,
                max_retries: int = 5) -> dict:
    """Call OpenAI API with exponential backoff retry."""
    try:
        import openai
    except ImportError:
        print("❌ Install openai: pip install openai")
        sys.exit(1)

    client = openai.OpenAI()

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            return _parse_judge_response(text)

        except openai.RateLimitError:
            wait = 30 * (2 ** (attempt - 1))
            print(f"    ⚠️  Rate limited (attempt {attempt}/{max_retries}). Waiting {wait}s...")
            time.sleep(wait)
        except openai.APIStatusError as e:
            if e.status_code in (500, 502, 503):
                wait = 15 * (2 ** (attempt - 1))
                print(f"    ⚠️  Server error (attempt {attempt}/{max_retries}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < max_retries:
                wait = 10 * (2 ** (attempt - 1))
                print(f"    ⚠️  Error: {e} (attempt {attempt}/{max_retries}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

    return {"error": f"Failed after {max_retries} attempts"}


def _parse_judge_response(text: str) -> dict:
    """Extract JSON from judge response, handling code fences."""
    # Strip markdown code fences if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"error": f"Could not parse response: {text[:200]}"}


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRAMMATIC METRICS (no LLM needed — computed automatically)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_auto_metrics(sample: dict) -> dict:
    """Compute automatic metrics that don't need an LLM judge."""
    metrics = {}

    # 1. Quality tag coverage (does output have [BEV]/[CAM]/[GT]/[DECISION]?)
    tags = sample.get("quality_tags", {})
    tag_count = sum(1 for v in tags.values() if v)
    metrics["tag_coverage"] = tag_count / max(1, len(tags))

    # 2. BLEU score (already computed)
    metrics["bleu"] = sample.get("bleu", 0)

    # 3. Distance accuracy (GSA — Grounded Spatial Accuracy)
    gt_val = sample.get("gt_value")
    gt_field = sample.get("gt_field", "")
    pred_text = sample.get("pred", "")

    if gt_val is not None and gt_field in ("distance_m", "nearest_ahead_m"):
        extracted = _extract_number_from_text(pred_text, "meter")
        if extracted is not None:
            error = abs(extracted - gt_val) / max(0.1, abs(gt_val))
            metrics["distance_error_pct"] = round(error * 100, 1)
            metrics["distance_correct_20pct"] = error <= 0.20

    # 4. TTC accuracy
    pred_ttc = sample.get("pred_ttc")
    gt_ttc = sample.get("gt_ttc")
    if pred_ttc is not None and gt_ttc is not None and gt_ttc > 0:
        error = abs(pred_ttc - gt_ttc) / gt_ttc
        metrics["ttc_error_pct"] = round(error * 100, 1)
        metrics["ttc_correct_20pct"] = error <= 0.20

    # 5. Velocity accuracy
    if gt_val is not None and gt_field == "velocity_ms":
        extracted = _extract_number_from_text(pred_text, "m/s")
        if extracted is not None:
            error = abs(extracted - gt_val) / max(0.1, abs(gt_val))
            metrics["velocity_error_pct"] = round(error * 100, 1)
            metrics["velocity_correct_20pct"] = error <= 0.20

    # 6. Action match
    gt_action = sample.get("gt_action", "")
    pred_action = sample.get("pred_action", "")
    if gt_action and pred_action:
        metrics["action_exact_match"] = (
            gt_action.upper().strip() == pred_action.upper().strip()
        )
        # Family match (BRAKE matches EMERGENCY_BRAKE, etc.)
        ACTION_FAMILIES = {
            "BRAKE": {"BRAKE", "EMERGENCY_BRAKE", "HARD_BRAKE"},
            "MAINTAIN": {"MAINTAIN", "CONTINUE", "CRUISE"},
            "ACCELERATE": {"ACCELERATE", "SPEED_UP"},
            "YIELD": {"YIELD", "STOP", "WAIT"},
            "LANE_CHANGE": {"LANE_CHANGE", "SWERVE", "MERGE"},
        }
        gt_family = None
        pred_family = None
        for fam, members in ACTION_FAMILIES.items():
            if gt_action.upper().strip() in members:
                gt_family = fam
            if pred_action.upper().strip() in members:
                pred_family = fam
        metrics["action_family_match"] = (
            gt_family is not None and gt_family == pred_family
        )

    # 7. Response length (tokens proxy — word count)
    metrics["response_word_count"] = len(pred_text.split())

    return metrics


def _extract_number_from_text(text: str, unit_hint: str = "") -> Optional[float]:
    """Extract a number from text, optionally near a unit hint."""
    # Try to find number near the unit hint
    if unit_hint:
        patterns = [
            rf"(\d+\.?\d*)\s*{unit_hint}",
            rf"(\d+\.?\d*)\s*m\b",
            rf"approximately\s+(\d+\.?\d*)",
            rf"about\s+(\d+\.?\d*)",
            rf"(\d+\.?\d*)\s*meters?\b",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

    # Fallback: find all numbers and return the first reasonable one
    numbers = re.findall(r"(\d+\.?\d*)", text)
    for n in numbers:
        try:
            val = float(n)
            if 0.1 < val < 500:  # reasonable range for driving distances/speeds
                return val
        except ValueError:
            continue

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION — combine scores into thesis tables
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_results(judged_samples: list) -> dict:
    """Aggregate per-sample judge results into summary statistics."""
    summary = {
        "total_samples": len(judged_samples),
        "by_qtype": {},
        "overall": {},
    }

    # Group by question type
    by_qtype = defaultdict(list)
    for s in judged_samples:
        by_qtype[s.get("qtype", "unknown")].append(s)

    DIMENSIONS = [
        "reasoning_structure", "spatial_accuracy", "temporal_accuracy",
        "action_correctness", "sensor_grounding", "physical_plausibility",
        "overall_quality",
    ]

    AUTO_METRICS = [
        "tag_coverage", "bleu", "distance_error_pct",
        "distance_correct_20pct", "ttc_error_pct", "ttc_correct_20pct",
        "action_exact_match", "action_family_match",
    ]

    def _agg_dimension(samples, dim):
        vals = [s["judge"].get(dim) for s in samples
                if "judge" in s and s["judge"].get(dim) is not None
                and isinstance(s["judge"].get(dim), (int, float))]
        if not vals:
            return None
        return {
            "mean": round(sum(vals) / len(vals), 2),
            "min": min(vals),
            "max": max(vals),
            "count": len(vals),
        }

    def _agg_auto(samples, metric):
        vals = [s["auto_metrics"].get(metric) for s in samples
                if "auto_metrics" in s and s["auto_metrics"].get(metric) is not None]
        if not vals:
            return None
        if isinstance(vals[0], bool):
            return {"accuracy": round(sum(vals) / len(vals) * 100, 1), "count": len(vals)}
        return {"mean": round(sum(vals) / len(vals), 2), "count": len(vals)}

    # Per question type
    for qtype, samples in sorted(by_qtype.items()):
        summary["by_qtype"][qtype] = {
            "count": len(samples),
            "judge_scores": {d: _agg_dimension(samples, d) for d in DIMENSIONS},
            "auto_metrics": {m: _agg_auto(samples, m) for m in AUTO_METRICS},
        }

    # Overall
    summary["overall"] = {
        "judge_scores": {d: _agg_dimension(judged_samples, d) for d in DIMENSIONS},
        "auto_metrics": {m: _agg_auto(judged_samples, m) for m in AUTO_METRICS},
    }

    return summary


def generate_comparison_table(summaries: dict) -> str:
    """Generate a markdown comparison table across multiple runs."""
    lines = []
    lines.append("# MoRAL — Cross-Run Comparison Table\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n")

    # Header
    run_names = list(summaries.keys())
    header = "| Metric | " + " | ".join(run_names) + " |"
    sep = "|--------|" + "|".join(["-------"] * len(run_names)) + "|"
    lines.append(header)
    lines.append(sep)

    # Overall judge scores
    for dim in ["overall_quality", "spatial_accuracy", "temporal_accuracy",
                "action_correctness", "sensor_grounding", "reasoning_structure",
                "physical_plausibility"]:
        row = f"| **{dim}** |"
        for run_name in run_names:
            s = summaries[run_name].get("overall", {}).get("judge_scores", {}).get(dim)
            if s and s.get("mean") is not None:
                row += f" {s['mean']:.2f} |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")
    lines.append("### Automatic Metrics\n")
    header2 = "| Metric | " + " | ".join(run_names) + " |"
    lines.append(header2)
    lines.append(sep)

    for metric in ["tag_coverage", "bleu", "distance_correct_20pct",
                   "ttc_correct_20pct", "action_exact_match", "action_family_match"]:
        row = f"| **{metric}** |"
        for run_name in run_names:
            s = summaries[run_name].get("overall", {}).get("auto_metrics", {}).get(metric)
            if s is None:
                row += " — |"
            elif "accuracy" in s:
                row += f" {s['accuracy']:.1f}% |"
            elif "mean" in s:
                row += f" {s['mean']:.3f} |"
            else:
                row += " — |"
        lines.append(row)

    # Per qtype breakdown
    all_qtypes = set()
    for s in summaries.values():
        all_qtypes.update(s.get("by_qtype", {}).keys())

    if all_qtypes:
        lines.append("")
        lines.append("### Per Question Type — Overall Quality\n")
        header3 = "| Question Type | " + " | ".join(run_names) + " |"
        lines.append(header3)
        lines.append(sep)

        for qtype in sorted(all_qtypes):
            row = f"| {qtype} |"
            for run_name in run_names:
                qt = summaries[run_name].get("by_qtype", {}).get(qtype, {})
                js = qt.get("judge_scores", {}).get("overall_quality")
                if js and js.get("mean") is not None:
                    row += f" {js['mean']:.2f} ({qt['count']}) |"
                else:
                    row += " — |"
            lines.append(row)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="MoRAL LLM-as-Judge — Evaluate VLM outputs with Claude/GPT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_file", default=None,
                        help="Single JSONL results file to grade")
    parser.add_argument("--results_dir", nargs="+", default=None,
                        help="Directory(ies) containing results JSONL files")
    parser.add_argument("--output_dir", default="saves/judge_results",
                        help="Output directory for judge results")

    parser.add_argument("--provider", choices=["anthropic", "openai", "auto_only"],
                        default="anthropic",
                        help="LLM provider. 'auto_only' skips LLM calls and uses only programmatic metrics")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="Model name for the provider")

    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to judge per file")
    parser.add_argument("--compare", action="store_true",
                        help="Generate comparison table across runs")
    parser.add_argument("--dry_run", action="store_true",
                        help="Process 5 samples with auto metrics only (no API calls)")

    parser.add_argument("--batch_delay", type=float, default=1.0,
                        help="Seconds to wait between API calls (rate limit protection)")

    return parser.parse_args()


def find_results_files(args) -> list:
    """Collect all results JSONL files to process."""
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


def process_file(filepath: Path, args) -> tuple:
    """Process a single results file. Returns (run_id, summary)."""
    # Derive run ID from filename
    run_id = filepath.stem
    if run_id.startswith("results_"):
        run_id = run_id[len("results_"):]

    print(f"\n{'═' * 60}")
    print(f"  Judging: {run_id}")
    print(f"  File:    {filepath}")
    print(f"{'═' * 60}")

    # Load samples
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
    if args.dry_run:
        samples = samples[:5]

    print(f"  Loaded {len(samples)} samples")

    # Select API caller
    use_llm = (args.provider != "auto_only") and (not args.dry_run)
    if use_llm:
        if args.provider == "anthropic":
            api_call = lambda s, u: call_anthropic(s, u, args.model)
        else:
            api_call = lambda s, u: call_openai(s, u, args.model)

    # Grade each sample
    judged = []
    for i, sample in enumerate(samples):
        # Auto metrics (always computed)
        auto = compute_auto_metrics(sample)
        sample["auto_metrics"] = auto

        # LLM judge
        if use_llm:
            prompt = build_judge_prompt(sample)
            print(f"  [{i+1}/{len(samples)}] scene={sample.get('scene', '?')} "
                  f"qtype={sample.get('qtype', '?')} ...", end="", flush=True)
            judge_result = api_call(RUBRIC, prompt)
            sample["judge"] = judge_result
            status = "✅" if "error" not in judge_result else "❌"
            overall = judge_result.get("overall_quality", "?")
            print(f" → {status} quality={overall}")
            time.sleep(args.batch_delay)
        else:
            sample["judge"] = {}

        judged.append(sample)

    # Aggregate
    summary = aggregate_results(judged)
    summary["run_id"] = run_id
    summary["source_file"] = str(filepath)
    summary["provider"] = args.provider if use_llm else "auto_only"
    summary["model"] = args.model if use_llm else "N/A"

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample results
    out_jsonl = out_dir / f"judge_{run_id}.jsonl"
    with open(out_jsonl, "w") as f:
        for s in judged:
            f.write(json.dumps(s, default=str) + "\n")

    # Summary
    out_summary = out_dir / f"judge_summary_{run_id}.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Saved: {out_jsonl}")
    print(f"  Saved: {out_summary}")

    # Print quick summary
    overall = summary.get("overall", {})
    js = overall.get("judge_scores", {})
    am = overall.get("auto_metrics", {})

    print(f"\n  ── Summary: {run_id} ──")
    if js.get("overall_quality") and js["overall_quality"].get("mean"):
        print(f"  Overall Quality:    {js['overall_quality']['mean']:.2f}/5")
    if js.get("spatial_accuracy") and js["spatial_accuracy"].get("mean"):
        print(f"  Spatial Accuracy:   {js['spatial_accuracy']['mean']:.2f}/5")
    if am.get("tag_coverage") and am["tag_coverage"].get("mean"):
        print(f"  Tag Coverage:       {am['tag_coverage']['mean']:.1%}")
    if am.get("bleu") and am["bleu"].get("mean"):
        print(f"  BLEU (mean):        {am['bleu']['mean']:.4f}")
    if am.get("distance_correct_20pct") and am["distance_correct_20pct"].get("accuracy"):
        print(f"  Distance ±20%:      {am['distance_correct_20pct']['accuracy']:.1f}%")
    if am.get("action_family_match") and am["action_family_match"].get("accuracy"):
        print(f"  Action Family:      {am['action_family_match']['accuracy']:.1f}%")

    return run_id, summary


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print(" MoRAL LLM-as-Judge Evaluator")
    print("=" * 60)
    print(f"  Provider: {args.provider}")
    print(f"  Model:    {args.model}")
    print(f"  Output:   {args.output_dir}")
    if args.dry_run:
        print(f"  ⚡ DRY RUN — 5 samples, auto metrics only")

    # Check API keys
    if args.provider == "anthropic" and not args.dry_run:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("\n❌ Set ANTHROPIC_API_KEY environment variable")
            print("   export ANTHROPIC_API_KEY='sk-ant-...'")
            sys.exit(1)
    elif args.provider == "openai" and not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            print("\n❌ Set OPENAI_API_KEY environment variable")
            print("   export OPENAI_API_KEY='sk-...'")
            sys.exit(1)

    # Find files
    files = find_results_files(args)
    if not files:
        print("\n❌ No results files found. Specify --results_file or --results_dir")
        sys.exit(1)

    print(f"\n  Found {len(files)} results file(s)")

    # Process each file
    all_summaries = {}
    for filepath in files:
        run_id, summary = process_file(filepath, args)
        all_summaries[run_id] = summary

    # Generate comparison table
    if args.compare and len(all_summaries) > 1:
        table = generate_comparison_table(all_summaries)
        out_dir = Path(args.output_dir)
        table_path = out_dir / "comparison_table.md"
        with open(table_path, "w") as f:
            f.write(table)
        print(f"\n  📊 Comparison table: {table_path}")

    print(f"\n{'=' * 60}")
    print(f" ✅ Done — {len(all_summaries)} run(s) evaluated")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
