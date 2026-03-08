"""
evaluate_zeroshot.py
────────────────────
Zero-shot evaluation for MoRAL ablation study.
Supports all models, input conditions, and sensor conditions.

USAGE
-----
# Cosmos-Reason2-8B, conditionB, image only
python evaluate_zeroshot.py \
    --model nvidia/Cosmos-Reason2-8B \
    --val_file 02_cosmos_integration/hf_data/local_conditionB_val.jsonl \
    --input_level img \
    --condition B \
    --max_samples 200 \
    --output_dir saves/zeroshot_results

# cam_only (no BEV)
python evaluate_zeroshot.py \
    --model nvidia/Cosmos-Reason2-8B \
    --val_file 02_cosmos_integration/hf_data/local_conditionB_val.jsonl \
    --input_level cam_only \
    --condition B \
    --max_samples 200 \
    --output_dir saves/zeroshot_results

INPUT LEVELS
------------
cam_only     : CAM_FRONT only (no BEV)
img          : BEV + CAM_FRONT
img+det      : BEV + CAM_FRONT + detections.json text in prompt
"""

import os, sys, json, re, argparse, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

# ── Action family matching (from evaluate_moral.py) ───────────────────────────
ACTION_FAMILIES = {
    "EMERGENCY_BRAKE": {"EMERGENCY_BRAKE", "BRAKE", "STOP"},
    "BRAKE":           {"BRAKE", "EMERGENCY_BRAKE", "STOP"},
    "YIELD":           {"YIELD", "BRAKE"},
    "MAINTAIN":        {"MAINTAIN"},
    "STOP":            {"STOP", "BRAKE", "EMERGENCY_BRAKE"},
}

COSMOS_CATEGORY_MAP = {
    "spatial":        "Spatial & Temporal",
    "zone":           "Spatial & Temporal",
    "trajectory":     "Spatial & Temporal",
    "multi_conflict": "Actions & Motion",
    "velocity":       "Actions & Motion",
    "near_miss":      "Actions & Motion",
    "planning":       "Ego Vehicle Behavior",
    "occlusion":      "Key Objects",
    "counting":       "Key Objects",
    "scene_type":     "Scene Description",
    "sensor_limit":   "Scene Description",
    "pedestrian":     "Characters & Interactions",
    "gap":            "Spatial & Temporal",
    "physics":        "Actions & Motion",
    "counterfactual": "Ego Vehicle Behavior",
    "ethical":        "Ego Vehicle Behavior",
    "radar":          "Actions & Motion",
    "safety":         "Ego Vehicle Behavior",
}


def action_family_match(pred, gt):
    if not pred or not gt:
        return False
    return pred.upper() in ACTION_FAMILIES.get(gt.upper(), {gt.upper()})


def extract_action(text):
    patterns = [
        r'\[DECISION\]\s*([A-Z_]+)',
        r'<answer>[^<]*\b(EMERGENCY_BRAKE|BRAKE|YIELD|MAINTAIN|STOP)\b',
        r'\b(EMERGENCY_BRAKE|BRAKE|YIELD|MAINTAIN|STOP)\b',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def extract_ttc(text):
    patterns = [
        r'TTC\s*[=:≈~]\s*(\d+\.?\d*)\s*s',
        r'(\d+\.?\d*)\s*seconds?\s+(?:until|to\s+(?:impact|collision))',
        r'time.to.collision[^\d]*(\d+\.?\d*)',
        r'approximately\s+(\d+\.?\d*)\s*s(?:econds?)?',
        r'(\d+\.?\d*)\s*s\s+TTC',
    ]
    vals = []
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            v = float(m.group(1))
            if 0.1 < v < 300:
                vals.append(v)
    return vals[0] if vals else None


def extract_distances(text):
    """Extract all distance mentions from model output."""
    pattern = r'(\d+\.?\d*)\s*m(?:eters?)?\b'
    return [float(m.group(1)) for m in re.finditer(pattern, text)]


def check_quality_tags(text):
    return {tag: (tag in text) for tag in ['[BEV]', '[CAM]', '[GT]', '[DECISION]']}


def bleu_score(pred_text, gt_text):
    """BLEU using sacrebleu, fallback to nltk, fallback to unigram overlap."""
    if not pred_text or not gt_text:
        return None
    try:
        from sacrebleu.metrics import BLEU
        return round(BLEU(effective_order=True).sentence_score(pred_text, [gt_text]).score / 100, 4)
    except ImportError:
        pass
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        score = sentence_bleu([gt_text.split()], pred_text.split(),
                               smoothing_function=SmoothingFunction().method1)
        return round(score, 4)
    except ImportError:
        pass
    # Fallback: simple unigram overlap
    ref_tok = set(gt_text.lower().split())
    hyp_tok = pred_text.lower().split()
    if not hyp_tok:
        return 0.0
    return round(sum(1 for w in hyp_tok if w in ref_tok) / len(hyp_tok), 4)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name, load_in_4bit=True):
    """Load any supported VLM. Tries Unsloth first, falls back to transformers."""
    print(f"\nLoading model: {model_name}")

    # Try Unsloth (Qwen2.5-VL, Qwen3-VL, Cosmos-Reason2 all use same arch)
    try:
        from unsloth import FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing=False,
        )
        FastVisionModel.for_inference(model)
        print(f"  Loaded via Unsloth ✓")
        return model, tokenizer, 'unsloth'
    except Exception as e:
        print(f"  Unsloth failed ({e}), trying transformers...")

    # Fallback: exact Cosmos-Reason2 model card pattern
    try:
        import torch, transformers
        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.float16,
            device_map='auto', attn_implementation='sdpa',
        )
        tokenizer = transformers.AutoProcessor.from_pretrained(model_name)
        print(f"  Loaded via transformers ✓")
        return model, tokenizer, 'transformers'
    except Exception as e2:
        raise RuntimeError(f"Both Unsloth and transformers failed: {e}\n{e2}")


def run_inference(model, tokenizer, messages, backend, max_new_tokens=1500,
                  use_thinking=False):
    """Run inference for one record."""
    if backend == 'unsloth':
        from unsloth import FastVisionModel
        from qwen_vl_utils import process_vision_info

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        ).to('cuda')

        # Thinking models need sampling; standard needs greedy
        gen_kwargs = dict(max_new_tokens=max_new_tokens)
        if use_thinking:
            gen_kwargs.update(temperature=0.7, top_p=0.8, top_k=20,
                              do_sample=True)
        else:
            gen_kwargs['do_sample'] = False

        import torch
        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)

        generated = out_ids[0][inputs['input_ids'].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    else:
        # transformers backend — exact Cosmos-Reason2 model card pattern
        import torch
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
        )
        inputs = inputs.to(model.device)

        gen_kwargs = dict(max_new_tokens=max_new_tokens)
        if use_thinking:
            gen_kwargs.update(temperature=0.7, top_p=0.8, top_k=20, do_sample=True)
        else:
            gen_kwargs['do_sample'] = False

        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)

        trimmed = out_ids[0][inputs['input_ids'].shape[1]:]
        return tokenizer.batch_decode(
            [trimmed], skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]


# ── Detection injection ────────────────────────────────────────────────────────

def get_det_path(record, det_root, condition):
    """Derive detections.json path from record metadata and det_root."""
    meta  = record.get('_meta', {})
    scene = meta.get('scene', '')
    if not scene:
        return None
    subdir = '01_gt_annotations' if condition == 'B' else '02_gt_with_radar'
    if det_root:
        return os.path.join(det_root, scene, 'detections.json')
    # Default paths
    base = os.path.dirname(os.path.abspath(__file__))
    path_b = os.path.join(base, 'outputs', '01_gt_annotations', scene, 'detections.json')
    path_d = os.path.join(base, 'outputs', '02_gt_with_radar',  scene, 'detections.json')
    return path_b if condition == 'B' else path_d


def build_messages(record, input_level, condition, det_root=None, bev_root=None):
    """
    Build messages list for the given input level.

    input_level options:
        cam_only         : CAM_FRONT only
        img              : GT-box BEV + CAM_FRONT
        img+det          : GT-box BEV + CAM_FRONT + detections text
        bev_only         : GT-box BEV only (no camera)
        clean_lidar      : lidar-only clean BEV + CAM_FRONT
        clean_radar      : lidar+radar clean BEV + CAM_FRONT
        clean_lidar_only : lidar-only clean BEV (no camera)
        clean_radar_only : lidar+radar clean BEV (no camera)
    """
    import copy
    orig_content = record['messages'][0]['content']
    scene        = record.get('_meta', {}).get('scene', '')

    # Extract original parts
    images        = [p for p in orig_content if p['type'] == 'image']
    texts         = [p for p in orig_content if p['type'] == 'text']
    question_text = texts[0]['text'] if texts else ''
    gt_box_bev    = images[0] if len(images) > 0 else None  # BEV with GT boxes
    cam_img       = images[1] if len(images) > 1 else None  # CAM_FRONT

    # ── Helper: load a clean BEV image block ──
    def _clean_bev(fname):
        """Return image dict for clean BEV file, or None if missing."""
        root = bev_root or 'outputs/03_clean_bev'
        path = os.path.join(root, scene, fname)
        if not os.path.exists(path):
            return None
        block = dict(gt_box_bev) if gt_box_bev else {'type': 'image'}
        block['image'] = path
        return block

    # ── Helper: build detection text ──
    def _det_text():
        det_path = get_det_path(record, det_root, condition)
        if det_path and os.path.exists(det_path):
            try:
                from format_detections import format_detections_text
                with open(det_path) as f:
                    dets = json.load(f)
                meta = record.get('_meta', {})
                return format_detections_text(
                    dets, condition=condition,
                    ego_speed=meta.get('ego_speed_ms')
                ) + '\n\n'
            except Exception:
                pass
        return ''

    new_content = []

    if input_level == 'cam_only':
        # Camera only — no BEV at all
        if cam_img:
            new_content.append(cam_img)
        new_content.append({'type': 'text', 'text': question_text})

    elif input_level == 'img':
        # GT-box BEV + CAM_FRONT (original training format)
        new_content = copy.deepcopy(orig_content)

    elif input_level == 'img+det':
        # GT-box BEV + CAM_FRONT + detection text
        if gt_box_bev:
            new_content.append(gt_box_bev)
        if cam_img:
            new_content.append(cam_img)
        new_content.append({'type': 'text', 'text': _det_text() + question_text})

    elif input_level == 'bev_only':
        # GT-box BEV only — no camera
        if gt_box_bev:
            new_content.append(gt_box_bev)
        new_content.append({'type': 'text', 'text': question_text})

    elif input_level == 'clean_lidar':
        # Clean lidar-only BEV + CAM_FRONT
        bev = _clean_bev('bev_lidar_only.png')
        if bev:
            new_content.append(bev)
        if cam_img:
            new_content.append(cam_img)
        new_content.append({'type': 'text', 'text': question_text})

    elif input_level == 'clean_radar':
        # Clean lidar+radar BEV + CAM_FRONT
        bev = _clean_bev('bev_lidar_radar.png')
        if bev:
            new_content.append(bev)
        if cam_img:
            new_content.append(cam_img)
        new_content.append({'type': 'text', 'text': question_text})

    elif input_level == 'clean_lidar_only':
        # Clean lidar-only BEV — no camera
        bev = _clean_bev('bev_lidar_only.png')
        if bev:
            new_content.append(bev)
        new_content.append({'type': 'text', 'text': question_text})

    elif input_level == 'clean_radar_only':
        # Clean lidar+radar BEV — no camera
        bev = _clean_bev('bev_lidar_radar.png')
        if bev:
            new_content.append(bev)
        new_content.append({'type': 'text', 'text': question_text})

    else:
        raise ValueError(f"Unknown input_level: {input_level}")

    return [
    {'role': 'user', 'content': new_content}
    ]


# ── Main eval loop ─────────────────────────────────────────────────────────────

def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load val records
    with open(args.val_file) as f:
        all_records = [json.loads(l) for l in f]
    records = all_records[:args.max_samples]
    print(f"Loaded {len(records)} records from {args.val_file}")

    # Determine model short name for output files
    model_short = args.model.split('/')[-1].replace('-', '_').lower()
    bev_tag = '_cleanbev' if args.bev_root else ''
    run_id = f"{model_short}__{args.input_level}__{args.condition}{bev_tag}"
    results_path = os.path.join(args.output_dir, f"results_{run_id}.jsonl")
    summary_path = os.path.join(args.output_dir, f"summary_{run_id}.json")

    # Detect thinking model
    use_thinking = 'thinking' in args.model.lower() or 'reason' in args.model.lower()

    # Load model
    model, tokenizer, backend = load_model(args.model, load_in_4bit=not args.fp16)

    # BEV already contains GT boxes — no bbox cache needed

    # ── Metrics accumulators ──
    results     = []
    action_exact, action_family, action_total = 0, 0, 0
    ttc_errors  = []
    bleu_scores = []
    quality_counts = defaultdict(int)
    quality_total  = 0
    cosmos_cat_correct = defaultdict(int)
    cosmos_cat_total   = defaultdict(int)
    dist_errors = []

    print(f"\n{'='*60}")
    print(f" Zero-Shot Eval: {run_id}")
    print(f" Model:       {args.model}")
    print(f" Input level: {args.input_level}")
    print(f" Condition:   {args.condition}")
    print(f" Samples:     {len(records)}")
    print(f"{'='*60}")

    start_time = time.time()
    for i, record in enumerate(records):
        meta     = record.get('_meta', {})
        qtype    = meta.get('question_type', 'unknown')
        gt_raw    = record['messages'][1]['content'] if len(record['messages']) > 1 else ''
        _ans_m    = re.search(r'<answer>(.*?)</answer>', gt_raw, re.DOTALL)
        gt_ans    = _ans_m.group(1).strip() if _ans_m else gt_raw
        gt_action = (meta.get('gt_action') or '').upper()
        gt_ttc    = meta.get('gt_value') if meta.get('gt_unit') == 's' else None
        gt_value  = meta.get('gt_value')
        gt_field  = meta.get('gt_field')

        # ── Live progress bar ──
        elapsed  = time.time() - start_time
        avg_s    = elapsed / max(i, 1)
        eta_s    = avg_s * (len(records) - i)
        eta_str  = f"{int(eta_s//3600):02d}h{int((eta_s%3600)//60):02d}m" if i > 0 else "--h--m"
        bar_done = int(30 * (i / len(records)))
        bar      = '\u2588' * bar_done + '\u2591' * (30 - bar_done)
        print(f"\r  [{bar}] {i+1}/{len(records)} | ETA {eta_str} | {qtype:<14}",
              end='', flush=True)

        try:
            messages = build_messages(
                record, args.input_level, args.condition,
                det_root=args.det_root, bev_root=args.bev_root
            )
            pred = run_inference(model, tokenizer, messages, backend,
                                 max_new_tokens=args.max_new_tokens,
                                 use_thinking=use_thinking)
        except Exception as e:
            pred = f"ERROR: {e}"

        # ── Metrics ──
        pred_action = extract_action(pred)
        pred_ttc    = extract_ttc(pred)
        tags        = check_quality_tags(pred)
        _pred_ans_m = re.search(r'<answer>(.*?)</answer>', pred, re.DOTALL)
        pred_ans    = _pred_ans_m.group(1).strip() if _pred_ans_m else pred
        bleu        = bleu_score(pred_ans, gt_ans)

        # Action accuracy
        if gt_action:
            action_total += 1
            if pred_action and pred_action == gt_action:
                action_exact += 1
            if pred_action and action_family_match(pred_action, gt_action):
                action_family += 1

        # TTC error
        if pred_ttc and gt_ttc:
            try:
                ttc_errors.append(abs(float(pred_ttc) - float(gt_ttc)))
            except:
                pass

        # Distance/velocity error using gt_value + gt_field
        if gt_value is not None and gt_field in ('distance_m', 'velocity_ms'):
            dists = extract_distances(pred)
            if dists:
                closest = min(dists, key=lambda d: abs(d - gt_value))
                dist_errors.append(abs(closest - gt_value))

        # Quality tags
        quality_total += 1
        for tag, present in tags.items():
            if present:
                quality_counts[tag] += 1

        # BLEU
        if bleu is not None:
            bleu_scores.append(bleu)

        # Cosmos category breakdown
        cosmos_cat = COSMOS_CATEGORY_MAP.get(qtype, 'Other')
        cosmos_cat_total[cosmos_cat] += 1
        # Simple correctness: action match OR BLEU > 0.3
        correct = (pred_action and gt_action and action_family_match(pred_action, gt_action)) \
                  or (bleu is not None and bleu > 0.3)
        if correct:
            cosmos_cat_correct[cosmos_cat] += 1

        # Store result
        result = {
            'idx': i, 'scene': meta.get('scene'), 'qtype': qtype,
            'gt_action': gt_action, 'pred_action': pred_action,
            'gt_ttc': gt_ttc, 'pred_ttc': pred_ttc,
            'gt_value': gt_value, 'gt_field': gt_field,
            'bleu': bleu, 'quality_tags': tags,
            'pred': pred[:2000],
        }
        results.append(result)

        # ── Update bar with metrics after inference ──
        a_fam    = action_family / action_total * 100 if action_total else 0.0
        q_all    = quality_counts.get('[DECISION]', 0) / max(quality_total, 1) * 100
        b_avg    = np.mean(bleu_scores) * 100 if bleu_scores else 0.0
        elapsed  = time.time() - start_time
        avg_s    = elapsed / (i + 1)
        eta_s    = avg_s * (len(records) - i - 1)
        eta_str  = f"{int(eta_s//3600):02d}h{int((eta_s%3600)//60):02d}m"
        bar_done = int(30 * ((i + 1) / len(records)))
        bar      = '\u2588' * bar_done + '\u2591' * (30 - bar_done)
        line = (f"  [{bar}] {i+1}/{len(records)} | ETA {eta_str} | "
                f"bleu={b_avg:.1f}% q={q_all:.1f}% fam={a_fam:.1f}% | {qtype:<14}")
        print(f"\r{line}", end='', flush=True)
        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            print(flush=True)

    # ── Write results ──
    with open(results_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    # ── Compute summary ──
    summary = {
        'run_id':       run_id,
        'model':        args.model,
        'input_level':  args.input_level,
        'condition':    args.condition,
        'n_samples':    len(records),
        'action_exact_pct':  round(action_exact  / action_total * 100, 2) if action_total else 'N/A',
        'action_family_pct': round(action_family / action_total * 100, 2) if action_total else 'N/A',
        'action_n':          action_total,
        'mean_ttc_error_s':  round(float(np.mean(ttc_errors)), 3) if ttc_errors else 'N/A',
        'ttc_n':             len(ttc_errors),
        'mean_bleu':         round(float(np.mean(bleu_scores)), 4) if bleu_scores else 'N/A',
        'bleu_n':            len(bleu_scores),
        'quality_all_tags_pct': round(
            sum(quality_counts[t] for t in ['[BEV]','[CAM]','[GT]','[DECISION]'])
            / (quality_total * 4) * 100, 2) if quality_total else 'N/A',
        'quality_per_tag': {
            tag: round(quality_counts[tag] / quality_total * 100, 1)
            for tag in ['[BEV]', '[CAM]', '[GT]', '[DECISION]']
        } if quality_total else {},
        'cosmos_category_acc': {
            cat: round(cosmos_cat_correct[cat] / cosmos_cat_total[cat] * 100, 1)
            for cat in cosmos_cat_total
        },
        'spatial_temporal_acc': round(
            cosmos_cat_correct.get('Spatial & Temporal', 0) /
            max(cosmos_cat_total.get('Spatial & Temporal', 1), 1) * 100, 1
        ),
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ──
    print(f"\n{'─'*60}")
    print(f"  RESULTS — {run_id}")
    print(f"{'─'*60}")
    print(f"  Action exact:    {summary['action_exact_pct']}%  (n={action_total})")
    print(f"  Action family:   {summary['action_family_pct']}%")
    print(f"  Mean BLEU:       {summary['mean_bleu']}  (n={len(bleu_scores)})")
    print(f"  TTC error:       {summary['mean_ttc_error_s']}s  (n={len(ttc_errors)})")
    print(f"  Quality tags:    {summary['quality_all_tags_pct']}%")
    print(f"  Spatial & Temporal acc: {summary['spatial_temporal_acc']}%")
    print(f"\n  Cosmos category breakdown:")
    for cat, acc in summary['cosmos_category_acc'].items():
        n = cosmos_cat_total[cat]
        print(f"    {cat:<30} {acc:5.1f}%  (n={n})")
    print(f"{'─'*60}")
    print(f"  Results: {results_path}")
    print(f"  Summary: {summary_path}")

    return summary


def main():
    ap = argparse.ArgumentParser(description='MoRAL Zero-Shot Ablation Evaluator')
    ap.add_argument('--model',        required=True,
                    help='HuggingFace model ID or local path')
    ap.add_argument('--val_file',     required=True,
                    help='JSONL val file path')
    ap.add_argument('--input_level',  required=True,
                    choices=['cam_only', 'img', 'img+det',
                             'bev_only',
                             'clean_lidar', 'clean_radar',
                             'clean_lidar_only', 'clean_radar_only'],
                    help=(
                        'cam_only         : CAM_FRONT only\n'
                        'img              : GT-box BEV + CAM_FRONT\n'
                        'img+det          : GT-box BEV + CAM_FRONT + detections text\n'
                        'bev_only         : GT-box BEV only (no camera)\n'
                        'clean_lidar      : lidar-only BEV + CAM_FRONT\n'
                        'clean_radar      : lidar+radar BEV + CAM_FRONT\n'
                        'clean_lidar_only : lidar-only BEV (no camera)\n'
                        'clean_radar_only : lidar+radar BEV (no camera)'
                    ))
    ap.add_argument('--condition',    default='B', choices=['B', 'D'],
                    help='Sensor condition: B=LiDAR, D=LiDAR+Radar')
    ap.add_argument('--det_root',     default=None,
                    help='Root dir for detections.json files (auto-detected if None)')
    ap.add_argument('--bev_root',     default=None,
                    help='Override BEV image root (e.g. outputs/03_clean_bev for no-box BEVs)')
    ap.add_argument('--max_samples',  type=int, default=200)
    ap.add_argument('--max_new_tokens', type=int, default=1500)
    ap.add_argument('--output_dir',   default='saves/zeroshot_results')
    ap.add_argument('--fp16',         action='store_true',
                    help='Load in fp16 instead of 4-bit (needs more VRAM)')
    args = ap.parse_args()

    evaluate(args)


if __name__ == '__main__':
    main()
