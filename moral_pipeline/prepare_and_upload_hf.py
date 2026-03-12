#!/usr/bin/env python3
"""
MoRAL — Upload all assets to HuggingFace
==========================================
Uploads everything needed to reproduce MoRAL on a new machine:
  - outputs/ (5 folders, ~3.3GB BEV images)
  - saves/zeroshot_results/ (~4MB)
  - saves/finetuned_results/ (when ready)
  - fine-tuned model weights (~150MB)
  - JSONL training/val files (~294MB, paths normalized)

Uses upload_folder() — minimal API calls, HF handles chunking.

Usage:
    python prepare_and_upload_hf.py --hf_token YOUR_TOKEN --hf_user AmbarishGK

    # Skip outputs (large) and just upload model + JSONLs:
    python prepare_and_upload_hf.py --hf_token YOUR_TOKEN --hf_user AmbarishGK --skip_images

Install:
    pip install huggingface_hub --break-system-packages
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--hf_token",    required=True)
parser.add_argument("--hf_user",     required=True,  help="e.g. AmbarishGK")
parser.add_argument("--data_root",   default=".",    help="moral_pipeline root dir")
parser.add_argument("--model_path",  default="saves/cosmos2b_condD_finetuned",
                    help="Root of fine-tuned model (contains adapter_model.safetensors)")
parser.add_argument("--skip_images", action="store_true", help="Skip outputs/ upload")
parser.add_argument("--skip_model",  action="store_true", help="Skip model upload")
parser.add_argument("--skip_jsonl",  action="store_true", help="Skip JSONL upload")
args = parser.parse_args()

ROOT       = Path(args.data_root).resolve()
HF_USER    = args.hf_user
HF_TOKEN   = args.hf_token
DATASET_ID = f"{HF_USER}/MoRAL-nuscenes"
MODEL_ID   = f"{HF_USER}/MoRAL-Cosmos2B-condD"

print(f"\n{'='*60}")
print(f" MoRAL HuggingFace Upload")
print(f"  Dataset repo: {DATASET_ID}")
print(f"  Model repo:   {MODEL_ID}")
print(f"{'='*60}\n")

from huggingface_hub import HfApi, login
login(token=HF_TOKEN)
api = HfApi()

# Create repos
for repo_id, repo_type in [(DATASET_ID, "dataset"), (MODEL_ID, "model")]:
    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
        print(f"  ✅ Repo ready: {repo_id}")
    except Exception as e:
        print(f"  ⚠️  {repo_id}: {e}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Normalize + upload JSONL files
# ═══════════════════════════════════════════════════════════════════════════════
if not args.skip_jsonl:
    print("[1/4] Normalizing and uploading JSONL files...")

    PATH_REWRITES = {
        str(ROOT / "outputs/03_clean_bev"):
            "outputs/03_clean_bev",
        str(ROOT / "outputs/04_clean_bev_lidar_only"):
            "outputs/04_clean_bev_lidar_only",
        str(ROOT / "outputs/01_gt_annotations"):
            "outputs/01_gt_annotations",
        str(ROOT / "outputs/02_gt_with_radar"):
            "outputs/02_gt_with_radar",
        "/home/shadeform/moral_pipeline/outputs/condition_D_lidar_bev_radar":
            "outputs/03_clean_bev",
        "/home/shadeform/moral_pipeline/outputs/condition_B_lidar_bev":
            "outputs/04_clean_bev_lidar_only",
    }

    JSONL_FILES = [
        "02_cosmos_integration/hf_data/clean_conditionD_train.jsonl",
        "02_cosmos_integration/hf_data/clean_conditionD_val.jsonl",
        "02_cosmos_integration/hf_data/clean_conditionB_train.jsonl",
        "02_cosmos_integration/hf_data/clean_conditionB_val.jsonl",
        "02_cosmos_integration/hf_data/local_conditionD_val_zeroshot_v2.jsonl",
        "02_cosmos_integration/hf_data/local_conditionB_val_zeroshot_v2.jsonl",
        # add to JSONL_FILES list in the script:
        "02_cosmos_integration/hf_data/clean_conditionE_train.jsonl",
        "02_cosmos_integration/hf_data/clean_conditionE_val.jsonl",
        "02_cosmos_integration/hf_data/simple_eval_val.jsonl",
    ]

    tmp_jsonl = Path(tempfile.mkdtemp()) / "jsonl"
    tmp_jsonl.mkdir()

    for rel_path in JSONL_FILES:
        src = ROOT / rel_path
        if not src.exists():
            print(f"  ⚠️  Not found: {rel_path}")
            continue
        dst = tmp_jsonl / src.name
        changed = 0
        with open(src) as fin, open(dst, "w") as fout:
            for line in fin:
                orig = line
                for old, new in PATH_REWRITES.items():
                    line = line.replace(old, new)
                if line != orig:
                    changed += 1
                fout.write(line)
        n = sum(1 for _ in open(dst))
        print(f"  {src.name}: {n} records, {changed} paths rewritten")

    print(f"  Uploading all JSONL files in one call...")
    api.upload_folder(
        folder_path=str(tmp_jsonl),
        path_in_repo="data",
        repo_id=DATASET_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    shutil.rmtree(str(tmp_jsonl.parent))
    print(f"  ✅ JSONL files uploaded to {DATASET_ID}/data/\n")
else:
    print("[1/4] Skipping JSONL upload\n")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Upload output folders (BEV images)
# ═══════════════════════════════════════════════════════════════════════════════
if not args.skip_images:
    print("[2/4] Uploading outputs/ folders (~3.3GB, 20-40 min)...")

    OUTPUT_FOLDERS = [
        ("outputs/00_camera_only",          "outputs/00_camera_only"),
        ("outputs/01_gt_annotations",       "outputs/01_gt_annotations"),
        ("outputs/02_gt_with_radar",        "outputs/02_gt_with_radar"),
        ("outputs/03_clean_bev",            "outputs/03_clean_bev"),
        ("outputs/04_clean_bev_lidar_only", "outputs/04_clean_bev_lidar_only"),
    ]

    for local_rel, repo_path in OUTPUT_FOLDERS:
        local_path = ROOT / local_rel
        if not local_path.exists():
            print(f"  ⚠️  Not found: {local_rel}")
            continue
        size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
        print(f"  Uploading {local_rel} ({size/1e6:.0f}MB)...")
        api.upload_folder(
            folder_path=str(local_path),
            path_in_repo=repo_path,
            repo_id=DATASET_ID,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        print(f"    ✅ Done")
    print()
else:
    print("[2/4] Skipping outputs/ upload (--skip_images)\n")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Upload eval results
# ═══════════════════════════════════════════════════════════════════════════════
print("[3/4] Uploading eval results...")

for results_dir, repo_path in [
    ("saves/zeroshot_results",  "saves/zeroshot_results"),
    ("saves/finetuned_results", "saves/finetuned_results"),
]:
    local_path = ROOT / results_dir
    if not local_path.exists() or not any(local_path.iterdir()):
        print(f"  ⚠️  Empty or not found: {results_dir} — skipping")
        continue
    size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
    print(f"  Uploading {results_dir} ({size/1e6:.1f}MB)...")
    api.upload_folder(
        folder_path=str(local_path),
        path_in_repo=repo_path,
        repo_id=DATASET_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"    ✅ Done")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Upload fine-tuned model (final adapter + best_model only, skip checkpoints)
# ═══════════════════════════════════════════════════════════════════════════════
if not args.skip_model:
    print("[4/4] Uploading fine-tuned model weights...")

    model_root = ROOT / args.model_path
    if not model_root.exists():
        print(f"  ⚠️  Model path not found: {model_root}")
    else:
        # Upload root adapter files (final epoch — skip checkpoint-* dirs)
        tmp_model = Path(tempfile.mkdtemp()) / "model"
        tmp_model.mkdir()

        root_files = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "processor_config.json",
            "chat_template.jinja",
            "README.md",
        ]
        copied = []
        for fname in root_files:
            src = model_root / fname
            if src.exists():
                shutil.copy2(src, tmp_model / fname)
                copied.append(fname)

        print(f"  Uploading final adapter files: {copied}")
        api.upload_folder(
            folder_path=str(tmp_model),
            path_in_repo="",
            repo_id=MODEL_ID,
            repo_type="model",
            token=HF_TOKEN,
        )
        shutil.rmtree(str(tmp_model.parent))
        print(f"  ✅ Final adapter uploaded")

        # Upload best_model/ (best val loss checkpoint)
        best_model = model_root / "best_model"
        if best_model.exists():
            size = sum(f.stat().st_size for f in best_model.rglob("*") if f.is_file())
            print(f"  Uploading best_model/ ({size/1e6:.0f}MB)...")
            api.upload_folder(
                folder_path=str(best_model),
                path_in_repo="best_model",
                repo_id=MODEL_ID,
                repo_type="model",
                token=HF_TOKEN,
            )
            print(f"  ✅ best_model/ uploaded")

        print(f"\n  Model: https://huggingface.co/{MODEL_ID}")
        print(f"  Load on new machine:")
        print(f"    from peft import PeftModel")
        print(f"    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor")
        print(f"    base = Qwen3VLForConditionalGeneration.from_pretrained('nvidia/Cosmos-Reason2-2B', ...)")
        print(f"    model = PeftModel.from_pretrained(base, '{MODEL_ID}')")
        print(f"    processor = AutoProcessor.from_pretrained('{MODEL_ID}')")
else:
    print("[4/4] Skipping model upload (--skip_model)\n")

print(f"\n{'='*60}")
print(f" Upload Complete!")
print(f"  Dataset: https://huggingface.co/datasets/{DATASET_ID}")
print(f"  Model:   https://huggingface.co/{MODEL_ID}")
print(f"{'='*60}")
print(f"""
To reproduce on a new machine:
  1. git clone https://github.com/AmbarishGK/bevfusionV2.git
  2. cd bevfusionV2/moral_pipeline
  3. huggingface-cli download {DATASET_ID} --repo-type dataset --local-dir .
  4. # Fix paths to match new machine:
     DATA_ROOT=$(pwd)
     for f in data/*.jsonl; do
         sed -i "s|outputs/|$DATA_ROOT/outputs/|g" "$f"
     done
  5. python train_cosmos2b.py --train_file data/clean_conditionD_train.jsonl ...
     # OR use pre-trained model directly:
     # python evaluate_zeroshot.py --model {MODEL_ID} ...
""")
