#!/usr/bin/env python3
"""
MoRAL — Prepare & Upload to HuggingFace
========================================
1. Normalizes absolute paths in JSONL files → relative paths
2. Uploads JSONL datasets to HuggingFace Datasets
3. Uploads BEV images to HuggingFace Datasets
4. Uploads fine-tuned model to HuggingFace Hub

Usage:
    python prepare_and_upload_hf.py --hf_token YOUR_TOKEN --hf_user YOUR_HF_USERNAME

Install:
    pip install huggingface_hub datasets --break-system-packages
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--hf_token",   required=True,  help="HuggingFace write token")
parser.add_argument("--hf_user",    required=True,  help="HuggingFace username e.g. AmbarishGK")
parser.add_argument("--data_root",  default=".",    help="moral_pipeline root dir")
parser.add_argument("--model_path", default="saves/cosmos2b_condD_finetuned/best_model")
parser.add_argument("--skip_images",action="store_true", help="Skip BEV image upload (slow)")
parser.add_argument("--skip_model", action="store_true", help="Skip model upload")
args = parser.parse_args()

ROOT       = Path(args.data_root).resolve()
HF_USER    = args.hf_user
HF_TOKEN   = args.hf_token
DATASET_ID = f"{HF_USER}/MoRAL-nuscenes"
MODEL_ID   = f"{HF_USER}/MoRAL-Cosmos2B-condD"

print(f"\n{'='*60}")
print(f" MoRAL HuggingFace Upload")
print(f"  Dataset: {DATASET_ID}")
print(f"  Model:   {MODEL_ID}")
print(f"{'='*60}\n")

# ── Login ──────────────────────────────────────────────────────────────────────
from huggingface_hub import HfApi, login
login(token=HF_TOKEN)
api = HfApi()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Normalize paths in JSONL files
# ═══════════════════════════════════════════════════════════════════════════════
print("[1/4] Normalizing JSONL paths (absolute → relative)...")

# Map old absolute prefixes → new relative prefix
PATH_REWRITES = {
    # condD clean BEV (lidar+radar)
    str(ROOT / "outputs/03_clean_bev"):
        "outputs/03_clean_bev",
    # condB clean BEV (lidar only)
    str(ROOT / "outputs/04_clean_bev_lidar_only"):
        "outputs/04_clean_bev_lidar_only",
    # Old shadeform paths for condD
    "/home/shadeform/moral_pipeline/outputs/condition_D_lidar_bev_radar":
        "outputs/03_clean_bev",
    # Old shadeform paths for condB
    "/home/shadeform/moral_pipeline/outputs/condition_B_lidar_bev":
        "outputs/04_clean_bev_lidar_only",
    # Local condB GT paths
    str(ROOT / "outputs/01_gt_annotations"):
        "outputs/01_gt_annotations",
    str(ROOT / "outputs/02_gt_with_radar"):
        "outputs/02_gt_with_radar",
}

# Files to normalize
JSONL_FILES = [
    "02_cosmos_integration/hf_data/clean_conditionD_train.jsonl",
    "02_cosmos_integration/hf_data/clean_conditionD_val.jsonl",
    "02_cosmos_integration/hf_data/clean_conditionB_train.jsonl",
    "02_cosmos_integration/hf_data/clean_conditionB_val.jsonl",
    "02_cosmos_integration/hf_data/local_conditionD_val_zeroshot_v2.jsonl",
    "02_cosmos_integration/hf_data/local_conditionB_val_zeroshot_v2.jsonl",
]

normalized_dir = ROOT / "hf_upload" / "jsonl"
normalized_dir.mkdir(parents=True, exist_ok=True)

for rel_path in JSONL_FILES:
    src = ROOT / rel_path
    if not src.exists():
        print(f"  ⚠️  Not found, skipping: {rel_path}")
        continue

    dst = normalized_dir / src.name
    changed = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            original = line
            for old, new in PATH_REWRITES.items():
                line = line.replace(old, new)
            if line != original:
                changed += 1
            fout.write(line)

    total = sum(1 for _ in open(dst))
    print(f"  ✅ {src.name}: {total} records, {changed} paths rewritten → {dst}")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Upload JSONL files to HuggingFace Datasets
# ═══════════════════════════════════════════════════════════════════════════════
print("[2/4] Uploading JSONL files to HuggingFace Datasets...")

# Create dataset repo if it doesn't exist
try:
    api.create_repo(repo_id=DATASET_ID, repo_type="dataset", exist_ok=True)
    print(f"  Dataset repo ready: {DATASET_ID}")
except Exception as e:
    print(f"  ⚠️  Could not create repo: {e}")

# Upload each normalized JSONL
for jsonl_file in normalized_dir.glob("*.jsonl"):
    remote_path = f"data/{jsonl_file.name}"
    print(f"  Uploading {jsonl_file.name} → {remote_path} ...")
    api.upload_file(
        path_or_fileobj=str(jsonl_file),
        path_in_repo=remote_path,
        repo_id=DATASET_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"    ✅ Done")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Upload BEV images (optional, slow)
# ═══════════════════════════════════════════════════════════════════════════════
if not args.skip_images:
    print("[3/4] Uploading BEV images to HuggingFace Datasets...")
    print("  This uploads ~850 scenes × 2 folders × 2 images = ~3400 files")
    print("  Use --skip_images to skip and distribute images separately\n")

    for bev_dir, hf_prefix in [
        (ROOT / "outputs/03_clean_bev",            "images/03_clean_bev"),
        (ROOT / "outputs/04_clean_bev_lidar_only", "images/04_clean_bev_lidar_only"),
    ]:
        if not bev_dir.exists():
            print(f"  ⚠️  Not found: {bev_dir}")
            continue

        scenes = sorted(bev_dir.glob("scene-*/"))
        print(f"  Uploading {len(scenes)} scenes from {bev_dir.name}...")

        for scene_dir in scenes:
            scene = scene_dir.name
            for img_name in ["bev_map.png", "CAM_FRONT.jpg"]:
                img_path = scene_dir / img_name
                if not img_path.exists():
                    continue
                api.upload_file(
                    path_or_fileobj=str(img_path),
                    path_in_repo=f"{hf_prefix}/{scene}/{img_name}",
                    repo_id=DATASET_ID,
                    repo_type="dataset",
                    token=HF_TOKEN,
                )

        print(f"    ✅ {bev_dir.name} uploaded")
else:
    print("[3/4] Skipping BEV image upload (--skip_images set)")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Upload fine-tuned model to HuggingFace Hub
# ═══════════════════════════════════════════════════════════════════════════════
if not args.skip_model:
    model_path = ROOT / args.model_path
    if not model_path.exists():
        print(f"[4/4] ⚠️  Model path not found: {model_path} — skipping")
    else:
        print(f"[4/4] Uploading fine-tuned model to HuggingFace Hub...")
        print(f"  Source: {model_path}")
        print(f"  Destination: {MODEL_ID}")

        try:
            api.create_repo(repo_id=MODEL_ID, repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"  ⚠️  Could not create model repo: {e}")

        api.upload_folder(
            folder_path=str(model_path),
            repo_id=MODEL_ID,
            repo_type="model",
            token=HF_TOKEN,
        )
        print(f"  ✅ Model uploaded to https://huggingface.co/{MODEL_ID}")
else:
    print("[4/4] Skipping model upload (--skip_model set)")

print(f"\n{'='*60}")
print(f" Upload Complete!")
print(f"  Dataset: https://huggingface.co/datasets/{DATASET_ID}")
print(f"  Model:   https://huggingface.co/{MODEL_ID}")
print(f"{'='*60}")
print(f"\nShare these URLs in your README for full reproducibility.")
