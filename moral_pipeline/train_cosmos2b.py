#!/usr/bin/env python3
"""
MoRAL — Fine-tuning nvidia/Cosmos-Reason2-2B (Qwen3VL architecture)
=====================================================================
Uses HuggingFace TRL + PEFT (no Unsloth — Cosmos-Reason2 not supported).
4-bit QLoRA on RTX 4090 (24GB). Expects ~16-18GB VRAM.

Quick start (condD):
    python train_cosmos2b.py \
        --train_file 02_cosmos_integration/hf_data/clean_conditionD_train.jsonl \
        --val_file   02_cosmos_integration/hf_data/clean_conditionD_val.jsonl \
        --output_dir saves/cosmos2b_condD_finetuned

Dry run (tests full pipeline on 20 records, 2 steps):
    python train_cosmos2b.py ... --dry_run

Resume:
    python train_cosmos2b.py ... --resume

Install:
    pip install peft trl bitsandbytes accelerate --break-system-packages
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_file",  default="02_cosmos_integration/hf_data/clean_conditionD_train.jsonl")
parser.add_argument("--val_file",    default="02_cosmos_integration/hf_data/clean_conditionD_val.jsonl")
parser.add_argument("--output_dir",  default="saves/cosmos2b_condD_finetuned")
parser.add_argument("--model",       default="nvidia/Cosmos-Reason2-2B")
parser.add_argument("--epochs",      type=int,   default=3)
parser.add_argument("--batch_size",  type=int,   default=1,      help="Per-device batch size (keep 1 for 4090)")
parser.add_argument("--grad_accum",  type=int,   default=16,     help="Gradient accumulation steps")
parser.add_argument("--lr",          type=float, default=2e-4)
parser.add_argument("--lora_rank",   type=int,   default=16)
parser.add_argument("--max_length",  type=int,   default=4096,   help="Max token sequence length")
parser.add_argument("--max_pixels",  type=int,   default=512*512, help="Cap image resolution in pixels")
parser.add_argument("--wandb_key",   default=None)
parser.add_argument("--dry_run",     action="store_true")
parser.add_argument("--resume",      action="store_true")
args = parser.parse_args()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(" MoRAL Cosmos-Reason2-2B Training — Pre-flight Checks")
print("="*60)

errors   = []
warnings = []

print("\n[1/6] Checking packages...")
try:
    import torch
    print(f"  torch:        {torch.__version__}")
    if not torch.cuda.is_available():
        errors.append("CUDA not available")
    else:
        gpu     = torch.cuda.get_device_properties(0)
        vram_gb = gpu.total_memory / 1024**3
        free_gb = (gpu.total_memory - torch.cuda.memory_reserved()) / 1024**3
        print(f"  GPU:          {gpu.name}")
        print(f"  VRAM:         {vram_gb:.1f} GB total, {free_gb:.1f} GB free")
        if vram_gb < 20:
            errors.append(f"GPU has only {vram_gb:.1f}GB VRAM — need ≥20GB")
except ImportError as e:
    errors.append(f"torch not found: {e}")

for pkg in ["transformers", "peft", "trl", "bitsandbytes", "PIL"]:
    try:
        import importlib
        m   = importlib.import_module(pkg if pkg != "PIL" else "PIL")
        ver = getattr(m, "__version__", "unknown")
        print(f"  {pkg+':':<14} {ver}")
    except ImportError as e:
        errors.append(f"{pkg} not found: {e}")

print("\n[2/6] Checking input files...")
for label, path in [("train", args.train_file), ("val", args.val_file)]:
    if not os.path.exists(path):
        errors.append(f"{label} file not found: {path}")
    else:
        size_mb = os.path.getsize(path) / 1e6
        with open(path) as f:
            n = sum(1 for _ in f)
        print(f"  {label}: {n} records, {size_mb:.1f} MB — {path}")

print("\n[3/6] Checking output dir...")
os.makedirs(args.output_dir, exist_ok=True)
checkpoints = sorted(Path(args.output_dir).glob("checkpoint-*"))
if checkpoints:
    print(f"  Found {len(checkpoints)} checkpoint(s). Latest: {checkpoints[-1].name}")
    if not args.resume:
        warnings.append("Checkpoints exist but --resume not set. Will restart from scratch.")
    else:
        print(f"  Will resume from: {checkpoints[-1]}")
else:
    print(f"  No checkpoints found. Starting fresh.")

print("\n[4/6] Checking image paths (first 5 records)...")
from PIL import Image
broken = []
try:
    with open(args.train_file) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            rec = json.loads(line)
            for part in rec["messages"][0]["content"]:
                if part.get("type") == "image":
                    p = part["image"]
                    if not os.path.exists(p):
                        broken.append(p)
                    elif i == 0:
                        img = Image.open(p)
                        print(f"  {os.path.basename(p)}: {img.size} px ✅")
    if broken:
        errors.append(f"Broken image paths: {broken[:3]}")
    else:
        print(f"  All sampled paths resolve ✅")
except Exception as e:
    errors.append(f"Image path check failed: {e}")

print("\n[5/6] Checking system RAM...")
try:
    with open("/proc/meminfo") as f:
        info = dict(ln.split(":") for ln in f.read().splitlines() if ":" in ln)
    avail_gb = int(info["MemAvailable"].strip().split()[0]) / 1e6
    print(f"  Available RAM: {avail_gb:.1f} GB")
    if avail_gb < 16:
        warnings.append(f"Only {avail_gb:.1f}GB RAM — may slow during data loading")
except Exception:
    print("  Could not read RAM info")

print("\n[6/6] Training config...")
with open(args.train_file) as f:
    n_train = sum(1 for _ in f)
effective_batch = args.batch_size * args.grad_accum
if args.dry_run:
    print(f"  ⚡ DRY RUN — 20 samples, 2 steps only")
else:
    steps_per_epoch = max(1, n_train // effective_batch)
    total_steps     = steps_per_epoch * args.epochs
    est_hrs         = total_steps * 2.5 / 3600   # ~2.5s/step for 2B on 4090
    print(f"  model:         {args.model}")
    print(f"  lora_rank:     {args.lora_rank}")
    print(f"  batch:         {args.batch_size} × accum {args.grad_accum} = effective {effective_batch}")
    print(f"  epochs:        {args.epochs}")
    print(f"  max_length:    {args.max_length}")
    print(f"  steps/epoch:   {steps_per_epoch}")
    print(f"  total steps:   {total_steps}")
    print(f"  est. time:     ~{est_hrs:.1f} hrs on RTX 4090")

print("\n" + "─"*60)
for w in warnings:
    print(f"  ⚠️  {w}")
if errors:
    for e in errors:
        print(f"  ❌ {e}")
    print("\nPre-flight FAILED. Fix errors above.\n")
    sys.exit(1)
print("  ✅ Pre-flight passed")
print("─"*60 + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — W&B
# ═══════════════════════════════════════════════════════════════════════════════
if args.wandb_key:
    import wandb
    wandb.login(key=args.wandb_key)
    os.environ["WANDB_PROJECT"] = "MoRAL-finetune"
    report_to = "wandb"
    print("W&B logging enabled\n")
else:
    os.environ["WANDB_DISABLED"] = "true"
    report_to = "none"

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD MODEL (4-bit QLoRA)
# ═══════════════════════════════════════════════════════════════════════════════
import torch
import transformers
from transformers import BitsAndBytesConfig, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

print("Loading model in 4-bit...")
t0 = time.time()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
    args.model,
    quantization_config=bnb_config,
    device_map="cuda",
    attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained(args.model)

print(f"  Loaded in {time.time()-t0:.1f}s | VRAM: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# Enable gradient checkpointing before LoRA
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

print(f"Adding LoRA (rank={args.lora_rank})...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_rank,
    lora_alpha=args.lora_rank * 2,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print(f"  VRAM after LoRA: {torch.cuda.memory_reserved()/1024**3:.2f} GB\n")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATASET
# ═══════════════════════════════════════════════════════════════════════════════
from torch.utils.data import Dataset

class MoRALDataset(Dataset):
    """
    Loads JSONL records. Each record has:
      messages[0] = user turn: list of {type: image/text, image: path, text: str}
      messages[1] = assistant turn: {content: str}
    Images are loaded per-item (lazy).
    """
    def __init__(self, path, max_records=None):
        self.records = []
        skipped  = 0
        bad_imgs = []
        with open(path) as f:
            for i, line in enumerate(f):
                if max_records and i >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ans = str(rec["messages"][1]["content"]) if len(rec["messages"]) > 1 else ""
                    if "[ERROR]" in ans or "All 3 attempts failed" in ans:
                        skipped += 1
                        continue
                    # Validate image paths
                    valid = True
                    for part in rec["messages"][0]["content"]:
                        if part.get("type") == "image" and not os.path.exists(part["image"]):
                            valid = False
                            if len(bad_imgs) < 3:
                                bad_imgs.append(part["image"])
                            break
                    if valid:
                        self.records.append(rec)
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
        print(f"  {os.path.basename(path)}: {len(self.records)} loaded, {skipped} skipped")
        if bad_imgs:
            print(f"  Sample broken paths: {bad_imgs}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


print("Loading datasets...")
if args.dry_run:
    train_ds = MoRALDataset(args.train_file, max_records=20)
    val_ds   = MoRALDataset(args.val_file,   max_records=10)
else:
    train_ds = MoRALDataset(args.train_file)
    val_ds   = MoRALDataset(args.val_file)
print(f"  Ready: {len(train_ds)} train / {len(val_ds)} val\n")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — COLLATOR
# ═══════════════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    """
    Convert raw JSONL records → model inputs.
    Builds Qwen3VL chat format: [system], user (images + text), assistant.
    Labels: -100 for prompt tokens, real ids for assistant response only.
    """
    all_input_ids      = []
    all_attention_mask = []
    all_labels         = []

    for rec in batch:
        # Build user content list for processor
        user_content = []
        images       = []
        for part in rec["messages"][0]["content"]:
            if part["type"] == "image":
                img = Image.open(part["image"]).convert("RGB")
                # Resize to cap max_pixels while keeping aspect ratio
                w, h   = img.size
                pixels = w * h
                if pixels > args.max_pixels:
                    scale = (args.max_pixels / pixels) ** 0.5
                    img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                images.append(img)
                user_content.append({"type": "image"})
            else:
                user_content.append({"type": "text", "text": part["text"]})

        assistant_text = str(rec["messages"][1]["content"])

        messages = [
            {"role": "system", "content": [{"type": "text",
                "text": "You are a helpful autonomous driving assistant that reasons about BEV sensor data."}]},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]

        # Tokenize full conversation
        full_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize prompt only (to find where assistant starts)
        prompt_messages = messages[:-1]
        prompt_text = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process with images
        inputs = processor(
            text=full_text,
            images=images if images else None,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )
        prompt_inputs = processor(
            text=prompt_text,
            images=images if images else None,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )

        input_ids = inputs["input_ids"][0]
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Labels: -100 for prompt, real ids for response
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        all_input_ids.append(input_ids)
        all_attention_mask.append(inputs["attention_mask"][0])
        all_labels.append(labels)

    # Pad to longest in batch
    max_len = max(x.shape[0] for x in all_input_ids)
    pad_id  = processor.tokenizer.pad_token_id or 0

    padded_ids   = torch.full((len(batch), max_len), pad_id,  dtype=torch.long)
    padded_mask  = torch.zeros((len(batch), max_len),          dtype=torch.long)
    padded_labels= torch.full((len(batch), max_len), -100,    dtype=torch.long)

    for i, (ids, mask, lbl) in enumerate(zip(all_input_ids, all_attention_mask, all_labels)):
        L = ids.shape[0]
        padded_ids[i, :L]    = ids
        padded_mask[i, :L]   = mask
        padded_labels[i, :L] = lbl

    return {
        "input_ids":      padded_ids,
        "attention_mask": padded_mask,
        "labels":         padded_labels,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PRE-TRAINING INFERENCE TEST
# ═══════════════════════════════════════════════════════════════════════════════
print("[Pre-training inference test] Verifying model can process a BEV+CAM sample...")
try:
    model.eval()
    sample = train_ds[0]
    images = []
    user_content = []
    for part in sample["messages"][0]["content"]:
        if part["type"] == "image":
            images.append(Image.open(part["image"]).convert("RGB"))
            user_content.append({"type": "image"})
        else:
            user_content.append({"type": "text", "text": part["text"][:200]})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful autonomous driving assistant."}]},
        {"role": "user",   "content": user_content},
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=images if images else None,
                       return_tensors="pt").to("cuda")
    print(f"  Input tokens: {inputs['input_ids'].shape[1]} | VRAM: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    with torch.no_grad():
        out  = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    pred = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Output: {pred[:200]}")
    print(f"  ✅ Pre-training inference OK\n")
except Exception as e:
    print(f"  ⚠️  Inference test failed (non-fatal): {e}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW

model.train()

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  collate_fn=collate_fn, num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=1,
                          shuffle=False, collate_fn=collate_fn, num_workers=2)

# Optimizer — only trainable (LoRA) params
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr, weight_decay=0.01,
)

total_steps   = (2 if args.dry_run else
                 len(train_loader) * args.epochs // args.grad_accum)
warmup_steps  = max(1, int(total_steps * 0.05))
scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

gpu_stats  = torch.cuda.get_device_properties(0)
max_mem_gb = round(gpu_stats.total_memory / 1024**3, 2)
print(f"\n{'─'*60}")
print(f"  {'DRY RUN — ' if args.dry_run else ''}Training started")
print(f"  {len(train_ds)} records | effective batch {args.batch_size * args.grad_accum}")
print(f"  GPU: {gpu_stats.name} | {max_mem_gb} GB")
print(f"{'─'*60}\n")

scaler       = torch.cuda.amp.GradScaler(enabled=False)  # bf16 doesn't need scaler
global_step  = 0
best_val_loss= float("inf")
t_start      = time.time()

try:
    for epoch in range(1 if args.dry_run else args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            if args.dry_run and global_step >= 2:
                break

            batch = {k: v.to("cuda") for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss    = outputs.loss / args.grad_accum

            loss.backward()
            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0 or args.dry_run:
                    elapsed = (time.time() - t_start) / 60
                    vram    = torch.cuda.memory_reserved() / 1024**3
                    print(f"  step {global_step:>5} | loss {epoch_loss/(step+1):.4f} "
                          f"| lr {scheduler.get_last_lr()[0]:.2e} "
                          f"| {elapsed:.1f}min | VRAM {vram:.1f}GB")

                # Save checkpoint every 200 steps
                if not args.dry_run and global_step % 200 == 0:
                    ckpt = Path(args.output_dir) / f"checkpoint-{global_step}"
                    model.save_pretrained(str(ckpt))
                    processor.save_pretrained(str(ckpt))
                    print(f"  💾 Saved checkpoint-{global_step}")

        # Validation at end of each epoch
        if not args.dry_run:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = {k: v.to("cuda") for k, v in vbatch.items()}
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        vout = model(**vbatch)
                    val_loss += vout.loss.item()
            val_loss /= max(1, len(val_loader))
            print(f"\n  📊 Epoch {epoch+1} val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = Path(args.output_dir) / "best_model"
                model.save_pretrained(str(best_path))
                processor.save_pretrained(str(best_path))
                print(f"  ✅ New best val loss {val_loss:.4f} — saved to best_model/\n")

except torch.cuda.OutOfMemoryError:
    used = torch.cuda.memory_reserved() / 1024**3
    print(f"\n❌ OOM — {used:.1f} GB / {max_mem_gb} GB used")
    print("Try:")
    print("  --lora_rank 8          (fewer LoRA params)")
    print("  --max_length 2048      (shorter sequences)")
    print("  --max_pixels 262144    (512×512 max)")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — POST-TRAINING REPORT
# ═══════════════════════════════════════════════════════════════════════════════
runtime  = round((time.time() - t_start) / 60, 1)
used_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)

print(f"\n{'='*60}")
print(f" {'DRY RUN ' if args.dry_run else ''}Training Complete")
print(f"{'='*60}")
print(f"  Time:       {runtime} min")
print(f"  Peak VRAM:  {used_mem} GB / {max_mem_gb} GB ({round(used_mem/max_mem_gb*100,1)}%)")
if not args.dry_run:
    print(f"  Best val loss: {best_val_loss:.4f}")

# Post-training inference check
print("\n[Post-training inference] Checking output on val sample...")
try:
    model.eval()
    sample = val_ds[0]
    images = []
    user_content = []
    for part in sample["messages"][0]["content"]:
        if part["type"] == "image":
            images.append(Image.open(part["image"]).convert("RGB"))
            user_content.append({"type": "image"})
        else:
            user_content.append({"type": "text", "text": part["text"][:300]})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful autonomous driving assistant."}]},
        {"role": "user",   "content": user_content},
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=images if images else None,
                       return_tensors="pt").to("cuda")
    with torch.no_grad():
        out  = model.generate(**inputs, max_new_tokens=200, use_cache=True)
    pred = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    gt   = str(sample["messages"][1]["content"])[:300]
    print(f"  Prediction:   {pred[:300]}")
    print(f"  Ground truth: {gt[:300]}")
    print(f"  ✅ Post-training inference OK")
except Exception as e:
    print(f"  ⚠️  Post-training inference failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SAVE FINAL
# ═══════════════════════════════════════════════════════════════════════════════
if not args.dry_run:
    print(f"\nSaving final LoRA to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\n✅ Done. Next step:")
    print(f"   python evaluate_zeroshot.py \\")
    print(f"     --model {args.output_dir} \\")
    print(f"     --val_file {args.val_file}")
else:
    print("\n✅ Dry run complete — full pipeline works.")
    print("   Remove --dry_run to start real training.")
