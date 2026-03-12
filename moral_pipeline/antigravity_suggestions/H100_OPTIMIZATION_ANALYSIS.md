# H100 Optimization Analysis — `train_cosmos8b.py`

## Overall Assessment: ✅ Good, with improvements available

The existing `train_cosmos8b.py` is well-structured and will **work correctly** on an H100.
It already includes many important optimizations. Below is a detailed analysis of what's
good, what's suboptimal, and what the unified trainer (`train_cosmos_unified.py`) fixes.

---

## ✅ What `train_cosmos8b.py` Already Does Right

| Feature | Status | Notes |
|---------|--------|-------|
| 4-bit QLoRA (NF4 + double quant) | ✅ | Correct — minimizes VRAM |
| BF16 autocast | ✅ | Best for H100 (native BF16 support) |
| FlashAttention-2 auto-fallback | ✅ | Tries FA2, falls back to SDPA |
| TF32 toggle | ✅ | `--tf32` flag, should be ON for H100 |
| Fused AdamW | ✅ | Automatically uses fused when available |
| Gradient checkpointing | ✅ | `--grad_ckpt` flag |
| Pin memory + persistent workers | ✅ | Good for GPU-CPU overlap |
| Checkpoint rotation | ✅ | `--save_total_limit` prevents disk fill |
| Full trainer state resume | ✅ | Optimizer + scheduler + RNG states |
| Gradient clipping (1.0) | ✅ | Prevents gradient explosion |
| Cosine LR schedule with warmup | ✅ | Standard best practice |
| Non-blocking CUDA transfers | ✅ | `non_blocking=True` in batch move |

---

## ⚠️ Issues Found (Fixed in Unified Trainer)

### 1. **Docstring / defaults say "2B" but file is for 8B**
The file header still says `Cosmos-Reason2-2B` and defaults to `nvidia/Cosmos-Reason2-2B`:
```python
# Line 2-6: Says "Fine-tuning nvidia/Cosmos-Reason2-2B"
# Line 36: default="nvidia/Cosmos-Reason2-2B"  ← wrong
# Line 35: default="saves/cosmos2b_condD_finetuned"  ← wrong
```
**Impact:** User must always pass `--model nvidia/Cosmos-Reason2-8B` manually.
**Fix:** Unified trainer defaults to 8B.

### 2. **`--grad_ckpt` is OFF by default — will OOM on 48GB GPUs**
For 8B model, gradient checkpointing should be ON by default. Without it:
- 8B 4-bit needs ~8GB base weight
- Activations for 4096 tokens with images can reach 20-30GB
- Total: ~38-48GB without grad_ckpt → borderline on 80GB, OOM on 48GB

**Fix:** Unified trainer enables grad_ckpt by default for 8B models.

### 3. **No SIGTERM handler — cloud preemption loses progress**
Spot/preemptible instances send SIGTERM before shutdown. Without a handler,
the current training step's progress is lost.

**Fix:** Unified trainer catches SIGTERM and saves an emergency checkpoint.

### 4. **Checkpoint save is not atomic**
If the process is killed mid-save (power loss, OOM, preemption), the checkpoint
directory may be partially written and corrupt. Next `--resume` will fail.

**Fix:** Unified trainer writes to a temp directory, then atomic renames.

### 5. **No gradient scaler state saved in checkpoints**
When using `--precision fp16`, the `GradScaler` state is not saved/restored.
After resume, the scaler restarts from default — can cause loss spikes.
```python
# train_cosmos8b.py saves:
#   optimizer, scheduler, global_step, epoch, best_val_loss, rng states
# Missing: scaler.state_dict()
```
**Fix:** Unified trainer saves scaler state in `trainer_state.pt`.

### 6. **`--torch_compile` applied AFTER wrapping with PeftModel**
`torch.compile` on the outer PeftModel may not optimize as well as compiling
the inner base model. However, this is a known limitation of PEFT + compile
and may cause issues on some torch versions.

**Fix:** Unified trainer wraps with try/except and warns if compile fails.

### 7. **Estimated time calculation uses 4090 numbers**
```python
est_hrs = total_steps * 2.5 / 3600   # ~2.5s/step for 2B on 4090
```
This underestimates for 8B and overestimates for H100.

**Fix:** Unified trainer uses model-size and profile-aware estimates.

### 8. **Images processed twice in collator (wasteful)**
The collator calls `processor()` twice — once for `full_text` and once for
`prompt_text` — both with the same images. This doubles image encode time.
```python
inputs = processor(text=full_text, images=images, ...)        # ← encodes images
prompt_inputs = processor(text=prompt_text, images=images, ...)  # ← encodes same images again
```
**Impact:** ~2× slower collation. For H100 with fast GPU, this CPU bottleneck matters.

**Possible fix:** Tokenize text manually and use only one processor call for images.
However, this requires deeper changes to handle image token insertion correctly.
The unified trainer keeps the same approach for correctness but mitigates with
higher `num_workers` and `prefetch_factor`.

### 9. **`set_to_none=True` not used in 2B trainer**
The 2B trainer uses the default `optimizer.zero_grad()` instead of
`optimizer.zero_grad(set_to_none=True)`. The 8B trainer correctly uses it.
Small performance difference but matters over thousands of steps.

---

## 🚀 H100-Specific Recommendations

### Enable These Flags
```bash
python train_cosmos8b.py \
    --model nvidia/Cosmos-Reason2-8B \
    --tf32 \
    --torch_compile \
    --grad_ckpt \
    --batch_size 2 \
    --grad_accum 8 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --precision bf16 \
    --attn_impl auto   # auto → FlashAttention-2
```

### Why These Matter on H100
| Flag | H100 Benefit |
|------|-------------|
| `--tf32` | H100 FP32 tensor cores use TF32 by default — 3× faster matmul |
| `--torch_compile` | Inductor compiler fuses kernels — 10-20% speedup |
| `--grad_ckpt` | Saves ~40% activation memory — allows batch_size 2 |
| `--batch_size 2` | Double throughput vs batch_size 1 |
| `--precision bf16` | H100 has 2× BF16 FLOPs vs FP16, no scaler needed |
| `--attn_impl auto` | FlashAttention-2 on H100 is 2-3× faster than SDPA |
| `--num_workers 8` | H100 instances usually have 64+ CPU cores — use them |

### Expected Performance on H100 80GB
| Metric | Value |
|--------|-------|
| VRAM usage | ~35-45 GB (4-bit + grad_ckpt + batch 2) |
| Throughput | ~3-4 seconds/step |
| 2000 samples, 3 epochs, effective batch 16 | ~375 steps → ~25 min |
| 5000 samples, 3 epochs, effective batch 16 | ~940 steps → ~63 min |

---

## Summary of Changes in Unified Trainer

| What | `train_cosmos8b.py` | `train_cosmos_unified.py` |
|------|---------------------|--------------------------|
| Models supported | 8B (with wrong 2B defaults) | Both 2B and 8B |
| GPU profiles | Manual flags | Auto-detect or `--profile h100` |
| Default for 8B | grad_ckpt OFF | grad_ckpt ON |
| SIGTERM handler | ❌ | ✅ Emergency checkpoint |
| Atomic checkpoint saves | ❌ | ✅ Temp dir + rename |
| GradScaler state in resume | ❌ | ✅ |
| HF Hub retry (429) | ❌ | ✅ Exponential backoff |
| HF upload after training | ❌ | ✅ `--hf_upload_repo` |
| ETA logging | Basic | Accurate with samples/sec |
| Time estimate | 4090-based | Profile-aware |
