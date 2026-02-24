"""
moral_pipeline/02_cosmos_integration/local_vlm_backend.py
==========================================================
Drop-in local backend for generate_cosmos_qa.py
Uses Qwen2.5-VL-7B (4-bit) on your RTX 4070 8GB laptop.

Cosmos Reason 2 is post-trained on Qwen3-VL (same family as Qwen2.5-VL).
Same prompt format, same <think>/<answer> output structure.

USAGE:
  # First run — downloads ~15GB of weights:
  python local_vlm_backend.py --download

  # Smoke test on one BEV image:
  python local_vlm_backend.py --test moral_pipeline/outputs/01_gt_annotations/scene-0061/bev_map.png

  # generate_cosmos_qa.py uses this automatically with --api-mode local-hf
"""

import os
import sys
import argparse
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the question in the following format: "
    "<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."
)

_model = None
_processor = None


def load_model(verbose: bool = True) -> tuple:
    """Load Qwen2.5-VL-7B with 4-bit quantization. RTX 4070 8GB: uses ~6.5GB VRAM."""
    global _model, _processor

    if _model is not None:
        return _model, _processor

    if verbose:
        print(f"Loading {MODEL_ID} (4-bit quantized)...")
        print("First run downloads ~15GB to ~/.cache/huggingface/")

    import torch
    from transformers import (
        AutoProcessor,
        Qwen2VLForConditionalGeneration,
        BitsAndBytesConfig,
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Processor — trust_remote_code needed for Qwen2.5-VL
    _processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    # Model with 4-bit quant
    _model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    _model.eval()

    if verbose:
        try:
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved  = torch.cuda.memory_reserved(0) / 1e9
            print(f"Model loaded. VRAM: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        except Exception:
            print("Model loaded.")

    return _model, _processor


def call_cosmos_local(
    bev_image_b64: str,
    scene_description: str,
    question: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
    verbose: bool = False,
) -> dict:
    """
    Drop-in replacement for call_cosmos() in generate_cosmos_qa.py.

    Input:
      bev_image_b64    : base64-encoded PNG of the BEV map
      scene_description: text from scene_description.txt
      question         : one question string

    Output:
      {"reasoning": str, "answer": str, "raw_response": str, "tokens_used": int}
    """
    import torch
    import base64
    from io import BytesIO
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    model, processor = load_model(verbose=verbose)

    # Decode base64 → PIL image
    img_bytes = base64.b64decode(bev_image_b64)
    bev_image = Image.open(BytesIO(img_bytes)).convert("RGB")

    # Qwen2.5-VL message format — image FIRST, then text
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": bev_image,   # PIL Image directly
                },
                {
                    "type": "text",
                    "text": (
                        f"SCENE SENSOR DATA:\n{scene_description}\n\n"
                        f"QUESTION: {question}"
                    ),
                },
            ],
        },
    ]

    # Build text prompt with chat template
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Extract image tensors using qwen_vl_utils
    image_inputs, video_inputs = process_vision_info(messages)

    # Tokenise
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    elapsed = time.time() - t0

    # Decode new tokens only
    generated = output_ids[:, input_len:]
    raw = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    tokens_generated = generated.shape[1]

    if verbose:
        print(f"  {tokens_generated} tokens in {elapsed:.1f}s  "
              f"({tokens_generated/elapsed:.1f} tok/s)")

    # Parse <think> and <answer> blocks
    reasoning = ""
    answer = raw

    if "<think>" in raw and "</think>" in raw:
        t_start = raw.index("<think>") + len("<think>")
        t_end   = raw.index("</think>")
        reasoning = raw[t_start:t_end].strip()
        after_think = raw[t_end + len("</think>"):]
        if "<answer>" in after_think and "</answer>" in after_think:
            a_start = after_think.index("<answer>") + len("<answer>")
            a_end   = after_think.index("</answer>")
            answer  = after_think[a_start:a_end].strip()
        else:
            answer = after_think.strip()

    return {
        "reasoning":    reasoning,
        "answer":       answer,
        "raw_response": raw,
        "tokens_used":  tokens_generated,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Local Qwen2.5-VL-7B backend for MoRAL QA generation"
    )
    parser.add_argument("--download", action="store_true",
                        help="Pre-download model weights (~15GB)")
    parser.add_argument("--test", type=str, default=None,
                        help="Path to bev_map.png for smoke test")
    args = parser.parse_args()

    if args.download:
        print(f"Pre-downloading {MODEL_ID}...")
        load_model(verbose=True)
        print("Done. Model cached at ~/.cache/huggingface/")
        return

    if args.test:
        import base64
        bev_path = Path(args.test)
        if not bev_path.exists():
            print(f"ERROR: {bev_path} not found")
            sys.exit(1)

        print(f"Smoke test: {bev_path}")
        with open(bev_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode()

        scene_desc = (
            "EGO STATE: Speed 7.4 m/s (26.6 km/h). Stopping distance: 6.8m.\n"
            "NEAREST OBJECT PER ZONE: directly ahead: car at 20.86m; "
            "right: truck at 7.52m stationary.\n"
            "SAFETY CRITICAL: truck right 7.52m (TTC ~1.0s)."
        )

        result = call_cosmos_local(
            bev_image_b64=b64,
            scene_description=scene_desc,
            question=(
                "What is the nearest object directly ahead of the ego vehicle? "
                "State its class, distance in metres, and bearing."
            ),
            verbose=True,
        )

        print("\n── REASONING ──────────────────────────────────────────")
        r = result["reasoning"]
        print(r[:800] + "\n...[truncated]" if len(r) > 800 else r)
        print("\n── ANSWER ─────────────────────────────────────────────")
        print(result["answer"])
        print(f"\nTotal tokens: {result['tokens_used']}")
        return

    print("Usage:")
    print("  python local_vlm_backend.py --download")
    print("  python local_vlm_backend.py --test <path/to/bev_map.png>")


if __name__ == "__main__":
    main()
