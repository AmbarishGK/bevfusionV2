#!/usr/bin/env python3
"""
convert_to_llama_factory.py
============================
Converts MoRAL ShareGPT JSONL output -> LLaMA-Factory format for Qwen2.5-VL-7B.
"""

import json
import argparse
from pathlib import Path


def convert_record(rec: dict) -> dict:
    images = rec.get("images", [])
    messages = rec["messages"]

    user_msg = messages[0]
    assert user_msg["role"] == "user", f"Expected user first, got {user_msg['role']}"

    text = user_msg["content"]
    for _ in images:
        text = text.replace("<image>\n", "", 1)
    text = text.strip()

    new_content = [{"type": "image", "image": str(p)} for p in images]
    new_content.append({"type": "text", "text": text})

    new_messages = [
        {"role": "user", "content": new_content},
        messages[1],
    ]

    out = {"messages": new_messages}
    if "_meta" in rec:
        out["_meta"] = rec["_meta"]
    return out


def convert_file(in_path: Path, out_path: Path, check: bool = False) -> int:
    if not in_path.exists():
        print(f"  SKIP (not found): {in_path}")
        return 0

    count = 0
    errors = 0
    with open(in_path) as fin, open(out_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                converted = convert_record(rec)
                fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1
                if check and i == 0:
                    print(f"\n--- First record sample ({in_path.name}) ---")
                    print(json.dumps(converted, indent=2, ensure_ascii=False)[:2000])
                    print("---")
            except Exception as e:
                errors += 1
                print(f"  ERROR on line {i}: {e}")

    print(f"  {in_path.name} -> {out_path.name}: {count} records, {errors} errors")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Print first converted record and exit")
    parser.add_argument("--root", default="/home/shadeform/moral_pipeline", help="Pipeline root")
    args = parser.parse_args()

    root = Path(args.root)

    pairs = [
        ("all_conditionB_sharegpt.jsonl",      "all_conditionB_llama_factory.jsonl"),
        ("all_conditionB_train_sharegpt.jsonl", "all_conditionB_train_llama_factory.jsonl"),
        ("all_conditionB_val_sharegpt.jsonl",   "all_conditionB_val_llama_factory.jsonl"),
        ("all_conditionD_sharegpt.jsonl",       "all_conditionD_llama_factory.jsonl"),
        ("all_conditionD_train_sharegpt.jsonl", "all_conditionD_train_llama_factory.jsonl"),
        ("all_conditionD_val_sharegpt.jsonl",   "all_conditionD_val_llama_factory.jsonl"),
    ]

    total = 0
    for in_name, out_name in pairs:
        total += convert_file(root / in_name, root / out_name, check=args.check)
        if args.check:
            break

    print(f"\nTotal records converted: {total}")


if __name__ == "__main__":
    main()
