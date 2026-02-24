"""
Run this once to patch generate_cosmos_qa.py to support --api-mode local-hf.
Usage: python patch_generate_cosmos_qa.py
"""
from pathlib import Path

script = Path("moral_pipeline/02_cosmos_integration/generate_cosmos_qa.py")
if not script.exists():
    print(f"ERROR: {script} not found. Run from bevfusionV2/ root.")
    raise SystemExit(1)

text = script.read_text()

# ── Patch 1: add local-hf to api_mode choices ─────────────────────────────
old1 = '''    parser.add_argument(
        "--api-mode",
        choices=["nim", "local"],
        default="nim",'''
new1 = '''    parser.add_argument(
        "--api-mode",
        choices=["nim", "local", "local-hf"],
        default="local-hf",'''

# ── Patch 2: add local-hf branch inside call_cosmos() ─────────────────────
old2 = '''    elif api_mode == "local":
        url = base_url or os.environ.get("COSMOS_BASE_URL", "http://localhost:8000/v1")
        client = OpenAI(base_url=url, api_key="not-needed")
    else:
        raise ValueError(f"Unknown api_mode: {api_mode}. Use 'nim' or 'local'")'''
new2 = '''    elif api_mode == "local":
        url = base_url or os.environ.get("COSMOS_BASE_URL", "http://localhost:8000/v1")
        client = OpenAI(base_url=url, api_key="not-needed")
    elif api_mode == "local-hf":
        # Use local Qwen3-VL-7B backend — bypass OpenAI client entirely
        sys.path.insert(0, str(Path(__file__).parent))
        from local_vlm_backend import call_cosmos_local
        return call_cosmos_local(
            bev_image_b64=bev_image_b64,
            scene_description=scene_description,
            question=question,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown api_mode: {api_mode}. Use 'nim', 'local', or 'local-hf'")'''

# ── Patch 3: add sys import at top if missing ──────────────────────────────
old3 = "import os\nimport json\nimport base64"
new3 = "import os\nimport sys\nimport json\nimport base64"

# Apply patches
changed = False
for old, new, label in [(old1, new1, "api-mode choices + default"),
                         (old2, new2, "local-hf branch in call_cosmos"),
                         (old3, new3, "sys import")]:
    if old in text:
        text = text.replace(old, new, 1)
        print(f"✅ Patched: {label}")
        changed = True
    else:
        print(f"⚠️  Already patched or not found: {label}")

if changed:
    script.write_text(text)
    print(f"\nSaved → {script}")
else:
    print("\nNo changes needed.")

# Syntax check
import ast
try:
    ast.parse(text)
    print("Syntax OK ✅")
except SyntaxError as e:
    print(f"Syntax ERROR: {e}")
