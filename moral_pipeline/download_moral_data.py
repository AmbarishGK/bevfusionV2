import os
from huggingface_hub import snapshot_download

# Configuration
REPO_ID = "AmbarishGK/MoRAL-nuscenes"
LOCAL_DIR = "02_cosmos_integration/hf_data"

print(f"Starting resilient download from {REPO_ID}...")

try:
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        max_workers=1,           # Vital to avoid 429 errors
        resume_download=True,    # Picks up where the CLI failed
        local_dir_use_symlinks=False
    )
    print("\n✅ Download Complete!")
    
    # Auto-fix the directory structure for evaluate_zeroshot.py
    print("Fixing directory structure (images/ -> outputs/)...")
    src_images = os.path.join(LOCAL_DIR, "images")
    dst_outputs = os.path.join(LOCAL_DIR, "outputs")
    
    if os.path.exists(src_images):
        if not os.path.exists(dst_outputs):
            os.makedirs(dst_outputs)
        # Move folders inside images/ to outputs/
        for folder in os.listdir(src_images):
            os.rename(os.path.join(src_images, folder), os.path.join(dst_outputs, folder))
        print("✅ Structure aligned with evaluate_zeroshot.py")

except Exception as e:
    print(f"\n❌ Error: {e}")
