#!/bin/bash
set -e

# ══════════════════════════════════════════════════════════════
# MoRAL Pipeline Auto-Setup — A100 80GB SXM4
# Runs automatically on VM boot via Brev startup script
# Replace HF_TOKEN below with your actual token before deploying
# Get token: https://huggingface.co/settings/tokens (READ access)
# ══════════════════════════════════════════════════════════════

HF_TOKEN="YOUR_HF_TOKEN"

# Log everything to file for debugging
exec > /home/ubuntu/setup.log 2>&1
echo "=== Setup started: $(date) ==="

# ── System packages ────────────────────────────────────────────
apt-get update -qq
apt-get install -y git wget curl screen htop -qq

# ── Python deps ────────────────────────────────────────────────
pip install --upgrade pip -q
pip install vllm openai huggingface_hub YOUR_HF_TOKEN -q

# ── HuggingFace auth ───────────────────────────────────────────
export HF_TOKEN=$HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

# ── Download Cosmos-Reason2-8B (~16GB) ────────────────────────
echo "=== Downloading Cosmos-Reason2-8B ==="
mkdir -p /home/ubuntu/models
huggingface-cli download nvidia/Cosmos-Reason2-8B \
    --local-dir /home/ubuntu/models/Cosmos-Reason2-8B
echo "✅ Cosmos downloaded"

# ── Download MoRAL dataset (~1GB) ─────────────────────────────
echo "=== Downloading MoRAL dataset ==="
mkdir -p /home/ubuntu/moral_pipeline
huggingface-cli download AmbarishGK/MoRAL-nuScenes-BEV-850 \
    --repo-type dataset \
    --local-dir /home/ubuntu/moral_pipeline/YOUR_HF_TOKEN
echo "✅ Dataset downloaded"

# ── Verify dataset ─────────────────────────────────────────────
for cond in condition_A_camera_only condition_B_lidar_bev condition_D_lidar_bev_radar; do
    count=$(ls /home/ubuntu/moral_pipeline/YOUR_HF_TOKEN/$cond/ 2>/dev/null | grep -c "scene" || echo 0)
    echo "  $cond: $count scenes"
done

# ── Symlinks (pipeline_root/outputs/<condition>) ───────────────
mkdir -p /home/ubuntu/moral_pipeline/outputs
ln -sf /home/ubuntu/moral_pipeline/YOUR_HF_TOKEN/condition_A_camera_only \
       /home/ubuntu/moral_pipeline/outputs/condition_A_camera_only
ln -sf /home/ubuntu/moral_pipeline/YOUR_HF_TOKEN/condition_B_lidar_bev \
       /home/ubuntu/moral_pipeline/outputs/condition_B_lidar_bev
ln -sf /home/ubuntu/moral_pipeline/YOUR_HF_TOKEN/condition_D_lidar_bev_radar \
       /home/ubuntu/moral_pipeline/outputs/condition_D_lidar_bev_radar

# ── Clone repo and copy QA script ─────────────────────────────
echo "=== Cloning repo ==="
git clone https://github.com/AmbarishGK/bevfusionV2.git /home/ubuntu/bevfusionV2
cp /home/ubuntu/bevfusionV2/moral_pipeline/02_cosmos_integration/generate_cosmos_qa.py \
   /home/ubuntu/moral_pipeline/generate_cosmos_qa.py

# ── Build scene list (first 100 scenes) ───────────────────────
python3 -c "
import os
scenes = sorted([d for d in os.listdir('/home/ubuntu/moral_pipeline/outputs/condition_B_lidar_bev') if d.startswith('scene')])
print(f'Total scenes: {len(scenes)}')
open('/home/ubuntu/moral_pipeline/scenes_100.txt', 'w').write('\n'.join(scenes[:100]))
print(f'Written: {scenes[0]} to {scenes[99]}')
"

# ── Create output dirs ─────────────────────────────────────────
mkdir -p /home/ubuntu/moral_pipeline/logs
mkdir -p /home/ubuntu/moral_pipeline/02_cosmos_integration/cosmos_qa

# ── Write run_vllm.sh ──────────────────────────────────────────
cat > /home/ubuntu/moral_pipeline/run_vllm.sh << 'EOF'
#!/bin/bash
# NO --reasoning-parser flag -- strips <think> from content, breaks parser
python3 -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/models/Cosmos-Reason2-8B \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    > /home/ubuntu/moral_pipeline/logs/vllm.log 2>&1 &
echo "vLLM PID: $!"
echo "Waiting 3 minutes for model to load..."
sleep 180
curl -s http://localhost:8000/health && echo "✅ vLLM ready" || echo "❌ Check logs/vllm.log"
EOF

# ── Write smoke_test.sh ────────────────────────────────────────
cat > /home/ubuntu/moral_pipeline/smoke_test.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/moral_pipeline
FIRST=$(head -1 scenes_100.txt)
echo "Smoke test: $FIRST condition B"
python3 generate_cosmos_qa.py \
    --scene $FIRST \
    --condition B \
    --api-mode local \
    --base-url http://localhost:8000/v1 \
    --pipeline-root /home/ubuntu/moral_pipeline \
    --no-skip 2>&1 | tee logs/smoke_test.log
python3 -c "
import json, glob
files = glob.glob('02_cosmos_integration/cosmos_qa/*/qa_pairs_conditionB.json')
if not files: print('❌ No output found'); exit(1)
pairs = json.load(open(files[0]))
ok = sum(1 for p in pairs if p.get('quality_ok'))
print(f'Quality: {ok}/{len(pairs)} passed')
print('Sample reasoning:', pairs[0]['reasoning'][:400])
"
EOF

# ── Write run_qa_B.sh ──────────────────────────────────────────
cat > /home/ubuntu/moral_pipeline/run_qa_B.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/moral_pipeline
screen -dmS qa_B bash -c "
python3 generate_cosmos_qa.py \
    --condition B \
    --scene-list scenes_100.txt \
    --api-mode local \
    --base-url http://localhost:8000/v1 \
    --pipeline-root /home/ubuntu/moral_pipeline \
    2>&1 | tee logs/qa_B.log
echo CONDITION_B_DONE
"
echo "✅ Condition B started in screen qa_B"
echo "Monitor: tail -f logs/qa_B.log | grep 'Progress\|EXCEPTION\|done'"
echo "Attach:  screen -r qa_B"
EOF

# ── Write run_qa_D.sh ──────────────────────────────────────────
cat > /home/ubuntu/moral_pipeline/run_qa_D.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/moral_pipeline
screen -dmS qa_D bash -c "
python3 generate_cosmos_qa.py \
    --condition D \
    --scene-list scenes_100.txt \
    --api-mode local \
    --base-url http://localhost:8000/v1 \
    --pipeline-root /home/ubuntu/moral_pipeline \
    2>&1 | tee logs/qa_D.log
echo CONDITION_D_DONE
"
echo "✅ Condition D started in screen qa_D"
echo "Monitor: tail -f logs/qa_D.log | grep 'Progress\|EXCEPTION\|done'"
echo "Attach:  screen -r qa_D"
EOF

chmod +x /home/ubuntu/moral_pipeline/run_vllm.sh
chmod +x /home/ubuntu/moral_pipeline/smoke_test.sh
chmod +x /home/ubuntu/moral_pipeline/run_qa_B.sh
chmod +x /home/ubuntu/moral_pipeline/run_qa_D.sh

# ── GPU check ──────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== SETUP COMPLETE: $(date) ==="
echo "SSH in and run:"
echo "  cd /home/ubuntu/moral_pipeline"
echo "  bash run_vllm.sh        # start server (3min wait built in)"
echo "  bash smoke_test.sh      # verify 1 scene"
echo "  bash run_qa_B.sh        # condition B"
echo "  bash run_qa_D.sh        # condition D (after B finishes)"
echo ""
echo "Download on LOCAL machine:"
echo "  rsync -avz ubuntu@IP:/home/ubuntu/moral_pipeline/02_cosmos_integration/cosmos_qa/ ./cosmos_qa/"
