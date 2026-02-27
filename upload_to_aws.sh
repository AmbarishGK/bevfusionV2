#!/usr/bin/env bash
# =============================================================================
# MoRAL Pipeline — Upload files to AWS instance
# =============================================================================
# Run this from your laptop (bevfusionV2/ root directory).
#
# USAGE:
#   bash upload_to_aws.sh <instance-ip>
#   bash upload_to_aws.sh 54.123.45.67
#
# Prerequisites:
#   - localbevfusion.pem in current directory or ~/
#   - AWS instance already running (g5.2xlarge)
#   - Outputs already generated locally (01_gt_annotations, 02_gt_with_radar)
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Args ──────────────────────────────────────────────────────────────────────
INSTANCE_IP="${1:-}"
if [ -z "$INSTANCE_IP" ]; then
    error "Usage: bash upload_to_aws.sh <instance-ip>"
fi

# Find PEM key
PEM_KEY=""
for candidate in "./localbevfusion.pem" "$HOME/localbevfusion.pem" "$HOME/.ssh/localbevfusion.pem"; do
    if [ -f "$candidate" ]; then
        PEM_KEY="$candidate"
        break
    fi
done
if [ -z "$PEM_KEY" ]; then
    error "localbevfusion.pem not found. Put it in current dir or ~/"
fi
chmod 400 "$PEM_KEY"

SSH_OPTS="-i $PEM_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"
REMOTE="ubuntu@$INSTANCE_IP"
SCP="scp $SSH_OPTS"
SSH="ssh $SSH_OPTS $REMOTE"

echo ""
echo "============================================================"
echo "  MoRAL Pipeline — Upload to AWS"
echo "  Target: $REMOTE"
echo "  PEM:    $PEM_KEY"
echo "  $(date)"
echo "============================================================"
echo ""

# ── Test connection ───────────────────────────────────────────────────────────
info "Testing SSH connection..."
$SSH "echo connected" > /dev/null || error "Cannot connect to $REMOTE"
success "SSH connection OK"

# ── Create remote directory structure ────────────────────────────────────────
info "Creating remote directory structure..."
$SSH "mkdir -p ~/outputs/01_gt_annotations ~/outputs/02_gt_with_radar ~/02_cosmos_integration/cosmos_qa ~/logs"
success "Remote directories created"

# ── Upload generate_cosmos_qa.py ─────────────────────────────────────────────
info "Uploading generate_cosmos_qa.py..."

# Find the script — check a few likely locations
SCRIPT_PATH=""
for candidate in \
    "./moral_pipeline/02_cosmos_integration/generate_cosmos_qa.py" \
    "./generate_cosmos_qa.py"; do
    if [ -f "$candidate" ]; then
        SCRIPT_PATH="$candidate"
        break
    fi
done

if [ -z "$SCRIPT_PATH" ]; then
    error "generate_cosmos_qa.py not found. Expected at moral_pipeline/02_cosmos_integration/generate_cosmos_qa.py"
fi

$SCP "$SCRIPT_PATH" "$REMOTE:~/generate_cosmos_qa.py"
success "Uploaded generate_cosmos_qa.py from $SCRIPT_PATH"

# ── Upload setup_and_run.sh ───────────────────────────────────────────────────
info "Uploading setup_and_run.sh..."
for candidate in "./setup_and_run.sh" "$HOME/setup_and_run.sh"; do
    if [ -f "$candidate" ]; then
        $SCP "$candidate" "$REMOTE:~/setup_and_run.sh"
        $SSH "chmod +x ~/setup_and_run.sh"
        success "Uploaded setup_and_run.sh"
        break
    fi
done

# ── Upload scene data ─────────────────────────────────────────────────────────
CLEAN_SCENES=("scene-0061" "scene-0553" "scene-0655" "scene-0757" "scene-0796"
              "scene-0916" "scene-1077" "scene-1094" "scene-1100")

upload_condition() {
    local local_dir="$1"
    local remote_dir="$2"
    local cond_name="$3"

    info "Uploading $cond_name scenes..."

    if [ ! -d "$local_dir" ]; then
        error "Local directory not found: $local_dir"
    fi

    TOTAL=0
    for SCENE in "${CLEAN_SCENES[@]}"; do
        LOCAL_SCENE="$local_dir/$SCENE"
        if [ ! -d "$LOCAL_SCENE" ]; then
            echo "  WARNING: $LOCAL_SCENE not found, skipping"
            continue
        fi

        # Count files being uploaded
        N_FILES=$(find "$LOCAL_SCENE" -name "*.jpg" -o -name "*.png" \
                       -o -name "*.json" -o -name "*.txt" | wc -l)

        echo -n "  $SCENE ($N_FILES files)... "

        # rsync is faster than scp -r for many small files
        # Falls back to scp if rsync not available
        if command -v rsync &>/dev/null; then
            rsync -az \
                --include="bev_map.png" \
                --include="CAM_FRONT.jpg" \
                --include="CAM_FRONT_LEFT.jpg" \
                --include="CAM_FRONT_RIGHT.jpg" \
                --include="detections.json" \
                --include="scene_description.txt" \
                --include="metadata.json" \
                --exclude="*" \
                -e "ssh $SSH_OPTS" \
                "$LOCAL_SCENE/" \
                "$REMOTE:$remote_dir/$SCENE/"
        else
            # Create remote dir first
            $SSH "mkdir -p $remote_dir/$SCENE"
            # Upload only needed files (not CAM_BACK* — saves ~40% transfer)
            for FILE in bev_map.png CAM_FRONT.jpg CAM_FRONT_LEFT.jpg CAM_FRONT_RIGHT.jpg \
                        detections.json scene_description.txt metadata.json; do
                if [ -f "$LOCAL_SCENE/$FILE" ]; then
                    $SCP "$LOCAL_SCENE/$FILE" "$REMOTE:$remote_dir/$SCENE/$FILE" 2>/dev/null || true
                fi
            done
        fi

        echo "done"
        TOTAL=$((TOTAL + 1))
    done
    success "Uploaded $TOTAL/$((${#CLEAN_SCENES[@]})) scenes for $cond_name"
}

upload_condition \
    "moral_pipeline/outputs/01_gt_annotations" \
    "~/outputs/01_gt_annotations" \
    "Condition B"

upload_condition \
    "moral_pipeline/outputs/02_gt_with_radar" \
    "~/outputs/02_gt_with_radar" \
    "Condition D"

# ── Verify upload ─────────────────────────────────────────────────────────────
info "Verifying upload..."
$SSH "
MISSING=0
for SCENE in scene-0061 scene-0553 scene-0655 scene-0757 scene-0796 scene-0916 scene-1077 scene-1094 scene-1100; do
    for COND in 01_gt_annotations 02_gt_with_radar; do
        DIR=\$HOME/outputs/\$COND/\$SCENE
        if [ ! -f \$DIR/bev_map.png ]; then echo \"MISSING: \$DIR/bev_map.png\"; MISSING=\$((MISSING+1)); fi
        if [ ! -f \$DIR/detections.json ]; then echo \"MISSING: \$DIR/detections.json\"; MISSING=\$((MISSING+1)); fi
    done
done
echo \"Missing files: \$MISSING\"
"
success "Upload verification complete"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Upload complete — $(date)"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. SSH into the instance:"
echo "     ssh -i $PEM_KEY $REMOTE"
echo ""
echo "  2. Run the setup + generation script:"
echo "     bash ~/setup_and_run.sh"
echo ""
echo "  Or run it fully remotely (detached, survives disconnect):"
echo "     ssh -i $PEM_KEY $REMOTE 'nohup bash ~/setup_and_run.sh > ~/logs/run.log 2>&1 &'"
echo "     # Then tail the log:"
echo "     ssh -i $PEM_KEY $REMOTE 'tail -f ~/logs/run.log'"
echo ""
