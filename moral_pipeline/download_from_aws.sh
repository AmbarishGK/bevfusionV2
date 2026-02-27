#!/usr/bin/env bash
# =============================================================================
# MoRAL Pipeline — Download results from AWS instance
# =============================================================================
# Run this from your laptop (bevfusionV2/ root directory) after generation
# completes on AWS.
#
# USAGE:
#   bash download_from_aws.sh <instance-ip>
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

INSTANCE_IP="${1:-}"
if [ -z "$INSTANCE_IP" ]; then
    error "Usage: bash download_from_aws.sh <instance-ip>"
fi

PEM_KEY=""
for candidate in "./localbevfusion.pem" "$HOME/localbevfusion.pem" "$HOME/.ssh/localbevfusion.pem"; do
    if [ -f "$candidate" ]; then PEM_KEY="$candidate"; break; fi
done
[ -z "$PEM_KEY" ] && error "localbevfusion.pem not found"
chmod 400 "$PEM_KEY"

SSH_OPTS="-i $PEM_KEY -o StrictHostKeyChecking=no"
REMOTE="ubuntu@$INSTANCE_IP"
SCP="scp $SSH_OPTS"
LOCAL_OUT="moral_pipeline/02_cosmos_integration"

echo ""
echo "============================================================"
echo "  MoRAL Pipeline — Download from AWS"
echo "  Source: $REMOTE"
echo "  $(date)"
echo "============================================================"
echo ""

# Check generation is actually done
info "Checking generation status on instance..."
ssh $SSH_OPTS $REMOTE "
B=\$(cat ~/02_cosmos_integration/all_qa_conditionB_sharegpt.jsonl 2>/dev/null | wc -l || echo 0)
D=\$(cat ~/02_cosmos_integration/all_qa_conditionD_sharegpt.jsonl 2>/dev/null | wc -l || echo 0)
echo \"Condition B: \$B QA pairs\"
echo \"Condition D: \$D QA pairs\"
if [ \$B -eq 0 ] && [ \$D -eq 0 ]; then
    echo 'WARNING: No output files found — generation may not be complete'
fi
"

# Download concatenated training files
info "Downloading concatenated training files..."
mkdir -p "$LOCAL_OUT"
$SCP "$REMOTE:~/02_cosmos_integration/all_qa_conditionB_sharegpt.jsonl" "$LOCAL_OUT/" 2>/dev/null && \
    success "Downloaded all_qa_conditionB_sharegpt.jsonl" || \
    info "all_qa_conditionB not found yet"

$SCP "$REMOTE:~/02_cosmos_integration/all_qa_conditionD_sharegpt.jsonl" "$LOCAL_OUT/" 2>/dev/null && \
    success "Downloaded all_qa_conditionD_sharegpt.jsonl" || \
    info "all_qa_conditionD not found yet"

# Download per-scene QA JSON files
info "Downloading per-scene QA pairs..."
mkdir -p "$LOCAL_OUT/cosmos_qa"
if command -v rsync &>/dev/null; then
    rsync -az \
        -e "ssh $SSH_OPTS" \
        "$REMOTE:~/02_cosmos_integration/cosmos_qa/" \
        "$LOCAL_OUT/cosmos_qa/"
else
    $SCP -r "$REMOTE:~/02_cosmos_integration/cosmos_qa/" "$LOCAL_OUT/"
fi
success "Per-scene QA pairs downloaded"

# Download logs
info "Downloading logs..."
mkdir -p "$LOCAL_OUT/logs"
$SCP -r "$REMOTE:~/logs/" "$LOCAL_OUT/logs/" 2>/dev/null || true
success "Logs downloaded to $LOCAL_OUT/logs/"

# Summary
echo ""
echo "============================================================"
echo "  Download complete — $(date)"
echo "============================================================"
B_COUNT=$(cat "$LOCAL_OUT/all_qa_conditionB_sharegpt.jsonl" 2>/dev/null | wc -l || echo 0)
D_COUNT=$(cat "$LOCAL_OUT/all_qa_conditionD_sharegpt.jsonl" 2>/dev/null | wc -l || echo 0)
echo ""
echo "  QA pairs: $B_COUNT condition B + $D_COUNT condition D = $((B_COUNT + D_COUNT)) total"
echo ""
echo "  Files at: $LOCAL_OUT/"
echo ""
echo "  Inspect a sample:"
echo "    cat $LOCAL_OUT/cosmos_qa/scene-0061/qa_pairs_conditionB.json | python3 -m json.tool | head -80"
echo ""
