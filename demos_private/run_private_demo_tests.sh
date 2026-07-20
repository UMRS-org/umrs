#!/usr/bin/env bash
# Test all private demos and report blockers
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
UMRS_CLIENT="${UMRS_CLIENT:-$(which umrs-client 2>/dev/null || echo "$SCRIPT_DIR/../../../umrs/src/client_umrs.py")}"
PROFILE="${R4_PROFILE:-rccs-cloud}"
PYTHONPATH="$SCRIPT_DIR/../../../umrs/src"

PASS=0
FAIL=0
BLOCKED=0
SKIP=0

log_ok()  { echo -e "  \033[32m✓\033[0m $1"; PASS=$((PASS+1)); }
log_fail(){ echo -e "  \033[31m✗\033[0m $1"; FAIL=$((FAIL+1)); }
log_block(){ echo -e "  \033[33m⚠ BLOCKED\033[0m $1"; BLOCKED=$((BLOCKED+1)); }
log_skip(){ echo -e "  \033[90m⊘ SKIP\033[0m $1"; SKIP=$((SKIP+1)); }

echo "=============================================="
echo "  UMRS Private Demo Tester"
echo "=============================================="
echo ""

# ── Prerequisite checks ───────────────────────

echo "Checking prerequisites..."

UMRS_PY="$SCRIPT_DIR/../../../umrs/src/client_umrs.py"
if [ ! -f "$UMRS_PY" ]; then
    echo "✗ client_umrs.py not found"
    exit 1
fi

# Server ping
echo -n "  Server ping... "
PONG=$(PYTHONPATH="$SCRIPT_DIR/../../../umrs/src" python3 "$UMRS_PY" --profile "$PROFILE" ping 2>&1)
if echo "$PONG" | grep -q "Server ready"; then
    log_ok "Server reachable"
else
    echo ""
    echo "$PONG" | head -3
    log_fail "Server unreachable"
    echo ""
    echo "Cannot continue without server access."
    exit 1
fi

# Bucket list
echo -n "  Bucket list... "
BUCKET_JSON=$(PYTHONPATH="$SCRIPT_DIR/../../../umrs/src" python3 "$UMRS_PY" --profile "$PROFILE" --output json bucket list 2>/dev/null | tail -n1)
BUCKET_COUNT=$(echo "$BUCKET_JSON" | python3 -c "
import json, sys
line = next((l for l in sys.stdin if l.strip().startswith('{')), '')
data = json.loads(line) if line.strip() else {}
print(len(data.get('buckets', [])))
" 2>/dev/null || echo "0")
if [ "$BUCKET_COUNT" -gt 0 ]; then
    log_ok "$BUCKET_COUNT bucket(s) visible"
else
    log_skip "No buckets visible"
fi

# Model list
echo -n "  Model list... "
MODEL_JSON=$(PYTHONPATH="$SCRIPT_DIR/../../../umrs/src" python3 "$UMRS_PY" --profile "$PROFILE" --output json model list --scope system 2>/dev/null | tail -n1)
MODEL_COUNT=$(echo "$MODEL_JSON" | python3 -c "
import json, sys
line = next((l for l in sys.stdin if l.strip().startswith('{')), '')
data = json.loads(line) if line.strip() else {}
print(len(data.get('models', [])))
" 2>/dev/null || echo "0")
if [ "$MODEL_COUNT" -gt 0 ]; then
    log_ok "$MODEL_COUNT system models"
else
    log_skip "No system models"
fi

echo ""

# ── Private Demo Tests ────────────────────────

for demo_dir in 04_hmae_inference 09_defect_analysis 10_hmae_full_pipeline 14_opossum_analysis; do
    DEMO_PATH="$SCRIPT_DIR/../$demo_dir"
    
    if [ ! -d "$DEMO_PATH" ] || [ ! -f "$DEMO_PATH/README.md" ]; then
        log_skip "$demo_dir: directory not found"
        continue
    fi
    
    echo ""
    echo "=============================================="
    echo "  Testing: $demo_dir"
    echo "=============================================="
    echo ""
    
    # Check what bucket/aliases the demo expects
    EXPECTED_BUCKET=$(grep -oE 'user1-[a-z_-]+|demo-[a-z_-]+|demo_user' "$DEMO_PATH/README.md" | grep -E 'bucket|Bucket' | head -1 || echo "")
    
    # List available buckets
    echo "  Expected bucket: ${EXPECTED_BUCKET:-auto-detect}"
    echo "  Available buckets:"
    echo "$BUCKET_JSON" | python3 -c "
import json, sys
line = next((l for l in sys.stdin if l.strip().startswith('{')), '')
data = json.loads(line) if line.strip() else {}
for b in data.get('buckets', []):
    print(f'    {b.get(\"bucket_alias\", \"?\")} ({b.get(\"modality\", \"?\")} - {b.get(\"task\", \"?\")})')
" 2>/dev/null
    
    # Check demo structure
    if [ -d "$DEMO_PATH/workflow" ] && [ -f "$DEMO_PATH/workflow/workflow.json" ]; then
        echo -n "  Workflow bundle... "
        log_ok "Present"
    else
        echo -n "  Workflow bundle... "
        log_skip "Not present (demo uses bucket path only)"
    fi
    
    # Check if required bucket is available
    echo -n "  Bucket availability... "
    AVAIL=$(echo "$BUCKET_JSON" | python3 -c "
import json, sys
line = next((l for l in sys.stdin if l.strip().startswith('{')), '')
data = json.loads(line) if line.strip() else {}
aliases = [b.get('bucket_alias') for b in data.get('buckets', [])]
print(','.join(aliases))
" 2>/dev/null)
    
    HAS_BUCKET=false
    if [ -n "$EXPECTED_BUCKET" ]; then
        echo "$AVAIL" | grep -q "$EXPECTED_BUCKET" && HAS_BUCKET=true
    else
        # Auto-detect: any bucket is fine for basic tests
        if [ -n "$AVAIL" ]; then
            HAS_BUCKET=true
        fi
    fi
    
    if $HAS_BUCKET; then
        log_ok "Bucket available"
    else
        log_block "Required bucket not configured"
        log_info "  This demo needs an operator to provision: $EXPECTED_BUCKET"
        continue
    fi
    
    echo -n "  Sample data... "
    if [ -d "$DEMO_PATH/../../demos/sample_data" ]; then
        log_ok "Sample data exists"
    else
        log_skip "No sample data"
    fi
    
    echo -n "  Upload test (dry)... "
    # Quick upload test with existing test data
    if [ -d "$DEMO_PATH/../../demos/sample_data" ]; then
        UPLOAD_TEST=$(PYTHONPATH="$SCRIPT_DIR/../../../umrs/src" python3 "$UMRS_PY" --profile "$PROFILE" data dir-upload "$DEMO_PATH/../../demos/sample_data/text" "$AVAIL" 2>&1)
        if echo "$UPLOAD_TEST" | grep -q "status=ok\|Processing file\|s3_key_list"; then
            log_ok "Upload works"
        else
            log_fail "Upload failed: $(echo "$UPLOAD_TEST" | tail -3 | head -1)"
        fi
    else
        log_skip "No test data"
    fi
    
    echo ""
done

# ── Summary ───────────────────────────────────

echo ""
echo "=============================================="
echo "  Private Demo Test Summary"
echo "=============================================="
echo ""
echo -e "  \033[32mPassed: $PASS\033[0m"
echo -e "  \033[31mFailed: $FAIL\033[0m"
echo -e "  \033[33mBlocked: $BLOCKED\033[0m"
echo -e "  Skipped: $SKIP"
echo ""

if [ "$BLOCKED" -gt 0 ]; then
    echo "  Blocked demos require operator intervention:"
    echo "  - Create bucket aliases in site.yaml"
    echo "  - Run: umrs-client reload (after adding experiences)"
    echo "  - Ensure model checkpoints are staged on server"
fi

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
