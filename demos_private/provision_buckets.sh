#!/usr/bin/env bash
# Provision private demo bucket aliases via experience → reload → verify
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_BASE="$SCRIPT_DIR"
UMRS_PY="/home/mnagaso/workspace/RCCS/ai4s-platform/umrs/src/client_umrs.py"
PYTHONPATH="/home/mnagaso/workspace/RCCS/ai4s-platform/umrs/src"
PROFILE="rccs-cloud"

log_ok()  { echo -e "  \033[32m✓\033[0m $1"; }
log_fail(){ echo -e "  \033[31m✗\033[0m $1"; }
log_step(){ echo -e "\033[36m▶\033[0m $1"; }
log_info(){ echo -e "  \033[33mℹ\033[0m $1"; }

echo "=============================================="
echo "  UMRS Private Demo Bucket Provisioning"
echo "=============================================="
echo ""

# ── Step 1: Verify server connectivity ──────────

log_step "Checking server connectivity..."
PONG=$(cd /home/mnagaso/workspace/RCCS/ai4s-platform/umrs && PYTHONPATH="$PYTHONPATH" python3 "$UMRS_PY" --profile "$PROFILE" ping 2>&1)
if echo "$PONG" | grep -q "Server ready"; then
    log_ok "Server reachable"
else
    log_fail "Server unreachable"
    echo "$PONG" | head -3
    exit 1
fi

# ── Step 2: List current experiences ────────────

log_step "Listing current experiences..."
CURRENT_EXPS=$(cd /home/mnagaso/workspace/RCCS/ai4s-platform/umrs && PYTHONPATH="$PYTHONPATH" python3 "$UMRS_PY" --profile "$PROFILE" experience list 2>&1)
echo "$CURRENT_EXPS" | head -20 | sed 's/^/  /'
echo ""

# ── Step 3: Add each experience ─────────────────

log_step "Adding private demo experiences..."

DEMO_DIRS="04_hmae_inference 09_defect_analysis 10_hmae_full_pipeline 14_opossum_analysis"

for demo_dir in $DEMO_DIRS; do
    EXP_FILE="$DEMO_BASE/$demo_dir/experience.json"
    
    if [ ! -f "$EXP_FILE" ]; then
        log_fail "$demo_dir: experience.json not found"
        continue
    fi
    
    XP_NAME=$(python3 -c "
import json
data = json.load(open('$EXP_FILE'))
print(data.get('xp_name', 'unknown'))
" 2>/dev/null)
    
    echo ""
    echo "  --- Adding '$XP_NAME' ---"
    
    OUTPUT=$(cd /home/mnagaso/workspace/RCCS/ai4s-platform/umrs && PYTHONPATH="$PYTHONPATH" python3 "$UMRS_PY" --profile "$PROFILE" experience add "$EXP_FILE" --update 2>&1)
    
    if echo "$OUTPUT" | grep -q "✓\|Added"; then
        log_ok "$XP_NAME added"
    elif echo "$OUTPUT" | grep -q "already\|skip\|no change"; then
        log_ok "$XP_NAME already present (skipped)"
    else
        log_fail "$XP_NAME: $OUTPUT" | head -3
    fi
done

echo ""

# ── Step 4: Reload server ───────────────────────

log_step "Reloading server to provision buckets..."

RELOAD_OUTPUT=$(cd /home/mnagaso/workspace/RCCS/ai4s-platform/umrs && PYTHONPATH="$PYTHONPATH" python3 "$UMRS_PY" --profile "$PROFILE" reload 2>&1)
echo "$RELOAD_OUTPUT" | sed 's/^/  /'

if echo "$RELOAD_OUTPUT" | grep -q "✓\|accepted\|ok"; then
    log_ok "Reload accepted"
else
    log_fail "Reload failed: $RELOAD_OUTPUT"
    echo "  Retry with: umrs-client reload"
fi

# Wait for reload to complete
echo ""
log_step "Waiting for reload to complete..."
sleep 3

# ── Step 5: Verify bucket aliases ───────────────

log_step "Verifying bucket aliases..."
BUCKET_JSON=$(cd /home/mnagaso/workspace/RCCS/ai4s-platform/umrs && PYTHONPATH="$PYTHONPATH" python3 "$UMRS_PY" --profile "$PROFILE" --output json bucket list 2>&1 | tail -n1)

BUCKETS=$(echo "$BUCKET_JSON" | python3 -c "
import json, sys
line = next((l for l in sys.stdin if l.strip().startswith('{')), '')
if not line.strip():
    print('{}')
    sys.exit()
data = json.loads(line)
buckets = data.get('buckets', [])
for b in sorted(buckets, key=lambda x: x.get('bucket_alias', '')):
    alias = b.get('bucket_alias', '?')
    xp = b.get('experience', '?')
    mod = b.get('modality', '?')
    task = b.get('task', '?')
    objs = b.get('object_count', 0)
    remote = b.get('remote_status', '?')
    print(f'  {alias:20s} | {xp:20s} | {mod:10s} | {task:10s} | {objs:5d} objs | {remote}')
" 2>/dev/null)

echo "$BUCKETS"
TOTAL_BUCKETS=$(echo "$BUCKETS" | grep -c "^[A-Za-z]" 2>/dev/null || echo "0")
echo ""
echo "  Total buckets: $TOTAL_BUCKETS"

# ── Step 6: Verify private demo buckets ─────────

log_step "Checking private demo bucket availability..."

EXPECTED_BUCKETS=("demo-sam3" "demo-defect" "user1-hmae" "demo-hmae" "demo-opossum")
FOUND_BUCKETS=$(echo "$BUCKETS" | awk '{print $1}')

for expected in "${EXPECTED_BUCKETS[@]}"; do
    if echo "$FOUND_BUCKETS" | grep -q "^${expected}$"; then
        log_ok "$expected — ready"
    else
        log_fail "$expected — NOT found (may require additional provisioning)"
    fi
done

echo ""
echo "=============================================="
echo "  Provisioning Complete"
echo "=============================================="
echo ""
echo "  Next steps:"
echo "  1. Test demos with: bash run_private_demo_tests.sh"
echo "  2. Use demo-day scripts for live demos"
echo "  3. Review: DEMO_STATUS_REPORT.md for details"
echo ""
