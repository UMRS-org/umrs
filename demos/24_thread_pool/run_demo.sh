#!/usr/bin/env bash
# =============================================================================
# Demo 24: Thread Pool & Backpressure
# =============================================================================
# Shows: bounded thread pool metrics, live utilisation, backpressure status
#
# This demo queries the UMRS server's thread pool via the status endpoint
# and displays pool utilisation metrics:
#   - max_workers / max_pending (pool capacity)
#   - active_count / pending_count (current load)
#   - is_saturated (backpressure indicator)
#
# Optionally submits a lightweight workload to observe metrics change.
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running (thread pool is always active)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="pool" source "${SCRIPT_DIR}/../lib/common.sh"

banner "Demo 24: Thread Pool & Backpressure"
echo ""

TEXT_FILE_ALIAS=$(python3 - <<'PY' "$CONF"
import json
import sys
from pathlib import Path

conf_path = Path(sys.argv[1]).expanduser()
data = json.loads(conf_path.read_text())
for xp in data.get("experiences", []):
    for modality in xp.get("modalities", []):
        alias = ((modality.get("s3") or {}).get("bucket_alias"))
        data_cfg = (modality.get("s3") or {}).get("data") or {}
        mime_type = data_cfg.get("mime_type", "")
        modality_name = str(modality.get("modality_name") or modality.get("modality") or "").lower()
        if alias and mime_type == "text/plain" and ("file" in modality_name or data_cfg.get("type") == "file"):
            print(alias)
            raise SystemExit(0)
print("")
PY
)

# Step 0: Pre-flight
log "Step 0: Ping server..."
preflight
echo ""

# Step 1: Query pool metrics from status endpoint
log "Step 1: Query pool metrics..."
STATUS_JSON="$($CLIENT --output json status all 2>/dev/null || echo '{}')"
STATUS_JSON="$(printf '%s\n' "$STATUS_JSON" | extract_json_payload)"
echo "$STATUS_JSON" | python3 -c "
import sys, json

data = json.load(sys.stdin)
pool = data.get('pool')

if not pool:
    print('  No pool metrics in status response.')
    print('  (Thread pool metrics are only available if the server has the bounded pool feature.)')
    sys.exit(0)

print('  Thread Pool Status')
print('  ──────────────────')
print(f'  Max workers:    {pool[\"max_workers\"]}')
print(f'  Max pending:    {pool[\"max_pending\"]}')
print(f'  Active threads: {pool[\"active_count\"]}')
print(f'  Pending tasks:  {pool[\"pending_count\"]}')
print(f'  Total queued:   {pool[\"total_queued\"]}')
print(f'  Is saturated:   {pool[\"is_saturated\"]}')
"
ok "Pool metrics retrieved"
echo ""

# Step 2: Verify pool capacity fields
log "Step 2: Validate pool metric fields..."
echo "$STATUS_JSON" | python3 -c "
import sys, json

data = json.load(sys.stdin)
pool = data.get('pool')

if not pool:
    print('  SKIP: no pool metrics')
    sys.exit(0)

required = ['max_workers', 'max_pending', 'active_count', 'pending_count', 'total_queued', 'is_saturated']
missing = [f for f in required if f not in pool]
if missing:
    print(f'  FAIL: missing fields: {missing}')
    sys.exit(1)

assert isinstance(pool['max_workers'], int), 'max_workers must be int'
assert isinstance(pool['max_pending'], int), 'max_pending must be int'
assert isinstance(pool['active_count'], int), 'active_count must be int'
assert isinstance(pool['pending_count'], int), 'pending_count must be int'
assert isinstance(pool['total_queued'], int), 'total_queued must be int'
assert isinstance(pool['is_saturated'], bool), 'is_saturated must be bool'
assert pool['total_queued'] == pool['active_count'] + pool['pending_count'], 'total_queued mismatch'
assert pool['max_workers'] > 0, 'max_workers must be positive'
assert pool['max_pending'] > 0, 'max_pending must be positive'

print('  ✓ All pool metric fields present and valid')
"
echo ""

# Step 3: Check saturation status
log "Step 3: Check backpressure indicator..."
echo "$STATUS_JSON" | python3 -c "
import sys, json

data = json.load(sys.stdin)
pool = data.get('pool')

if not pool:
    print('  SKIP: no pool metrics')
    sys.exit(0)

if pool['is_saturated']:
    print('  ⚠ Pool is SATURATED — new tasks will be rejected until workers free up')
    print(f'    pending={pool[\"pending_count\"]} >= max_pending={pool[\"max_pending\"]}')
else:
    headroom = pool['max_pending'] - pool['pending_count']
    print(f'  ✓ Pool has capacity — {headroom} pending slots available')
    print(f'    active={pool[\"active_count\"]}/{pool[\"max_workers\"]}, pending={pool[\"pending_count\"]}/{pool[\"max_pending\"]}')
"
echo ""

# Step 4: Optional — submit a lightweight task and watch metrics change
if [[ "${DEMO_SUBMIT_TASK:-0}" = "1" ]]; then
    log "Step 4: Submit a lightweight task and re-query..."
    echo ""

    if [[ -z "$TEXT_FILE_ALIAS" ]]; then
        warn "No text-file bucket alias found in ${CONF}; skipping upload test"
        banner "Demo 24 Complete"
        exit 0
    fi

    # Upload a small test file to trigger processing
    TEST_FILE="$(mktemp)"
    echo "Pool demo test file $(date)" > "$TEST_FILE"
    trap "rm -f '$TEST_FILE'" EXIT

    $CLIENT upload_file "$TEST_FILE" "$TEXT_FILE_ALIAS" || warn "Upload failed (bucket may not exist)"
    sleep 1

    # Re-query pool metrics
    POST_STATUS="$($CLIENT --output json status all 2>/dev/null || echo '{}')"
    POST_STATUS="$(printf '%s\n' "$POST_STATUS" | extract_json_payload)"
    echo "$POST_STATUS" | python3 -c "
import sys, json

data = json.load(sys.stdin)
pool = data.get('pool')

if not pool:
    print('  No pool metrics after upload')
    sys.exit(0)

print('  Pool after task submission:')
print(f'    active={pool[\"active_count\"]}, pending={pool[\"pending_count\"]}, saturated={pool[\"is_saturated\"]}')
"
    rm -f "$TEST_FILE"
    trap - EXIT
    echo ""
else
    log "Step 4: Skipped (set DEMO_SUBMIT_TASK=1 to submit a test task)"
fi
echo ""

banner "Demo 24 Complete"
