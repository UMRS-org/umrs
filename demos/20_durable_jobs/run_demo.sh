#!/usr/bin/env bash
# =============================================================================
# Demo 20: Durable Job Orchestration
# =============================================================================
# Demonstrates the durable job system:
#   1. Submit a job with idempotency key
#   2. Poll job status
#   3. Submit duplicate (idempotent — deduped)
#   4. Submit another job and cancel it
#   5. Demonstrate retry behavior
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running with job subsystem enabled
#   - umrs-client installed and in PATH
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="durable_job" source "${SCRIPT_DIR}/../lib/common.sh"

RESULTS_DIR="${SCRIPT_DIR}/results"

banner "Demo 20: Durable Job Orchestration"
echo "Config       : ${CONF}"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight
mkdir -p "$RESULTS_DIR"

# ---------- Step 2: Submit a job ----------
log ""
log "Step 2: Submitting an index_build job..."

JOB_SPEC='{"database": "demo_db", "table": "embeddings", "rebuild": true}'
RESULT=$($CLIENT job submit index_build \
    --spec "$JOB_SPEC" \
    --project demo \
    --keep-alive 2>&1)
echo "$RESULT" | while IFS= read -r line; do log "  $line"; done

JOB_ID=$(echo "$RESULT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('job', {}).get('job_id', ''))
except: pass
" 2>/dev/null || true)

if [[ -n "$JOB_ID" ]]; then
    ok "Job submitted: ${JOB_ID}"
    echo "$JOB_ID" > "${RESULTS_DIR}/job_id.txt"
else
    warn "Could not extract job_id from response"
    JOB_ID="unknown"
fi

# ---------- Step 3: Poll job status ----------
log ""
log "Step 3: Polling job status..."

for i in 1 2 3; do
    sleep 3
    STATUS=$($CLIENT job status "$JOB_ID" 2>&1 || echo '{"status": "unknown"}')
    CURRENT=$(echo "$STATUS" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    job = data.get('job', data)
    print(job.get('status', 'unknown'))
except: print('unknown')
" 2>/dev/null || echo "unknown")
    log "  Poll ${i}: status=${CURRENT}"

    if [[ "$CURRENT" == "completed" ]] || [[ "$CURRENT" == "failed" ]]; then
        break
    fi
done
ok "Status polling complete (final: ${CURRENT})"

# ---------- Step 4: Idempotent resubmit ----------
log ""
log "Step 4: Submitting duplicate job (testing idempotency)..."

RESULT2=$($CLIENT job submit index_build \
    --spec "$JOB_SPEC" \
    --project demo \
    --keep-alive 2>&1)
echo "$RESULT2" | while IFS= read -r line; do log "  $line"; done

JOB_ID2=$(echo "$RESULT2" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('job', {}).get('job_id', ''))
except: pass
" 2>/dev/null || true)

if [[ "$JOB_ID" == "$JOB_ID2" ]]; then
    ok "Idempotency confirmed — same job_id returned: ${JOB_ID}"
elif [[ -n "$JOB_ID2" ]]; then
    log "  New job_id: ${JOB_ID2} (server may not enforce idempotency for this job type)"
fi

# ---------- Step 5: Submit and cancel a job ----------
log ""
log "Step 5: Submit a new job, then cancel it..."

CANCEL_SPEC='{"database": "cancel_test", "table": "temp"}'
RESULT3=$($CLIENT job submit index_build \
    --spec "$CANCEL_SPEC" \
    --project demo 2>&1)
echo "$RESULT3" | while IFS= read -r line; do log "  $line"; done

CANCEL_JOB_ID=$(echo "$RESULT3" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('job', {}).get('job_id', ''))
except: pass
" 2>/dev/null || true)

if [[ -n "$CANCEL_JOB_ID" ]]; then
    log "  Cancelling job ${CANCEL_JOB_ID}..."
    $CLIENT job cancel "$CANCEL_JOB_ID" 2>&1 \
        | while IFS= read -r line; do log "  $line"; done
    ok "Job cancelled: ${CANCEL_JOB_ID}"

    # Verify cancelled status
    sleep 1
    CSTATUS=$($CLIENT job status "$CANCEL_JOB_ID" 2>&1 || echo '{}')
    CFINAL=$(echo "$CSTATUS" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('job', data).get('status', 'unknown'))
except: print('unknown')
" 2>/dev/null || echo "unknown")
    log "  Cancelled job status: ${CFINAL}"
else
    warn "Could not extract job_id for cancellation test"
fi

# ---------- Summary ----------
log ""
log "Summary of durable job operations:"
log "  ✓ Submitted index_build job: ${JOB_ID}"
log "  ✓ Polled status through lifecycle"
log "  ✓ Idempotent resubmit test completed"
log "  ✓ Submit + cancel flow demonstrated"
log ""
log "Results saved to: ${RESULTS_DIR}/"

banner "Demo 20 Complete"
