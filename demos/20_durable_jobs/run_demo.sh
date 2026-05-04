#!/usr/bin/env bash
# =============================================================================
# Demo 20: Durable Job Orchestration
# =============================================================================
# Demonstrates the durable job system:
#   1. Submit an index_build job
#   2. Poll job status
#   3. Submit duplicate (idempotent — deduped)
#   4. Submit an endpoint_provision job and cancel it
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

PROJECT="demo20-$(date +%s)"
CANCEL_PROJECT="${PROJECT}-cancel"

extract_json() {
    extract_json_payload
}

extract_job_id() {
    python3 -c '
import json
import sys

payload = json.loads(sys.argv[1])
job = payload.get("job") or {}
print(job.get("job_id", ""))
' "$1"
}

extract_job_status() {
    python3 -c '
import json
import sys

payload = json.loads(sys.argv[1])
job = payload.get("job") or {}
print(job.get("status", "unknown"))
' "$1"
}

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight
mkdir -p "$RESULTS_DIR"
log "Using project id: ${PROJECT}"

# ---------- Step 2: Submit a job ----------
log ""
log "Step 2: Submitting an index_build job..."

JOB_SPEC='{"database": "text_embedding_usr1_text", "table": "text_file"}'
RESULT=$($CLIENT --output json job submit index_build \
    --spec "$JOB_SPEC" \
    --project "$PROJECT" 2>&1 || true)
echo "$RESULT" | while IFS= read -r line; do log "  $line"; done

JOB_JSON=$(echo "$RESULT" | extract_json 2>/dev/null || true)
JOB_ID=$(extract_job_id "$JOB_JSON" 2>/dev/null || true)

if [[ -n "$JOB_ID" ]]; then
    ok "Job submitted: ${JOB_ID}"
    echo "$JOB_ID" > "${RESULTS_DIR}/job_id.txt"
else
    warn "Could not extract job_id from index_build response; durable job API may be temporarily unavailable"
    warn "Skipping remaining durable job flow"
    banner "Demo 20 Complete"
    exit 0
fi

# ---------- Step 3: Poll job status ----------
log ""
log "Step 3: Polling job status..."

for i in 1 2 3; do
    sleep 3
    STATUS=$($CLIENT --output json job status "$JOB_ID" 2>&1 || echo '{"status": "unknown"}')
    STATUS_JSON=$(echo "$STATUS" | extract_json 2>/dev/null || true)
    CURRENT=$(extract_job_status "$STATUS_JSON" 2>/dev/null || echo "unknown")
    log "  Poll ${i}: status=${CURRENT}"

    if [[ "$CURRENT" == "completed" ]] || [[ "$CURRENT" == "failed" ]]; then
        break
    fi
done
ok "Status polling complete (final: ${CURRENT})"

# ---------- Step 4: Idempotent resubmit ----------
log ""
log "Step 4: Submitting duplicate job (testing idempotency)..."

RESULT2=$($CLIENT --output json job submit index_build \
    --spec "$JOB_SPEC" \
    --project "$PROJECT" 2>&1 || true)
echo "$RESULT2" | while IFS= read -r line; do log "  $line"; done

JOB_JSON2=$(echo "$RESULT2" | extract_json 2>/dev/null || true)
JOB_ID2=$(extract_job_id "$JOB_JSON2" 2>/dev/null || true)

if [[ "$JOB_ID" == "$JOB_ID2" ]]; then
    ok "Idempotency confirmed — same job_id returned: ${JOB_ID}"
elif [[ -n "$JOB_ID2" ]]; then
    log "  New job_id: ${JOB_ID2} (server may not enforce idempotency for this job type)"
fi

# ---------- Step 5: Submit and cancel a job ----------
log ""
log "Step 5: Submit a new endpoint_provision job, then cancel it..."

CANCEL_SPEC='{"model": "sam3"}'
RESULT3=$($CLIENT --output json job submit endpoint_provision \
    --spec "$CANCEL_SPEC" \
    --project "$CANCEL_PROJECT" 2>&1 || true)
echo "$RESULT3" | while IFS= read -r line; do log "  $line"; done

JOB_JSON3=$(echo "$RESULT3" | extract_json 2>/dev/null || true)
CANCEL_JOB_ID=$(extract_job_id "$JOB_JSON3" 2>/dev/null || true)

if [[ -n "$CANCEL_JOB_ID" ]]; then
    log "  Cancelling job ${CANCEL_JOB_ID}..."
    $CLIENT job cancel "$CANCEL_JOB_ID" 2>&1 \
        | while IFS= read -r line; do log "  $line"; done
    ok "Job cancelled: ${CANCEL_JOB_ID}"

    # Verify cancelled status
    sleep 1
    CSTATUS=$($CLIENT --output json job status "$CANCEL_JOB_ID" 2>&1 || echo '{}')
    CSTATUS_JSON=$(echo "$CSTATUS" | extract_json 2>/dev/null || true)
    CFINAL=$(extract_job_status "$CSTATUS_JSON" 2>/dev/null || echo "unknown")
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
log "  ✓ endpoint_provision submit + cancel flow demonstrated"
log ""
log "Results saved to: ${RESULTS_DIR}/"

banner "Demo 20 Complete"
