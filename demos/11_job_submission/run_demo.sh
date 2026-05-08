#!/usr/bin/env bash
# =============================================================================
# Demo 11: Job Submission & Endpoint Warm-Up
# =============================================================================
# Shows: endpoint_provision, index_build, cleanup, cancel
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556)
#   - UMRS server running
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="job_sub" source "${SCRIPT_DIR}/../lib/common.sh"

# Extract JSON from umrs-client output (last line starting with '{')
extract_job_id() {
    python3 -c '
import json
import re
import sys

lines = [line.rstrip("\n") for line in sys.stdin if line.strip()]
for line in reversed(lines):
    if line.startswith("{"):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            pass
        else:
            job = payload.get("job") or {}
            job_id = job.get("job_id")
            if job_id:
                print(job_id)
                raise SystemExit(0)
    match = re.search(r"Job submitted: (job-[a-z0-9]+)", line)
    if match:
        print(match.group(1))
        raise SystemExit(0)
' 2>/dev/null || true
}

banner "Demo 11: Job Submission & Endpoint Warm-Up"
echo ""

PROJECT="demo11-$RANDOM"

# Step 0: Check server connectivity
log "Step 0: Ping server..."
preflight
log "Using project id: ${PROJECT}"
echo ""

# Step 1: Pre-warm an endpoint (avoids cold start on first upload)
log "Step 1: Pre-warm endpoint (endpoint_provision)..."
RESULT=$($CLIENT --output json job submit endpoint_provision \
    --spec '{"model": "sam3"}' \
    --project "${PROJECT}")
echo "${RESULT}"
JOB_ID=$(echo "${RESULT}" | extract_job_id)
echo ""

log "Current jobs for ${PROJECT}:"
$CLIENT monitor --once --project "${PROJECT}" || true
echo ""

# Step 2: Rebuild RAG index after bulk inserts
log "Step 2: Rebuild RAG index (index_build)..."
$CLIENT job submit index_build \
    --spec '{"database": "text_embedding_usr1_text", "table": "text_file"}' \
    --project "${PROJECT}"
echo ""

# Step 3: Demonstrate explicit cleanup job submission (no-op for absent endpoint)
log "Step 3: Submit explicit cleanup for an already-absent endpoint id..."
$CLIENT job submit cleanup \
    --spec '{"target": "endpoint", "job_id": "job-stale000000"}' \
    --project "${PROJECT}"
echo ""

# Step 4: Cancel or teardown the pre-warmed endpoint
if [ -n "${JOB_ID}" ]; then
    log "Step 4: Cancel or teardown endpoint ${JOB_ID}..."
    CANCEL_OUT=$($CLIENT job cancel "${JOB_ID}" 2>&1) || true
    echo "${CANCEL_OUT}"
    echo ""
else
    log "Step 4: Skipped (could not extract job_id)"
fi

log "Final jobs for ${PROJECT}:"
$CLIENT monitor --once --project "${PROJECT}" || true

echo ""
banner "Demo 11 Complete"
