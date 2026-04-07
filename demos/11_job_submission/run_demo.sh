#!/usr/bin/env bash
# =============================================================================
# Demo 11: Job Submission & Service Lifecycle
# =============================================================================
# Shows: endpoint_provision, index_build, cleanup, --keep-alive, cancel
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556)
#   - UMRS server running
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="job_sub" source "${SCRIPT_DIR}/../lib/common.sh"

# Extract JSON from umrs-client output (last line starting with '{')
extract_json() {
    grep '^{' | tail -1
}

banner "Demo 11: Job Submission & Service Lifecycle"
echo ""

# Step 0: Check server connectivity
log "Step 0: Ping server..."
preflight
echo ""

# Step 1: Pre-warm an endpoint (avoids cold start on first upload)
log "Step 1: Pre-warm endpoint (endpoint_provision --keep-alive)..."
RESULT=$($CLIENT job submit endpoint_provision \
    --spec '{"model": "sam3"}' \
    --project demo --keep-alive)
echo "${RESULT}"
JOB_ID=$(echo "${RESULT}" | extract_json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job',{}).get('job_id',''))" 2>/dev/null || echo "")
echo ""

# Step 2: Rebuild RAG index after bulk inserts
log "Step 2: Rebuild RAG index (index_build)..."
$CLIENT job submit index_build \
    --spec '{"database": "xp_demo", "table": "text_embeddings"}' \
    --project demo
echo ""

# Step 3: Request cleanup of stale resources
log "Step 3: Cleanup stale resources..."
$CLIENT job submit cleanup \
    --spec '{"target": "endpoint", "job_id": "job-stale000000"}' \
    --project demo
echo ""

# Step 4: Cancel the pre-warmed endpoint (release node hours)
if [ -n "${JOB_ID}" ]; then
    log "Step 4: Cancel pre-warmed endpoint ${JOB_ID}..."
    $CLIENT job cancel "${JOB_ID}"
    echo ""
else
    log "Step 4: Skipped (could not extract job_id)"
fi

echo ""
banner "Demo 11 Complete"
