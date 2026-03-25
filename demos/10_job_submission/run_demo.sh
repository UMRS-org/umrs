#!/usr/bin/env bash
set -euo pipefail

# Demo 10: Job Submission & Service Lifecycle
# Shows: endpoint_provision, index_build, cleanup, --keep-alive, cancel
#
# Runs umrs-client inside the container. JSON extraction uses the last line
# of output (umrs-client prints debug lines before the JSON payload).

UMRS_CLIENT_CONTAINER="${UMRS_CLIENT_CONTAINER:-umrs-client}"

run_client() {
    podman exec "$UMRS_CLIENT_CONTAINER" umrs-client "$@"
}

# Extract JSON from umrs-client output (last line starting with '{')
extract_json() {
    grep '^{' | tail -1
}

echo "=== Demo 10: Job Submission & Service Lifecycle ==="
echo ""

# Step 0: Check server connectivity
echo "--- Step 0: Ping server ---"
run_client ping
echo ""

# Step 1: Pre-warm an endpoint (avoids cold start on first upload)
echo "--- Step 1: Pre-warm endpoint (endpoint_provision --keep-alive) ---"
RESULT=$(run_client job submit endpoint_provision \
    --spec '{"model": "sam3", "backend": "slurm"}' \
    --project demo --keep-alive)
echo "${RESULT}"
JOB_ID=$(echo "${RESULT}" | extract_json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job',{}).get('job_id',''))" 2>/dev/null || echo "")
echo ""

# Step 2: Rebuild RAG index after bulk inserts
echo "--- Step 2: Rebuild RAG index (index_build) ---"
run_client job submit index_build \
    --spec '{"database": "xp_demo", "table": "text_embeddings"}' \
    --project demo
echo ""

# Step 3: Request cleanup of stale resources
echo "--- Step 3: Cleanup stale resources ---"
run_client job submit cleanup \
    --spec '{"target": "endpoint", "job_id": "job-stale000000"}' \
    --project demo
echo ""

# Step 4: Cancel the pre-warmed endpoint (release node hours)
if [ -n "${JOB_ID}" ]; then
    echo "--- Step 4: Cancel pre-warmed endpoint ${JOB_ID} ---"
    run_client job cancel "${JOB_ID}"
    echo ""
else
    echo "--- Step 4: Skipped (could not extract job_id) ---"
    echo ""
fi

echo "=== Demo 10 Complete ==="
