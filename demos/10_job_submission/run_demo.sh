#!/usr/bin/env bash
set -euo pipefail

# Demo 10: Job Submission & Service Lifecycle
# Shows: endpoint_provision, index_build, cleanup, --keep-alive, cancel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT="umrs-client"
SERVER_IP="${UMRS_SERVER_IP:-127.0.0.1}"

echo "=== Demo 10: Job Submission & Service Lifecycle ==="
echo ""

# Step 0: Check server connectivity
echo "--- Step 0: Ping server ---"
${CLIENT} --server-ip "${SERVER_IP}" ping
echo ""

# Step 1: Pre-warm an endpoint (avoids cold start on first upload)
echo "--- Step 1: Pre-warm endpoint (endpoint_provision --keep-alive) ---"
RESULT=$(${CLIENT} --server-ip "${SERVER_IP}" job submit endpoint_provision \
    --spec '{"model": "sam3", "backend": "slurm"}' \
    --project demo --keep-alive)
echo "${RESULT}"
JOB_ID=$(echo "${RESULT}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job',{}).get('job_id',''))" 2>/dev/null || echo "")
echo ""

# Step 2: Rebuild RAG index after bulk inserts
echo "--- Step 2: Rebuild RAG index (index_build) ---"
${CLIENT} --server-ip "${SERVER_IP}" job submit index_build \
    --spec '{"database": "xp_demo", "table": "text_embeddings"}' \
    --project demo
echo ""

# Step 3: Request cleanup of stale resources
echo "--- Step 3: Cleanup stale resources ---"
${CLIENT} --server-ip "${SERVER_IP}" job submit cleanup \
    --spec '{"target": "endpoint", "job_id": "job-stale000000"}' \
    --project demo
echo ""

# Step 4: Cancel the pre-warmed endpoint (release node hours)
if [ -n "${JOB_ID}" ]; then
    echo "--- Step 4: Cancel pre-warmed endpoint ${JOB_ID} ---"
    ${CLIENT} --server-ip "${SERVER_IP}" job cancel "${JOB_ID}"
    echo ""
else
    echo "--- Step 4: Skipped (could not extract job_id) ---"
    echo ""
fi

echo "=== Demo 10 Complete ==="
