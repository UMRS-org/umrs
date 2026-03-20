#!/usr/bin/env bash
set -euo pipefail

# Demo 09: Job Tracking via ZeroMQ
# Shows: job submit, job list, job status, job cancel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT="umrs-client"
SERVER_IP="${UMRS_SERVER_IP:-127.0.0.1}"

echo "=== Demo 09: Job Tracking via ZeroMQ ==="
echo ""

# Step 0: Check server connectivity
echo "--- Step 0: Ping server ---"
${CLIENT} --server-ip "${SERVER_IP}" ping
echo ""

# Step 1: Check current task state (use status, not job list)
echo "--- Step 1: Check current task status ---"
echo "  Tip: 'umrs-client status all' shows running tasks from the in-memory store"
echo ""

# Step 2: Submit an embedding batch job
echo "--- Step 2: Submit embedding_batch job ---"
RESULT=$(${CLIENT} --server-ip "${SERVER_IP}" job submit embedding_batch \
    --spec '{"bucket": "demo-bucket", "model": "e5-base-v2"}' \
    --project demo --max-attempts 3)
echo "${RESULT}"
JOB_ID=$(echo "${RESULT}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job',{}).get('job_id',''))" 2>/dev/null || echo "")
echo ""

# Step 3: Submit a second job
echo "--- Step 3: Submit pipeline_run job ---"
${CLIENT} --server-ip "${SERVER_IP}" job submit pipeline_run \
    --spec '{"pipeline": "image-analysis", "input": "s3://images/"}' \
    --project demo
echo ""

# Step 4: Cancel the first job
if [ -n "${JOB_ID}" ]; then
    echo "--- Step 4: Cancel job ${JOB_ID} ---"
    ${CLIENT} --server-ip "${SERVER_IP}" job cancel "${JOB_ID}"
    echo ""
else
    echo "--- Step 4: Skipped (could not extract job_id) ---"
    echo ""
fi

echo "=== Demo 09 Complete ==="
