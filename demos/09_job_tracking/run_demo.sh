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

# Step 1: List jobs (initially empty)
echo "--- Step 1: List jobs (initial) ---"
${CLIENT} --server-ip "${SERVER_IP}" --output text job list
echo ""

# Step 2: Submit an embedding batch job
echo "--- Step 2: Submit embedding_batch job ---"
RESULT=$(${CLIENT} --server-ip "${SERVER_IP}" job submit embedding_batch \
    --spec '{"bucket": "demo-bucket", "model": "e5-base-v2"}' \
    --project demo --max-attempts 3)
echo "${RESULT}"
# Extract job_id from JSON response
JOB_ID=$(echo "${RESULT}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job',{}).get('job_id',''))" 2>/dev/null || echo "")
echo ""

# Step 3: Submit a second job
echo "--- Step 3: Submit pipeline_run job ---"
${CLIENT} --server-ip "${SERVER_IP}" job submit pipeline_run \
    --spec '{"pipeline": "image-analysis", "input": "s3://images/"}' \
    --project demo
echo ""

# Step 4: List all jobs
echo "--- Step 4: List all jobs ---"
${CLIENT} --server-ip "${SERVER_IP}" --output text job list
echo ""

# Step 5: Get status of first job (with attempt history)
if [ -n "${JOB_ID}" ]; then
    echo "--- Step 5: Status of job ${JOB_ID} ---"
    ${CLIENT} --server-ip "${SERVER_IP}" --output text job status "${JOB_ID}"
    echo ""

    # Step 6: Cancel the job
    echo "--- Step 6: Cancel job ${JOB_ID} ---"
    ${CLIENT} --server-ip "${SERVER_IP}" --output text job cancel "${JOB_ID}"
    echo ""

    # Step 7: Verify cancelled status
    echo "--- Step 7: Verify cancelled status ---"
    ${CLIENT} --server-ip "${SERVER_IP}" --output text job status "${JOB_ID}"
    echo ""
else
    echo "--- Step 5-7: Skipped (could not extract job_id) ---"
    echo ""
fi

# Step 8: List jobs filtered by status
echo "--- Step 8: List cancelled jobs ---"
${CLIENT} --server-ip "${SERVER_IP}" job list --status cancelled
echo ""

echo "=== Demo 09 Complete ==="
