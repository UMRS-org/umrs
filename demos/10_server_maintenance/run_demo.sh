#!/usr/bin/env bash
set -euo pipefail

# Demo 10: Server Maintenance
# Shows: list_buckets, check, update, recompute, status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT="umrs-client"
SERVER_IP="${UMRS_SERVER_IP:-127.0.0.1}"

echo "=== Demo 10: Server Maintenance ==="
echo ""

# Step 0: Check server connectivity
echo "--- Step 0: Ping server ---"
${CLIENT} --server-ip "${SERVER_IP}" ping
echo ""

# Step 1: List configured buckets
echo "--- Step 1: List configured buckets ---"
${CLIENT} --server-ip "${SERVER_IP}" list_buckets || echo "(no buckets configured — run Demo 01 first)"
echo ""

# Step 2: Check S3 ↔ DB consistency
echo "--- Step 2: Check S3 vs DB consistency ---"
${CLIENT} --server-ip "${SERVER_IP}" check
echo ""

# Step 3: Wait for check to complete, then show result
echo "--- Step 3: Show check result ---"
sleep 2
${CLIENT} --server-ip "${SERVER_IP}" status check
echo ""

# Step 4: Update missing/mismatched entries
echo "--- Step 4: Update missing or mismatched entries ---"
${CLIENT} --server-ip "${SERVER_IP}" update
echo ""

# Step 5: Wait and show update result
echo "--- Step 5: Show update result ---"
sleep 2
${CLIENT} --server-ip "${SERVER_IP}" status update
echo ""

# Step 6: Full recompute (optional — slow and destructive)
if [ "${DEMO_RECOMPUTE:-0}" = "1" ]; then
    echo "--- Step 6: Recompute all embeddings ---"
    ${CLIENT} --server-ip "${SERVER_IP}" recompute
    echo ""

    echo "--- Step 7: Show recompute result ---"
    sleep 5
    ${CLIENT} --server-ip "${SERVER_IP}" status recompute
    echo ""
else
    echo "--- Step 6: Skipping recompute (set DEMO_RECOMPUTE=1 to enable) ---"
    echo "  This re-embeds every object in every bucket. Use after changing the embedding model."
    echo ""
fi

echo "=== Demo 10 Complete ==="
