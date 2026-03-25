#!/usr/bin/env bash
set -euo pipefail

# Demo 05: Server Maintenance
# Shows: list_buckets, check, update, recompute, status
#
# Runs umrs-client inside the umrs-client container (has correct user config
# and UMRS_SERVER_HOST env var). Host-side umrs-client picks up the admin
# config which lacks a 'username' field.

UMRS_CLIENT_CONTAINER="${UMRS_CLIENT_CONTAINER:-umrs-client}"

run_client() {
    podman exec "$UMRS_CLIENT_CONTAINER" umrs-client "$@"
}

echo "=== Demo 05: Server Maintenance ==="
echo ""

# Step 0: Check server connectivity
echo "--- Step 0: Ping server ---"
run_client ping
echo ""

# Step 1: List configured buckets
echo "--- Step 1: List configured buckets ---"
run_client list_buckets || echo "(no buckets configured — run Demo 01 first)"
echo ""

# Step 2: Check S3 ↔ DB consistency
echo "--- Step 2: Check S3 vs DB consistency ---"
run_client check
echo ""

# Step 3: Wait for check to complete, then show result
echo "--- Step 3: Show check result ---"
sleep 2
run_client status check
echo ""

# Step 4: Update missing/mismatched entries
echo "--- Step 4: Update missing or mismatched entries ---"
run_client update
echo ""

# Step 5: Wait and show update result
echo "--- Step 5: Show update result ---"
sleep 2
run_client status update
echo ""

# Step 6: Full recompute (optional — slow and destructive)
if [ "${DEMO_RECOMPUTE:-0}" = "1" ]; then
    echo "--- Step 6: Recompute all embeddings ---"
    run_client recompute
    echo ""

    echo "--- Step 7: Show recompute result ---"
    sleep 5
    run_client status recompute
    echo ""
else
    echo "--- Step 6: Skipping recompute (set DEMO_RECOMPUTE=1 to enable) ---"
    echo "  This re-embeds every object in every bucket. Use after changing the embedding model."
    echo ""
fi

echo "=== Demo 05 Complete ==="
