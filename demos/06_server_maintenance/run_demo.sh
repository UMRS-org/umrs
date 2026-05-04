#!/usr/bin/env bash
# =============================================================================
# Demo 06: Server Maintenance
# =============================================================================
# Shows: list_buckets, admin-only check/update/recompute, status
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="maintenance" source "${SCRIPT_DIR}/../lib/common.sh"

banner "Demo 06: Server Maintenance"
echo ""

# Step 0: Check server connectivity
log "Step 0: Ping server..."
preflight
echo ""

if [ -z "${UMRS_ADMIN_TOKEN:-}" ]; then
    warn "UMRS_ADMIN_TOKEN is not set. On auth-enforced deployments, check/update/recompute will be denied."
    warn "See demos/23_auth_enforcement for admin token setup."
    echo ""
fi

# Step 1: List configured buckets
log "Step 1: List configured buckets..."
$CLIENT list_buckets || warn "no buckets configured — run Demo 02 first"
echo ""

# Step 2: Check S3 ↔ DB consistency
log "Step 2: Check S3 vs DB consistency..."
admin_check || warn "check returned non-zero"
echo ""

# Step 3: Wait for check to complete, then show result
log "Step 3: Show check result..."
sleep 2
$CLIENT status check || true
echo ""

# Step 4: Update missing/mismatched entries
log "Step 4: Update missing or mismatched entries..."
admin_update || warn "update returned non-zero"
echo ""

# Step 5: Wait and show update result
log "Step 5: Show update result..."
sleep 2
$CLIENT status update || true
echo ""

# Step 6: Full recompute (optional — slow and destructive)
if [ "${DEMO_RECOMPUTE:-0}" = "1" ]; then
    log "Step 6: Recompute all embeddings..."
    admin_recompute || warn "recompute returned non-zero"
    echo ""

    log "Step 7: Show recompute result..."
    sleep 5
    $CLIENT status recompute || true
    echo ""
else
    log "Step 6: Skipping recompute (set DEMO_RECOMPUTE=1 to enable)"
    log "  This re-embeds every object in every bucket."
fi

echo ""
banner "Demo 06 Complete"
