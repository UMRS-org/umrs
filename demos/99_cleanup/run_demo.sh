#!/usr/bin/env bash
# =============================================================================
# Demo 99: Full Demo Cleanup
# =============================================================================
# Tears down all resources created during the demo session:
#
#   1. Deprovisions the demo_user on the server:
#      - Empties and removes all S3 buckets
#      - Drops all PostgreSQL databases and roles
#      - Removes the MinIO IAM user and attached policies
#   2. Removes the server-side user JSON file
#   3. Removes client-side config files
#
# After running this, the system is back to a clean state — as if no
# demos were ever run.
#
# Prerequisites:
#   - umrs-server admin CLI accessible (run on the server node)
#   - UMRS server running
#
# Usage:
#   bash run_demo.sh
#   DEMO_USERNAME=other_user bash run_demo.sh   # clean up a different user
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="cleanup" source "${SCRIPT_DIR}/../lib/common.sh"

DEMO_USERNAME="${DEMO_USERNAME:-demo_user}"

banner "Demo 99: Full Demo Cleanup"
echo "User to clean up: ${DEMO_USERNAME}"
echo ""

# ---------- Step 1: Server-side deprovision ----------
log "Step 1: Server-side deprovisioning"
log "  This will remove all S3 buckets, Postgres databases, and the MinIO user."

if command -v umrs-server >/dev/null 2>&1; then
    umrs-server admin user delete "${DEMO_USERNAME}" --purge --deprovision --yes \
        || warn "Server-side deprovisioning failed (user may not exist or server unreachable)."
    ok "Server-side deprovisioning complete."
else
    warn "umrs-server not in PATH — skipping server-side deprovisioning."
    warn "Run this on the server node, or manually run:"
    warn "  umrs-server admin user delete ${DEMO_USERNAME} --purge --deprovision --yes"
fi

# ---------- Step 2: Client-side cleanup ----------
log ""
log "Step 2: Client-side cleanup"

# Remove user config file.
DEMO_CONF="${HOME}/.config/umrs/users/${DEMO_USERNAME}.json"
if [[ -f "${DEMO_CONF}" ]]; then
    rm -f "${DEMO_CONF}"
    ok "Removed user config: ${DEMO_CONF}"
else
    log "  User config not found (already clean): ${DEMO_CONF}"
fi

# Remove user S3 config file.
DEMO_S3_CFG="${HOME}/.config/umrs/s3cfg_${DEMO_USERNAME}"
if [[ -f "${DEMO_S3_CFG}" ]]; then
    rm -f "${DEMO_S3_CFG}"
    ok "Removed S3 config: ${DEMO_S3_CFG}"
else
    log "  S3 config not found (already clean): ${DEMO_S3_CFG}"
fi

# ---------- Summary ----------
log ""
log "Cleanup complete."
log "  All demo resources for '${DEMO_USERNAME}' have been removed."
log "  You can re-run 00_onboarding to start fresh."
banner "Demo 99 Complete"
