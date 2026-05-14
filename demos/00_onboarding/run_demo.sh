#!/usr/bin/env bash
# =============================================================================
# Demo 00: User Onboarding & Initialization
# =============================================================================
# First demo to run.  Creates a fresh ``demo_user`` with an empty experience
# list and provisions it on the server:
#
#   0. Create user config JSON for demo_user (config init)
#   1. Show active connection profile and user config
#   2. Pre-flight: verify umrs-client is installed and server is reachable
#   3. Onboard: push user config to server → triggers IAM user creation
#   4. Confirm: verify the user is provisioned
#
# This demo is idempotent — it is safe to re-run at any time.
# Subsequent demos add their own experiences via ``ensure_experience``.
#
# To clean up everything at the end, run ``99_cleanup/run_demo.sh``.
#
# Prerequisites:
#   - umrs-client installed and in PATH
#   - UMRS server running and reachable
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="onboarding" source "${SCRIPT_DIR}/../lib/common.sh"

DEMO_USERNAME="${DEMO_USERNAME:-demo_user}"
DEMO_FULLNAME="${DEMO_FULLNAME:-Demo User}"

banner "Demo 00: User Onboarding & Initialization"
echo "Username : ${DEMO_USERNAME}"
echo "Fullname : ${DEMO_FULLNAME}"
echo ""

# ---------- Step 0: Create user config if missing ----------
log "Step 0: Ensure user config for '${DEMO_USERNAME}' exists"
DEMO_CONF="${HOME}/.config/umrs/users/${DEMO_USERNAME}.json"

if [[ -f "${DEMO_CONF}" ]]; then
    ok "config already exists: ${DEMO_CONF}"
else
    log "No user config found — creating one with 'config init'."
    "${CLIENT_ARGS[@]}" config init \
        --username "${DEMO_USERNAME}" \
        --fullname "${DEMO_FULLNAME}" \
        || die "config init failed"
    ok "config created: ${DEMO_CONF}"
fi

# Register the demo_user config as the active user config.
"${CLIENT_ARGS[@]}" config use --user "${DEMO_CONF}" >/dev/null 2>&1 || true
CONF="${DEMO_CONF}"

# ---------- Step 1: Show active profile ----------
log ""
log "Step 1: Active connection profile"
if "${CLIENT_ARGS[@]}" --output text profile show 2>/dev/null; then
    ok "profile displayed"
else
    warn "No profile found — using defaults"
fi
echo ""
log "Active user config: ${CONF}"
ok "config file exists"

# ---------- Step 2: Pre-flight ----------
log ""
log "Step 2: Pre-flight checks"
preflight

# ---------- Step 3: Onboard ----------
log ""
log "Step 3: Onboarding (reload → status → buckets)"
log "  This triggers server-side provisioning:"
log "    • MinIO IAM user + access policy"
log "    • (No experiences yet — subsequent demos will add them)"
onboard "${CONF}"

# ---------- Summary ----------
log ""
log "Onboarding complete."
log "  User '${DEMO_USERNAME}' is provisioned on the server."
log "  Run any demo (02, 03, 04, ...) — each adds its own experience."
log "  Run 99_cleanup to tear everything down when finished."
banner "Demo 00 Complete"
