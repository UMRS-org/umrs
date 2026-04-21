#!/usr/bin/env bash
# =============================================================================
# Demo 00: User Onboarding & Initialization
# =============================================================================
# First demo to run after installing umrs-client and configuring a connection
# profile.  It performs the complete onboarding flow:
#
#   1. Show active connection profile and user config
#   2. Pre-flight: verify umrs-client is installed and server is reachable
#   3. Onboard: push user config to server → triggers bucket + DB creation
#   4. Confirm: verify bucket aliases are visible to the user
#
# This demo is idempotent — it is safe to re-run at any time.
#
# Prerequisites:
#   - umrs-client installed and in PATH
#   - UMRS server running and reachable
#   - User config at ~/.config/umrs/users/user1.json (or UMRS_CONF)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="onboarding" source "${SCRIPT_DIR}/../lib/common.sh"

banner "Demo 00: User Onboarding & Initialization"

# ---------- Step 1: Show active profile ----------
log "Step 1: Active connection profile"
if umrs-client --output text profile show 2>/dev/null; then
    ok "profile displayed"
else
    warn "No profile found — using defaults"
fi
echo ""
log "Active user config: ${CONF}"
[[ -f "${CONF}" ]] || die "Config not found: ${CONF}"
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
log "    • Postgres role + per-user database"
log "    • S3 buckets defined in the user config"
onboard

# ---------- Summary ----------
log ""
log "Onboarding complete."
log "  Your user is provisioned and buckets are ready."
log "  Continue with Demo 01 (Health Check) or Demo 02 (Text Embedding)."
banner "Demo 00 Complete"
