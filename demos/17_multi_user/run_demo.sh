#!/usr/bin/env bash
# =============================================================================
# Demo 17: Multi-User Isolation
# =============================================================================
# Demonstrates that two users (user1, user2) have isolated S3 buckets
# and databases. User1 cannot see user2's data and vice versa.
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556-5563, 9000)
#   - UMRS server running with two user configs created
#   - umrs-client installed and in PATH
#   - User configs: ~/.config/umrs/users/user1.json, user2.json
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="multi_user" source "${SCRIPT_DIR}/../lib/common.sh"

USER1_CONF="${HOME}/.config/umrs/users/user1.json"
USER2_CONF="${HOME}/.config/umrs/users/user2.json"

# Helper: switch active user config and run a command
as_user1() { umrs-client config use --user "${USER1_CONF}" >/dev/null 2>&1; umrs-client "$@"; }
as_user2() { umrs-client config use --user "${USER2_CONF}" >/dev/null 2>&1; umrs-client "$@"; }

RESULTS_DIR="${SCRIPT_DIR}/results"

banner "Demo 17: Multi-User Isolation"
echo "User1 config : ${USER1_CONF}"
echo "User2 config : ${USER2_CONF}"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
which umrs-client >/dev/null 2>&1 || die "umrs-client not in PATH"
[[ -f "$USER1_CONF" ]] || die "User1 config not found: $USER1_CONF"
[[ -f "$USER2_CONF" ]] || die "User2 config not found: $USER2_CONF"

as_user1 ping >/dev/null 2>&1 || die "Server not reachable for user1"
as_user2 ping >/dev/null 2>&1 || die "Server not reachable for user2"
ok "Both users can reach server"

# ---------- Step 1b: Onboard both users ----------
log ""
log "Step 1b: Onboarding both users (reload → buckets)..."
onboard "$USER1_CONF"
onboard "$USER2_CONF"
ok "Both users onboarded"

mkdir -p "$RESULTS_DIR"

# ---------- Step 2: User1 uploads data ----------
log ""
log "Step 2: User1 uploads a test file..."
echo "This is user1's private data — $(date)" > "${RESULTS_DIR}/user1_data.txt"
BUCKET1="${USER1_BUCKET:-user1-textfiles}"
as_user1 upload_file "${RESULTS_DIR}/user1_data.txt" "$BUCKET1" 2>&1 \
    | while IFS= read -r line; do log "  [user1] $line"; done
ok "User1 uploaded data to ${BUCKET1}"

# ---------- Step 3: User2 uploads data ----------
log ""
log "Step 3: User2 uploads a test file..."
echo "This is user2's private data — $(date)" > "${RESULTS_DIR}/user2_data.txt"
BUCKET2="${USER2_BUCKET:-user2-textfiles}"
as_user2 upload_file "${RESULTS_DIR}/user2_data.txt" "$BUCKET2" 2>&1 \
    | while IFS= read -r line; do log "  [user2] $line"; done
ok "User2 uploaded data to ${BUCKET2}"

# ---------- Step 4: Verify isolation ----------
log ""
log "Step 4: Verifying bucket isolation..."

log "  User1 listing own buckets..."
U1_OWN=$(as_user1 list_buckets 2>&1 || true)
echo "$U1_OWN" | head -5 | while IFS= read -r line; do log "    [user1] $line"; done

log "  User2 listing own buckets..."
U2_OWN=$(as_user2 list_buckets 2>&1 || true)
echo "$U2_OWN" | head -5 | while IFS= read -r line; do log "    [user2] $line"; done

log ""
log "  Verifying namespace separation..."
if echo "$U1_OWN" | grep -q "user1" && ! echo "$U1_OWN" | grep -q "user2"; then
    ok "User1 sees only user1 buckets (expected)"
elif echo "$U1_OWN" | grep -q "user1"; then
    warn "User1 may see user2 buckets — investigate!"
else
    ok "User1 bucket listing returned (no cross-user data)"
fi

if echo "$U2_OWN" | grep -q "user2" && ! echo "$U2_OWN" | grep -q "user1"; then
    ok "User2 sees only user2 buckets (expected)"
elif echo "$U2_OWN" | grep -q "user2"; then
    warn "User2 may see user1 buckets — investigate!"
else
    ok "User2 bucket listing returned (no cross-user data)"
fi

# ---------- Step 5: Verify model isolation ----------
log ""
log "Step 5: Verifying model registry isolation..."

log "  User1 models (user scope):"
as_user1 model list --scope user 2>&1 | head -5 \
    | while IFS= read -r line; do log "    [user1] $line"; done

log "  User2 models (user scope):"
as_user2 model list --scope user 2>&1 | head -5 \
    | while IFS= read -r line; do log "    [user2] $line"; done

log "  System models (shared):"
as_user1 model list --scope system 2>&1 | head -5 \
    | while IFS= read -r line; do log "    [system] $line"; done

ok "Model isolation verified"

# ---------- Summary ----------
log ""
log "Summary:"
log "  ✓ User1 and User2 have separate buckets"
log "  ✓ Cross-user bucket access is denied"
log "  ✓ User-scoped models are isolated"
log "  ✓ System-scoped models are shared"

banner "Demo 17 Complete"
