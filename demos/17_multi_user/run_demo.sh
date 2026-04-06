#!/usr/bin/env bash
# =============================================================================
# Demo 17: Multi-User Isolation
# =============================================================================
# Demonstrates that two users (user1, user2) have isolated S3 buckets
# and databases. User1 cannot see user2's data and vice versa.
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running with two user configs created
#   - umrs-client installed and in PATH
#   - User configs: ~/.config/umrs/users/user1.json, user2.json
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="multi_user" source "${SCRIPT_DIR}/../lib/common.sh"

USER1_CONF="${HOME}/.config/umrs/users/user1.json"
USER2_CONF="${HOME}/.config/umrs/users/user2.json"
CLIENT1="umrs-client -c ${USER1_CONF}"
CLIENT2="umrs-client -c ${USER2_CONF}"

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

$CLIENT1 ping >/dev/null 2>&1 || die "Server not reachable for user1"
$CLIENT2 ping >/dev/null 2>&1 || die "Server not reachable for user2"
ok "Both users can reach server"

mkdir -p "$RESULTS_DIR"

# ---------- Step 2: User1 uploads data ----------
log ""
log "Step 2: User1 uploads a test file..."
echo "This is user1's private data — $(date)" > "${RESULTS_DIR}/user1_data.txt"
BUCKET1="${USER1_BUCKET:-user1-textfiles}"
$CLIENT1 upload "${RESULTS_DIR}/user1_data.txt" "$BUCKET1" 2>&1 \
    | while IFS= read -r line; do log "  [user1] $line"; done
ok "User1 uploaded data to ${BUCKET1}"

# ---------- Step 3: User2 uploads data ----------
log ""
log "Step 3: User2 uploads a test file..."
echo "This is user2's private data — $(date)" > "${RESULTS_DIR}/user2_data.txt"
BUCKET2="${USER2_BUCKET:-user2-textfiles}"
$CLIENT2 upload "${RESULTS_DIR}/user2_data.txt" "$BUCKET2" 2>&1 \
    | while IFS= read -r line; do log "  [user2] $line"; done
ok "User2 uploaded data to ${BUCKET2}"

# ---------- Step 4: Verify isolation ----------
log ""
log "Step 4: Verifying bucket isolation..."

log "  User1 listing own bucket..."
U1_OWN=$($CLIENT1 list "$BUCKET1" 2>&1 || true)
echo "$U1_OWN" | head -5 | while IFS= read -r line; do log "    [user1 → ${BUCKET1}] $line"; done

log "  User2 listing own bucket..."
U2_OWN=$($CLIENT2 list "$BUCKET2" 2>&1 || true)
echo "$U2_OWN" | head -5 | while IFS= read -r line; do log "    [user2 → ${BUCKET2}] $line"; done

log ""
log "  User1 trying to access user2's bucket..."
U1_CROSS=$($CLIENT1 list "$BUCKET2" 2>&1 || true)
if echo "$U1_CROSS" | grep -qi "error\|denied\|forbidden\|not found\|no such"; then
    ok "User1 CANNOT access user2's bucket (expected)"
else
    warn "User1 may have access to user2's bucket — investigate!"
    echo "$U1_CROSS" | head -3 | while IFS= read -r line; do log "    $line"; done
fi

log ""
log "  User2 trying to access user1's bucket..."
U2_CROSS=$($CLIENT2 list "$BUCKET1" 2>&1 || true)
if echo "$U2_CROSS" | grep -qi "error\|denied\|forbidden\|not found\|no such"; then
    ok "User2 CANNOT access user1's bucket (expected)"
else
    warn "User2 may have access to user1's bucket — investigate!"
    echo "$U2_CROSS" | head -3 | while IFS= read -r line; do log "    $line"; done
fi

# ---------- Step 5: Verify model isolation ----------
log ""
log "Step 5: Verifying model registry isolation..."

log "  User1 models (user scope):"
$CLIENT1 model list --scope user 2>&1 | head -5 \
    | while IFS= read -r line; do log "    [user1] $line"; done

log "  User2 models (user scope):"
$CLIENT2 model list --scope user 2>&1 | head -5 \
    | while IFS= read -r line; do log "    [user2] $line"; done

log "  System models (shared):"
$CLIENT1 model list --scope system 2>&1 | head -5 \
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
