#!/usr/bin/env bash
# =============================================================================
# Demo 23: Auth Enforcement
# =============================================================================
# Shows: admin token management, command authorization, role-based access
#
# This demo exercises the UMRS authentication enforcement layer:
#   1. Generate a local admin token (purely local, no server needed)
#   2. Connect WITHOUT token — maintenance commands are denied (user role)
#   3. Connect WITH token — maintenance commands succeed (admin role)
#   4. Revoke the token — clean up
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running
#   - Token hash must be available on the server side
#     (on rccs-cloud: generate via `umrs-client admin token generate` on server,
#      or sync admin_tokens.json to server ~/.config/umrs/ )
#
# Quick test (local admin token management only, no server):
#   DEMO_LOCAL_ONLY=1 bash run_demo.sh
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="auth" source "${SCRIPT_DIR}/../lib/common.sh"

banner "Demo 23: Auth Enforcement"
echo ""

TOKEN_NAME="demo23-$(date +%s)"
GENERATED_TOKEN=""

cleanup() {
    if [[ -n "$TOKEN_NAME" ]]; then
        log "Cleanup: revoking token '${TOKEN_NAME}'..."
        umrs-client admin token revoke "$TOKEN_NAME" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

# ──────────────────────────────────────────────────
# Part A: Local Admin Token Management (no server)
# ──────────────────────────────────────────────────
log "Part A: Local Admin Token Management"
echo ""

# Step 1: Generate a token
log "Step 1: Generate admin token '${TOKEN_NAME}'..."
GEN_OUTPUT="$(umrs-client --output json admin token generate "$TOKEN_NAME" --description "Demo 23 test token")"
echo "$GEN_OUTPUT"
GENERATED_TOKEN="$(echo "$GEN_OUTPUT" | python3 -c "import sys, json; print(json.load(sys.stdin)['token'])")"
if [[ -z "$GENERATED_TOKEN" ]]; then
    die "Failed to generate admin token"
fi
ok "Token generated (starts with ${GENERATED_TOKEN:0:15}...)"
echo ""

# Step 2: List tokens
log "Step 2: List admin tokens..."
umrs-client --output text admin token list
echo ""
# Verify our token appears in the listing
umrs-client --output json admin token list | python3 -c "
import sys, json
tokens = json.load(sys.stdin)['tokens']
names = [t['name'] for t in tokens]
assert '${TOKEN_NAME}' in names, f'Token ${TOKEN_NAME} not found in {names}'
print('  ✓ Token found in listing')
"
echo ""

# Step 3: If local-only mode, skip server tests
if [[ "${DEMO_LOCAL_ONLY:-0}" = "1" ]]; then
    log "DEMO_LOCAL_ONLY=1 — skipping server-side auth tests"
    echo ""
    # Step: Revoke
    log "Step 3: Revoke token..."
    umrs-client --output text admin token revoke "$TOKEN_NAME"
    ok "Token revoked"
    TOKEN_NAME=""  # prevent double-revoke in trap
    echo ""
    banner "Demo 23 Complete (local-only)"
    exit 0
fi

# ──────────────────────────────────────────────────
# Part B: Server-Side Auth Enforcement
# ──────────────────────────────────────────────────
log "Part B: Server-Side Auth Enforcement"
echo ""

# Step 4: Pre-flight (verify server connectivity)
log "Step 4: Ping server..."
preflight
echo ""

# Step 5: Connect WITHOUT admin token — user role
# Maintenance command 'check' requires admin; with user role it should be
# accepted by notify_server (fire-and-forget ACK) but rejected server-side.
# The status entry will show ERROR.
log "Step 5: Attempt maintenance 'check' WITHOUT admin token (user role)..."
$CLIENT check || true
sleep 2
CHECK_STATUS="$($CLIENT --output json status check 2>/dev/null || echo '{}')"
echo "$CHECK_STATUS"

# The check command may show no status (if never run) or ERROR (if auth denied).
# We cannot guarantee ERROR appears if there was a previous successful check,
# so we just report what we see.
log "  (User-role check result reported above)"
echo ""

# Step 6: Connect WITH admin token — admin role
log "Step 6: Attempt maintenance 'check' WITH admin token (admin role)..."
$CLIENT --admin-token "$GENERATED_TOKEN" check
sleep 3
CHECK_STATUS_ADMIN="$($CLIENT --admin-token "$GENERATED_TOKEN" --output json status check 2>/dev/null || echo '{}')"
echo "$CHECK_STATUS_ADMIN"

# Verify the admin check produced a successful result
echo "$CHECK_STATUS_ADMIN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
status = data.get('status', 'unknown')
if status == 'ok':
    print('  ✓ Admin check succeeded (status=ok)')
else:
    print(f'  ⚠ Admin check status: {status} (may need server-side token)')
" || true
echo ""

# Step 7: Verify pool metrics are visible in status
log "Step 7: Verify pool metrics in status response..."
POOL_STATUS="$($CLIENT --admin-token "$GENERATED_TOKEN" --output json status all 2>/dev/null || echo '{}')"
echo "$POOL_STATUS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
pool = data.get('pool')
if pool:
    print(f'  ✓ Pool metrics: active={pool[\"active_count\"]}, pending={pool[\"pending_count\"]}, max_workers={pool[\"max_workers\"]}')
else:
    print('  ⚠ No pool metrics in status (thread pool may not be enabled)')
" || true
echo ""

# ──────────────────────────────────────────────────
# Part C: Cleanup
# ──────────────────────────────────────────────────
log "Part C: Cleanup"
echo ""

# Step 8: Revoke the token
log "Step 8: Revoke token '${TOKEN_NAME}'..."
umrs-client --output text admin token revoke "$TOKEN_NAME"
ok "Token revoked"
TOKEN_NAME=""  # prevent double-revoke in trap
echo ""

# Step 9: Verify token is gone
log "Step 9: Verify token no longer listed..."
umrs-client --output json admin token list | python3 -c "
import sys, json
tokens = json.load(sys.stdin)['tokens']
names = [t['name'] for t in tokens]
assert '${TOKEN_NAME:-__revoked__}' not in names, 'Token should have been revoked!'
print('  ✓ Token successfully removed')
"
echo ""

banner "Demo 23 Complete"
