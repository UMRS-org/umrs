#!/usr/bin/env bash
# =============================================================================
# Health Check — rccs-cloud deployment
# =============================================================================
# Verifies that SSH tunnels are working and UMRS services are reachable.
# Run from your local Mac with SSH tunnels active.
# =============================================================================
set -euo pipefail

USER_CONF="${UMRS_USER_CONF:-$HOME/.config/umrs/users/user1.json}"

log()  { echo "[health] $*"; }
ok()   { echo "[health]   ✓ $1"; }
fail() { echo "[health]   ✗ $1"; }

passed=0
failed=0

check() {
    local name="$1"; shift
    if "$@" >/dev/null 2>&1; then
        ok "$name"; passed=$((passed + 1))
    else
        fail "$name"; failed=$((failed + 1))
    fi
}

log "Checking rccs-cloud services via SSH tunnels..."
echo

# ZMQ control port
check "ZMQ control (localhost:5555)" nc -z localhost 5555

# ZMQ data port
check "ZMQ data    (localhost:5556)" nc -z localhost 5556

# MinIO S3
check "MinIO S3    (localhost:9000)" curl -sf http://localhost:9000/minio/health/live

# UMRS server ping
check "UMRS server (ping)" umrs-client -c "$USER_CONF" ping

# RAG service (optional — only when UMRS_RAG_ENABLE=1 on the server)
if curl -sf http://localhost:8081/docs >/dev/null 2>&1; then
    ok "RAG service (localhost:8081)"
else
    log "   RAG service (localhost:8081) — not available (optional, needs UMRS_RAG_ENABLE=1)"
fi

echo
log "Results: $passed passed, $failed failed"
if [[ "$failed" -eq 0 ]]; then
    log "All healthy!"
else
    log "Some checks failed — ensure ssh_tunnel.sh is running."
    exit 1
fi
