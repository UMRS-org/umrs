#!/usr/bin/env bash
# =============================================================================
# Demo 01: Health Check
# =============================================================================
# Verify that the UMRS server is reachable, all required ports are open,
# and the user's configuration is valid.  Run this first to confirm
# connectivity before trying any data pipeline.
#
# Works with any deployment (Docker Compose, rccs-cloud, hpbd-atom, etc.)
# as long as the server is reachable (directly or via SSH tunnels).
#
# Prerequisites:
#   - UMRS server running (any deployment)
#   - Network connectivity (SSH tunnels if remote)
#   - umrs-client installed and in PATH
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UMRS_USER_CONF="${UMRS_USER_CONF:-${HOME}/.config/umrs/users/user1.json}"
DEMO_TAG="health_check" source "${SCRIPT_DIR}/../lib/common.sh"

# Configurable host/ports (defaults to localhost for tunneled setups)
ZMQ_HOST="${UMRS_ZMQ_HOST:-localhost}"
ZMQ_PORT="${UMRS_ZMQ_PORT:-5555}"
ZMQ_DATA_PORT="${UMRS_ZMQ_DATA_PORT:-5556}"
MINIO_HOST="${UMRS_MINIO_HOST:-localhost}"
MINIO_PORT="${UMRS_MINIO_PORT:-9000}"
RAG_URL="${RAG_URL:-http://localhost:8081}"

passed=0
failed=0
_check() {
    local name="$1"; shift
    if "$@" >/dev/null 2>&1; then
        ok "$name"; passed=$((passed + 1))
    else
        warn "$name — FAILED"; failed=$((failed + 1))
    fi
}

banner "Demo 01: Health Check"
echo "Config      : ${CONF}"
echo ""

# ---------- Step 1: UMRS service connectivity ----------
log "Step 1: Ping UMRS server via umrs-client..."
if preflight; then
    passed=$((passed + 1))
else
    failed=$((failed + 1))
fi
echo ""

# ---------- Step 2: Service endpoints (supplemental) ----------
log "Step 2: Check supplemental service endpoints..."

# MinIO is checked directly because demos depend on S3 uploads/downloads.
_check "MinIO S3     (${MINIO_HOST}:${MINIO_PORT})" curl -sf "http://${MINIO_HOST}:${MINIO_PORT}/minio/health/live"

# ZMQ data port binds on-demand after the client negotiates a data channel.
if nc -z "$ZMQ_HOST" "$ZMQ_DATA_PORT" 2>/dev/null; then
    ok "ZMQ data     (${ZMQ_HOST}:${ZMQ_DATA_PORT})"; passed=$((passed + 1))
else
    log "  ZMQ data     (${ZMQ_HOST}:${ZMQ_DATA_PORT}) — not yet bound (opens on first client connect, OK)"
fi

# The control socket may be resolved differently from the effective client path
# in tunnel mode, so treat the raw port probe as informational only.
if nc -z "$ZMQ_HOST" "$ZMQ_PORT" 2>/dev/null; then
    ok "ZMQ control  (${ZMQ_HOST}:${ZMQ_PORT})"
else
    log "  ZMQ control  (${ZMQ_HOST}:${ZMQ_PORT}) — raw port probe skipped as pass/fail; umrs-client ping is authoritative"
fi
echo ""

# ---------- Step 3: RAG service (optional) ----------
log "Step 3: Check RAG service (optional)..."
if curl -sf "${RAG_URL}/docs" >/dev/null 2>&1; then
    ok "RAG service  (${RAG_URL})"; passed=$((passed + 1))
else
    log "  RAG service  (${RAG_URL}) — not available (optional, needs UMRS_RAG_ENABLE=1)"
fi
echo ""

# ---------- Step 4: List configured buckets ----------
log "Step 4: List configured buckets..."
$CLIENT list_buckets || log "(no buckets configured yet — that's OK for a fresh install)"
echo ""

# ---------- Summary ----------
banner "Demo 01 Complete"
log "Results: ${passed} passed, ${failed} failed"
if [[ "$failed" -eq 0 ]]; then
    log "All healthy! Proceed with Demo 02 (Text Embedding)."
else
    log "Some checks failed — ensure the server is running and network is connected."
    log "If remote, check that SSH tunnels are active and that 'umrs-client ping' succeeds."
    exit 1
fi
