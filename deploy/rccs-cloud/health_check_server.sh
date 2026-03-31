#!/usr/bin/env bash
# =============================================================================
# Server-side health check for the rccs-cloud UMRS deployment (hybrid mode).
# =============================================================================
# Infrastructure (postgres, minio) — containers
# umrs-server — native process on the login node
#
# Usage:
#   ./health_check_server.sh
#   ssh rccs-cloud 'cd /path/to/umrs/deploy/rccs-cloud && bash health_check_server.sh'
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/server.pid"
LOG_FILE="${SCRIPT_DIR}/logs/server.log"

# Load .env if present (for POSTGRES_USER, POSTGRES_PASSWORD etc.)
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    # shellcheck disable=SC1091
    set -o allexport
    source "$SCRIPT_DIR/.env"
    set +o allexport
fi

POSTGRES_USER="${POSTGRES_USER:-umrs}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-umrs_rccs}"
POSTGRES_DB="${POSTGRES_DB:-umrs_db}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"

log()  { echo "[server-health] $*"; }
ok()   { echo "[server-health]   ✓ $1"; passed=$((passed + 1)); }
warn() { echo "[server-health]   ! $1 (warning)"; }
fail() { echo "[server-health]   ✗ $1"; failed=$((failed + 1)); }

passed=0
failed=0

check() {
    local name="$1"; shift
    if "$@" >/dev/null 2>&1; then ok "$name"; else fail "$name"; fi
}

if ! command -v podman >/dev/null 2>&1; then
    log "ERROR: podman not found — run this script on rccs-cloud, not locally."; exit 1
fi

log "Checking rccs-cloud UMRS services (hybrid mode)..."
echo

# ─── Container status (infrastructure only) ─────────────────────────────────
check "postgres container running"    podman inspect --format '{{.State.Running}}' umrs-postgres
check "minio container running"       podman inspect --format '{{.State.Running}}' umrs-minio

# ─── PostgreSQL ───
check "postgres accepting connections" \
    podman exec umrs-postgres pg_isready -h localhost -U "$POSTGRES_USER"

# ─── pgvector extension (check availability + usage in any database) ───
if podman exec umrs-postgres psql -h localhost -U "$POSTGRES_USER" -d postgres \
    -tAc "SELECT 1 FROM pg_available_extensions WHERE name='vector';" 2>/dev/null | grep -q 1; then
    ok "pgvector extension available"
else
    fail "pgvector extension available"
fi

# ─── MinIO ───
check "minio health endpoint" \
    curl -sf http://127.0.0.1:9000/minio/health/live

# ─── Native umrs-server process ─────────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    SERVER_PID=$(cat "$PID_FILE")
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        ok "umrs-server running (PID $SERVER_PID)"
    else
        fail "umrs-server PID $SERVER_PID not running (stale pid file)"
    fi
else
    fail "umrs-server: no server.pid file found"
fi

# ─── ZMQ ports (umrs-server) ───
check "ZMQ control port 5555 listening" nc -z 127.0.0.1 5555
# Port 5556 (data plane) opens on demand when a client connects — warn only
if nc -z 127.0.0.1 5556 2>/dev/null; then
    ok "ZMQ data port 5556 listening"
else
    warn "ZMQ data port 5556 not open (opens on first client connection)"
fi

# ─── Server logs: no recent errors ──────────────────────────────────────────
if [[ -f "$LOG_FILE" ]]; then
    # Check last 100 lines for recent errors
    recent_errors="$(tail -100 "$LOG_FILE" 2>/dev/null | grep -c 'CRITICAL\|Traceback\|ERROR' || true)"
    if [[ "$recent_errors" -eq 0 ]]; then
        ok "umrs-server logs (no recent errors)"
    else
        warn "umrs-server has $recent_errors recent error/traceback lines — check: tail -50 $LOG_FILE"
    fi
else
    warn "umrs-server log not found at $LOG_FILE"
fi

# ─── Slurm (optional) ───
if command -v sinfo >/dev/null 2>&1; then
    idle_nodes="$(sinfo -h -o '%n %T' 2>/dev/null | grep -c 'idle' || true)"
    if [[ "$idle_nodes" -gt 0 ]]; then
        ok "Slurm: $idle_nodes idle compute node(s)"
    else
        warn "Slurm: no idle nodes currently available"
    fi
else
    log "   Slurm check skipped (sinfo not found)"
fi

# ─── Model cache directory ───
MODEL_DIR="${MODEL_DIR:-/lvs0/rccs-dmp/mnagaso/umrs_models}"
if [[ -d "$MODEL_DIR" ]]; then
    ok "Model cache directory exists ($MODEL_DIR)"
else
    warn "Model cache directory missing: $MODEL_DIR"
fi

echo
log "Results: $passed passed, $failed failed"
if [[ "$failed" -eq 0 ]]; then
    log "All healthy!"
else
    log "Some checks failed — see above."
    exit 1
fi
