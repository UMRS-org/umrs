#!/usr/bin/env bash
# =============================================================================
# Server-side health check for the hpbd-atom UMRS deployment (hybrid mode).
# =============================================================================
# Infrastructure (postgres, seaweedfs) — containers
# umrs-server — native process on the login node
#
# Usage:
#   ./health_check_server.sh
#   ssh atop-0-tmp 'cd /path/to/umrs/deploy/hpbd-atom && bash health_check_server.sh'
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PID_FILE="${SCRIPT_DIR}/server.pid"
RAG_PID_FILE="${SCRIPT_DIR}/rag-service.pid"
LOG_FILE="${SCRIPT_DIR}/logs/server.log"
SITE_CONFIG_PATH="${SCRIPT_DIR}/site.yaml"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

is_truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_model_layout() {
    if [[ -n "${MODEL_DIR:-}" ]]; then
        printf '%s\n%s\n%s\n' \
            "$MODEL_DIR" \
            "$MODEL_DIR/hf" \
            "$MODEL_DIR/sentence_transformers"
        return 0
    fi

    if [[ -x "$VENV_PYTHON" && -f "$SITE_CONFIG_PATH" ]]; then
        if PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
            UMRS_SITE_CONFIG="$SITE_CONFIG_PATH" \
            "$VENV_PYTHON" - <<'PY' 2>/dev/null
from pathlib import Path
from umrs.orchestrator.site_config import load_site_config

default_root = Path("/home/mnagaso/workspace/umrs_models")
try:
    cfg = load_site_config()
    hf_cache = Path(cfg.storage.model_cache_dir) if cfg.storage.model_cache_dir else default_root / "hf"
except Exception:
    hf_cache = default_root / "hf"

model_root = hf_cache.parent
print(model_root)
print(hf_cache)
print(model_root / "sentence_transformers")
PY
        then
            return 0
        fi
    fi

    printf '%s\n%s\n%s\n' \
        "/home/mnagaso/workspace/umrs_models" \
        "/home/mnagaso/workspace/umrs_models/hf" \
        "/home/mnagaso/workspace/umrs_models/sentence_transformers"
}

# Load .env if present (for POSTGRES_USER, POSTGRES_PASSWORD etc.)
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    # shellcheck disable=SC1091
    set -o allexport
    source "$SCRIPT_DIR/.env"
    set +o allexport
fi

POSTGRES_USER="${POSTGRES_USER:-umrs}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD in .env}"
POSTGRES_DB="${POSTGRES_DB:-umrs_db}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:?Set MINIO_ROOT_USER in .env}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:?Set MINIO_ROOT_PASSWORD in .env}"
RAG_ENABLED=false
if is_truthy "${UMRS_RAG_ENABLE:-false}"; then
    RAG_ENABLED=true
fi

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
    log "ERROR: podman not found — run this script on hpbd-atom, not locally."; exit 1
fi

log "Checking hpbd-atom UMRS services (hybrid mode)..."
echo

# ─── Site config ─────────────────────────────────────────────────────────────
if [[ -f "$SITE_CONFIG_PATH" ]]; then
    ok "site config file present"
else
    fail "site config file present"
fi

if [[ -x "$VENV_PYTHON" ]] && \
    PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    UMRS_SITE_CONFIG="$SITE_CONFIG_PATH" "$VENV_PYTHON" - <<'PY' >/dev/null 2>&1
from umrs.orchestrator.site_config import load_site_config

load_site_config()
PY
then
    ok "site config loads via project venv"
else
    fail "site config loads via project venv"
fi

# ─── Container status (infrastructure only) ─────────────────────────────────
check "postgres container running"    podman inspect --format '{{.State.Running}}' umrs-postgres
check "seaweedfs container running"   podman inspect --format '{{.State.Running}}' umrs-seaweedfs

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
check "seaweedfs health endpoint" \
    { curl -sf http://127.0.0.1:9000/status || curl -sf http://127.0.0.1:9000/minio/health/live; }

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

if [[ "$RAG_ENABLED" = true ]]; then
    if [[ -f "$RAG_PID_FILE" ]]; then
        RAG_PID=$(cat "$RAG_PID_FILE")
        if kill -0 "$RAG_PID" 2>/dev/null; then
            ok "rag-service running (PID $RAG_PID)"
        else
            fail "rag-service PID $RAG_PID not running (stale pid file)"
        fi
    else
        fail "rag-service: no rag-service.pid file found"
    fi

    check "RAG service docs respond on 8081" curl -sf http://127.0.0.1:8081/docs
else
    warn "RAG service disabled (set UMRS_RAG_ENABLE=true to enable demos 05 and 19)"
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

# ─── Model cache directories ───
{
    IFS= read -r MODEL_DIR
    IFS= read -r HF_CACHE_DIR
    IFS= read -r _SENTENCE_TRANSFORMERS_DIR
} < <(resolve_model_layout)
if [[ -d "$MODEL_DIR" ]]; then
    ok "Model cache root exists ($MODEL_DIR)"
else
    warn "Model cache root missing: $MODEL_DIR"
fi

if [[ -d "$HF_CACHE_DIR" ]]; then
    ok "HF cache directory exists ($HF_CACHE_DIR)"
else
    warn "HF cache directory missing: $HF_CACHE_DIR"
fi

echo
log "Results: $passed passed, $failed failed"
if [[ "$failed" -eq 0 ]]; then
    log "All healthy!"
else
    log "Some checks failed — see above."
    exit 1
fi
