#!/usr/bin/env bash
# =============================================================================
# Start the UMRS stack on rccs-cloud — hybrid architecture.
# =============================================================================
# Infrastructure (postgres, minio) runs in podman containers.
# The umrs-server runs NATIVELY on the login node so it can call Slurm CLI
# (sbatch, squeue, scancel) and create SSH tunnels to compute nodes.
#
# Usage (on rccs-cloud):
#   cd deploy/rccs-cloud
#   ./start.sh              # start everything
#   ./start.sh --server     # restart only the native server (containers stay)
#
# Prerequisites:
#   - Run ./setup_server.sh once (creates venv, installs deps)
#   - Edit .env (copied from env.example on first run)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ─── Work around HPC login-node thread limit for Go/Podman ──────────────────
# Go's runtime defaults GOMAXPROCS to the number of CPUs visible via nproc.
# On large HPC login nodes (e.g. 64 cores) the Go runtime tries to create 64+
# OS threads at startup, which can exceed per-session thread-creation limits
# imposed by the system (systemd TasksMax, PAM, cgroup, or RLIMIT_NPROC).
# Capping GOMAXPROCS to a small value prevents the pthread_create EAGAIN crash
# that otherwise makes podman (and any Go binary) unusable.
export GOMAXPROCS="${GOMAXPROCS:-4}"

SCRIPT_DIR="$(pwd)"
REPO_ROOT="$(cd ../.. && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PID_FILE="${SCRIPT_DIR}/server.pid"
RAG_PID_FILE="${SCRIPT_DIR}/rag-service.pid"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/server.log"
RAG_LOG_FILE="${LOG_DIR}/rag-service.log"

usage() {
        cat <<'EOF'
Usage: ./start.sh [--server]

Start the RCCS UMRS stack.

Options:
    --server      Restart only the native UMRS server and RAG service; keep containers and Slurm jobs running.
    -h, --help    Show this help and exit without changing running services.

Examples:
    ./start.sh
    ./start.sh --server
EOF
}

START_MODE="${1:-}"
case "$START_MODE" in
        ""|--server )
                ;;
        -h|--help )
                usage
                exit 0
                ;;
        * )
                echo "ERROR: Unknown option: ${START_MODE}"
                usage
                exit 1
                ;;
esac

check_zmq_ready() {
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        return 1
    fi

    "${VENV_DIR}/bin/python" - <<'PY' >/dev/null 2>&1
from umrs.core.MessageQueue import MessageQueue

mq = MessageQueue(role="client", queue_type="control", server_ip="127.0.0.1")
try:
    raise SystemExit(0 if mq.is_server_ready(timeout_ms=2000) else 1)
finally:
    mq.close()
PY
}

check_server_bootstrap_complete() {
    local log_offset="$1"

    [[ -f "$LOG_FILE" ]] || return 1

    tail -c "+$((log_offset + 1))" "$LOG_FILE" 2>/dev/null \
        | grep -Fq "Server configuration done!"
}

check_postgres_catalog_ready() {
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        return 1
    fi

    "${VENV_DIR}/bin/python" - <<'PY' 2>&1
import os

import psycopg2

host = os.environ.get("DB_HOST") or "127.0.0.1"
port = os.environ.get("DB_PORT") or "5432"
user = os.environ.get("DB_SUPERUSER") or os.environ.get("POSTGRES_USER") or "umrs"
password = os.environ.get("DB_SUPERUSER_PASSWORD") or os.environ.get("POSTGRES_PASSWORD") or ""

try:
    with psycopg2.connect(
        dbname="postgres",
        user=user,
        password=password,
        host=host,
        port=port,
    ) as conn:
        with conn.cursor() as cur:
            # Mirror the first role lookup used during user provisioning so
            # catalog corruption is caught before the native server starts.
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s;", (user,))
            cur.fetchone()
except Exception as exc:
    print(exc)
    raise SystemExit(1)
PY
}

show_recent_log_tail() {
    local log_file="$1"
    local lines="${2:-40}"

    [[ -f "$log_file" ]] || return 0

    echo ""
    echo "Recent log tail (${log_file}):"
    tail -n "$lines" "$log_file" || true
}

check_rag_ready() {
    curl -sf "http://127.0.0.1:8081/docs" >/dev/null 2>&1
}

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

stop_managed_process() {
    local pid_file="$1"
    local label="$2"

    if [[ ! -f "$pid_file" ]]; then
        return 0
    fi

    local process_pid
    process_pid=$(cat "$pid_file")
    if kill -0 "$process_pid" 2>/dev/null; then
        echo "Stopping existing ${label} (PID ${process_pid})..."
        kill "$process_pid" 2>/dev/null || true
        sleep 2
        if kill -0 "$process_pid" 2>/dev/null; then
            kill -9 "$process_pid" 2>/dev/null || true
            sleep 1
        fi
    fi
    rm -f "$pid_file"
}

rootlessport_pid_for_port() {
    local port="$1"

    ss -ltnp "sport = :${port}" 2>/dev/null \
        | awk -F'pid=' '/rootlessport/ {print $2}' \
        | awk -F',' '{print $1}' \
        | head -1
}

reap_stale_rootlessport() {
    local port="$1"
    local pid

    pid="$(rootlessport_pid_for_port "$port")"
    if [[ -z "$pid" ]]; then
        return 0
    fi

    echo "Reaping stale rootlessport on 127.0.0.1:${port} (PID ${pid})..."
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi
}

container_exists() {
    local name="$1"

    podman inspect --type container "$name" >/dev/null 2>&1
}

kill_stuck_container_runtime() {
    local name="$1"
    local container_pid
    local conmon_pid

    container_pid="$(podman inspect --type container --format '{{.State.Pid}}' "$name" 2>/dev/null || true)"
    conmon_pid="$(podman inspect --type container --format '{{.State.ConmonPid}}' "$name" 2>/dev/null || true)"

    if [[ -n "$container_pid" && "$container_pid" != "0" ]]; then
        kill -9 "$container_pid" 2>/dev/null || true
    fi

    if [[ -n "$conmon_pid" && "$conmon_pid" != "0" ]]; then
        kill -9 "$conmon_pid" 2>/dev/null || true
    fi

    sleep 1
}

reconcile_infra_container() {
    local name="$1"
    shift
    local ports=("$@")
    local state

    state="$(podman inspect --type container --format '{{.State.Status}}' "$name" 2>/dev/null || true)"

    if [[ -z "$state" ]]; then
        for port in "${ports[@]}"; do
            reap_stale_rootlessport "$port"
        done
        return 0
    fi

    if [[ "$state" == "running" ]]; then
        return 0
    fi

    echo "Removing stale ${name} container (${state}) before compose up..."
    if ! podman rm -f "$name" >/dev/null 2>&1; then
        echo "Podman could not remove ${name} directly; killing host-side container processes..."
        kill_stuck_container_runtime "$name"
        podman rm -f "$name" >/dev/null 2>&1 || true
    fi

    for port in "${ports[@]}"; do
        reap_stale_rootlessport "$port"
    done

    for _ in $(seq 1 10); do
        if ! container_exists "$name"; then
            return 0
        fi
        sleep 0.5
    done

    echo "ERROR: Failed to remove stale ${name} container."
    podman inspect --type container --format 'state={{.State.Status}} pid={{.State.Pid}} conmon={{.State.ConmonPid}} err={{.State.Error}}' "$name" || true
    return 1
}

resolve_model_layout() {
    if [[ -n "${MODEL_DIR:-}" ]]; then
        printf '%s\n%s\n%s\n' \
            "$MODEL_DIR" \
            "$MODEL_DIR/hf" \
            "$MODEL_DIR/sentence_transformers"
        return 0
    fi

    if [[ -x "${VENV_DIR}/bin/python" && -f "${UMRS_SITE_CONFIG:-}" ]]; then
        if PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
            UMRS_SITE_CONFIG="$UMRS_SITE_CONFIG" \
            "${VENV_DIR}/bin/python" - <<'PY' 2>/dev/null
import os
from pathlib import Path
from umrs.orchestrator.site_config import load_site_config

default_root = Path(f"/lvs0/dne1/rccs-dmp/{os.environ.get('USER', 'test-operator')}/umrs_models")
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
        "/lvs0/dne1/rccs-dmp/${USER}/umrs_models" \
        "/lvs0/dne1/rccs-dmp/${USER}/umrs_models/hf" \
        "/lvs0/dne1/rccs-dmp/${USER}/umrs_models/sentence_transformers"
}

# ─── Guard: must run on a Linux host where podman is available ───────────────
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: start.sh must be run on rccs-cloud (Linux), not on $(uname -s)."
    echo "  Run:  ssh rccs-cloud 'cd /path/to/umrs/deploy/rccs-cloud && ./start.sh'"
    exit 1
fi

# ─── Check prerequisites ────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Server venv not found at ${VENV_DIR}"
    echo "  Run:  ./setup_server.sh"
    exit 1
fi

if ! command -v podman-compose &>/dev/null && ! command -v podman &>/dev/null; then
    echo "ERROR: podman/podman-compose not found."
    exit 1
fi

# ─── Warn if Podman storage is on a network filesystem ──────────────────────
GRAPHROOT=$(podman info --format '{{.Store.GraphRoot}}' 2>/dev/null || true)
if [[ "$GRAPHROOT" == /home/* || "$GRAPHROOT" == /lvs* || "$GRAPHROOT" == /hs/* ]]; then
    echo "WARNING: Podman graphroot ($GRAPHROOT) appears to be on a network filesystem."
    echo "  Image pulls/builds may fail with 'lsetxattr: operation not supported'."
    echo "  See README.md Step 0a for how to redirect storage to /var/tmp."
    echo ""
fi

# ─── .env setup ──────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    echo "No .env found — copying env.example to .env"
    echo "*** Edit .env to adjust passwords or local overrides as needed. ***"
    cp env.example .env
fi

# shellcheck disable=SC1091
set -o allexport
source .env
set +o allexport

if [[ -z "${UMRS_DATA_PORT_RANGE:-}" && -z "${UMRS_DATA_PORT:-}" ]]; then
    data_base_port="${UMRS_DATA_BASE_PORT:-5556}"
    data_port_count="${UMRS_DATA_PORT_COUNT:-8}"
    data_last_port=$((data_base_port + data_port_count - 1))
    export UMRS_DATA_PORT_RANGE="${data_base_port}-${data_last_port}"
fi

export UMRS_CONFIG="${UMRS_CONFIG:-${SCRIPT_DIR}}"

RAG_ENABLED=false
if is_truthy "${UMRS_RAG_ENABLE:-false}"; then
    RAG_ENABLED=true
fi

DEFAULT_UMRS_SITE_CONFIG="${HOME}/.config/umrs/site.yaml"
if [[ ! -e "$DEFAULT_UMRS_SITE_CONFIG" ]]; then
    DEFAULT_UMRS_SITE_CONFIG="${SCRIPT_DIR}/site.yaml"
fi
export UMRS_SITE_CONFIG="${UMRS_SITE_CONFIG:-${DEFAULT_UMRS_SITE_CONFIG}}"
if [[ ! -f "$UMRS_SITE_CONFIG" ]]; then
    echo "ERROR: Site config not found at ${UMRS_SITE_CONFIG}"
    echo "  Run: ./setup_server.sh  or set UMRS_SITE_CONFIG explicitly"
    exit 1
fi

# Extend the model registry allowlist for the concrete RCCS home layout.
# Shared defaults assume /home/<user>/models/*, but RCCS uses
# /home/users/<user>/models/*. Keep the shared defaults intact and append the
# resolved HOME-based path for this deployment only.
DEFAULT_MODEL_ALLOWED_PATHS="/shared/models/system/*:/shared/projects/*/models/*:/shared/users/*/models/*:/models/*:/home/*/models/*"
if [[ -n "${HOME:-}" ]]; then
    DEFAULT_MODEL_ALLOWED_PATHS+="${DEFAULT_MODEL_ALLOWED_PATHS:+:}${HOME%/}/models/*"
fi
# Auto-append the site.yaml model root (Lustre) so model registration
# can use paths resolved from the site config without manual overrides.
_site_model_root=$(resolve_model_layout | head -1)
if [[ -n "${_site_model_root:-}" && -d "$_site_model_root" ]]; then
    DEFAULT_MODEL_ALLOWED_PATHS+="${DEFAULT_MODEL_ALLOWED_PATHS:+:}${_site_model_root%/}/*"
fi
export MODEL_ALLOWED_PATHS="${MODEL_ALLOWED_PATHS:-${DEFAULT_MODEL_ALLOWED_PATHS}}"

# ─── Phase 2l invariant: bind ZMQ on 127.0.0.1 only ─────────────────────────
# RCCS-cloud exposes ZMQ to clients via SSH tunnels (Tier B); binding on
# 0.0.0.0 would let any login-node user hit the data plane. Keep tunnel mode
# on by default but allow operators to override via env when debugging.
export UMRS_TUNNEL_MODE="${UMRS_TUNNEL_MODE:-true}"

DEFAULT_RIVAULT_SLURM_SCRIPT="${SCRIPT_DIR}/serve-model.slurm"
if [[ ! -e "$DEFAULT_RIVAULT_SLURM_SCRIPT" ]]; then
    DEFAULT_RIVAULT_SLURM_SCRIPT="${HOME}/.config/umrs/rivault_data/serve-model.slurm"
fi

# ─── Create model cache directories (skip if path is not writable) ──────────
{
    IFS= read -r MODEL_DIR
    IFS= read -r HF_CACHE_DIR
    IFS= read -r SENTENCE_TRANSFORMERS_DIR
} < <(resolve_model_layout)
if mkdir -p "${SENTENCE_TRANSFORMERS_DIR}" "${HF_CACHE_DIR}" 2>/dev/null; then
    echo "Model cache root: ${MODEL_DIR}"
    echo "HF cache: ${HF_CACHE_DIR}"
else
    echo "WARNING: Could not create model cache directories under '${MODEL_DIR}' — ensure they exist."
fi

# ─── Start infrastructure containers ─────────────────────────────────────────
if [[ "$START_MODE" != "--server" ]]; then
    reconcile_infra_container umrs-postgres 5432
    reconcile_infra_container umrs-minio 9000 9001

    echo "Starting infrastructure containers (postgres, minio)..."
    podman compose up -d

    # Wait for postgres to accept connections
    echo -n "Waiting for postgres..."
    for i in $(seq 1 30); do
        if podman exec umrs-postgres pg_isready -h localhost -U "${POSTGRES_USER:-umrs}" >/dev/null 2>&1; then
            echo " ready."
            break
        fi
        echo -n "."
        sleep 1
        [[ "$i" -eq 30 ]] && echo " WARNING: postgres not ready after 30s."
    done

    # Wait for minio health
    echo -n "Waiting for minio..."
    for i in $(seq 1 30); do
        if curl -sf http://127.0.0.1:9000/minio/health/live >/dev/null 2>&1; then
            echo " ready."
            break
        fi
        echo -n "."
        sleep 1
        [[ "$i" -eq 30 ]] && echo " WARNING: minio not ready after 30s."
    done
fi

# ─── Stop existing native processes if running ───────────────────────────────
stop_managed_process "$PID_FILE" "server"
stop_managed_process "$RAG_PID_FILE" "RAG service"

# ─── Kill any process still holding the ZMQ port ────────────────────────────
ZMQ_PORT="${ZMQ_PORT:-5555}"
BLOCKING_PID=$(ss -tlnp "sport = :${ZMQ_PORT}" 2>/dev/null \
    | awk -F'pid=' '/pid=/{print $2}' | awk -F',' '{print $1}' | head -1)
if [[ -n "${BLOCKING_PID:-}" ]]; then
    echo "WARNING: Port ${ZMQ_PORT} is still held by PID ${BLOCKING_PID} — killing it."
    kill "$BLOCKING_PID" 2>/dev/null || true
    sleep 2
    if kill -0 "$BLOCKING_PID" 2>/dev/null; then
        kill -9 "$BLOCKING_PID" 2>/dev/null || true
        sleep 1
    fi
fi

if [[ "$RAG_ENABLED" = true ]]; then
    RAG_BLOCKING_PID=$(ss -tlnp "sport = :8081" 2>/dev/null \
        | awk -F'pid=' '/pid=/{print $2}' | awk -F',' '{print $1}' | head -1)
    if [[ -n "${RAG_BLOCKING_PID:-}" ]]; then
        echo "WARNING: Port 8081 is still held by PID ${RAG_BLOCKING_PID} — killing it."
        kill "$RAG_BLOCKING_PID" 2>/dev/null || true
        sleep 2
        if kill -0 "$RAG_BLOCKING_PID" 2>/dev/null; then
            kill -9 "$RAG_BLOCKING_PID" 2>/dev/null || true
            sleep 1
        fi
    fi
fi

# ─── Environment for the native server ───────────────────────────────────────
# Ensure ~/.local/bin is in PATH (mc, s3cmd and other user-installed tools).
export PATH="${HOME}/.local/bin:${PATH}"

# Database — containers publish postgres on localhost:5432
export DB_HOST="127.0.0.1"
export DB_PORT="${DB_PORT:-5432}"
export DB_SUPERUSER="${POSTGRES_USER:-umrs}"
export DB_SUPERUSER_PASSWORD="${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD in .env}"

if [[ -z "${UMRS_REGISTRY_POSTGRES_DSN:-}" ]]; then
    UMRS_REGISTRY_POSTGRES_DSN="$(
        "${VENV_DIR}/bin/python" - <<'PY'
import os
from urllib.parse import quote

user = os.environ.get("DB_SUPERUSER") or os.environ.get("POSTGRES_USER") or "umrs"
password = os.environ.get("DB_SUPERUSER_PASSWORD") or os.environ.get("POSTGRES_PASSWORD") or ""
host = os.environ.get("DB_HOST") or "127.0.0.1"
port = os.environ.get("DB_PORT") or "5432"
database = os.environ.get("POSTGRES_DB") or "umrs_db"

auth = quote(user, safe="")
if password:
    auth += f":{quote(password, safe='')}"
auth += "@"

print(f"postgres://{auth}{host}:{port}/{database}")
PY
    )"
    export UMRS_REGISTRY_POSTGRES_DSN
fi

if [[ "$START_MODE" != "--server" ]]; then
    echo -n "Validating postgres catalog..."
    postgres_catalog_check_output=""
    if ! postgres_catalog_check_output="$(check_postgres_catalog_ready 2>&1)"; then
        echo " failed."
        echo "ERROR: PostgreSQL is listening, but a system catalog query failed."
        echo "  This usually means the persistent pg_data volume is corrupted or incomplete."
        if [[ -n "$postgres_catalog_check_output" ]]; then
            echo "  Query error:"
            printf '%s\n' "$postgres_catalog_check_output"
        fi
        echo "  Recovery: ./stop.sh --postgres-clean && ./start.sh"
        echo "  Full reset: ./stop.sh --clean && ./start.sh"
        echo "  Warning: --postgres-clean removes only postgres state; --clean also removes minio data."
        echo "  Inspect postgres logs with: podman logs --tail 40 umrs-postgres"
        exit 1
    fi
    echo " ready."
fi

# S3 presigned URL endpoint — compute nodes access MinIO via SSH tunnel
# (localhost:9000 on compute → login node MinIO).
export S3_PRESIGN_ENDPOINT="${S3_PRESIGN_ENDPOINT:-http://localhost:9000}"

# BentoML vision runs inside a container on the compute node, so presigned
# URLs for vision tasks must use the host-gateway alias instead of localhost.
export VISION_S3_PRESIGN_ENDPOINT="${VISION_S3_PRESIGN_ENDPOINT:-http://minio-host:19000}"
export VISION_REQUIRE_PRESIGNED_URLS="${VISION_REQUIRE_PRESIGNED_URLS:-true}"

# Model cache on Lustre — shared between login node and compute nodes.
# Compute nodes mount the same Lustre path, so models downloaded here are
# immediately available to GPU containers (vLLM, BentoML).  These env vars
# also enable the optional CPU-only fallback (sentence-transformers) on the
# login node if model_type is set to TransformerEncoder/SentenceTransformer.
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export UMRS_MODEL_STAGING_ROOT="${MODEL_DIR}"

# Slurm wrapper script — site-specific (sets RIVAULT_DIR, model paths, profiles).
# Override with RIVAULT_SLURM_SCRIPT in .env if your wrapper is elsewhere.
export RIVAULT_SLURM_SCRIPT="${RIVAULT_SLURM_SCRIPT:-${DEFAULT_RIVAULT_SLURM_SCRIPT}}"

# Python environment
export PYTHONUNBUFFERED=1

# ─── Ensure admin s3cfg is present (restore from deploy dir if lost) ─────────
# $HOME/.config/umrs/s3cfg is created by setup_server.sh and is required by
# the server to provision per-user MinIO credentials.  It lives in the HOME
# filesystem which can be wiped/remounted independently of the Lustre deploy
# directory.  Auto-restore to avoid a cryptic FileNotFoundError mid-boot.
if [[ ! -f "${HOME}/.config/umrs/s3cfg" ]]; then
    if [[ -f "${SCRIPT_DIR}/s3cfg" ]]; then
        echo "NOTE: Admin s3cfg missing — restoring from ${SCRIPT_DIR}/s3cfg"
        mkdir -p "${HOME}/.config/umrs"
        cp "${SCRIPT_DIR}/s3cfg" "${HOME}/.config/umrs/s3cfg"
    else
        echo "ERROR: Admin s3cfg not found at ${HOME}/.config/umrs/s3cfg"
        echo "  or at ${SCRIPT_DIR}/s3cfg."
        echo "  Run: ./setup_server.sh  to recreate it."
        exit 1
    fi
fi

# ─── Remove stale per-user s3cfg files ───────────────────────────────────────
# The server regenerates these from the admin .s3cfg on startup.
# Stale files may point to wrong host (e.g. minio:9000 from a previous
# Docker-based deployment).
rm -f "${HOME}/.config/umrs/s3cfg_"* 2>/dev/null || true
rm -f "${HOME}/.s3cfg_user"* 2>/dev/null || true  # legacy path

# ─── Start the native UMRS server ───────────────────────────────────────────
mkdir -p "$LOG_DIR"

echo "Starting umrs-server (native) ..."
echo "  Config:    ${SCRIPT_DIR}/conf_umrs.json"
echo "  Site:      ${UMRS_SITE_CONFIG}"
echo "  Log:       ${LOG_FILE}"
echo "  PID file:  ${PID_FILE}"
echo "  Slurm:     ${RIVAULT_SLURM_SCRIPT}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

LOG_START_SIZE=0
if [[ -f "$LOG_FILE" ]]; then
    LOG_START_SIZE=$(wc -c < "$LOG_FILE")
fi

nohup python "${REPO_ROOT}/src/server_umrs.py" \
    -c "${SCRIPT_DIR}/conf_umrs.json" \
    >> "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# Verify the server finished bootstrap before starting dependent services.
echo -n "Waiting for UMRS server bootstrap..."
for i in $(seq 1 60); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo " failed (PID ${SERVER_PID} exited) — check ${LOG_FILE}"
        show_recent_log_tail "$LOG_FILE"
        exit 1
    fi
    if check_zmq_ready && check_server_bootstrap_complete "$LOG_START_SIZE"; then
        echo " ready (PID $SERVER_PID)."
        break
    fi
    echo -n "."
    sleep 0.5
    if [[ "$i" -eq 60 ]]; then
        echo " NOTE: server bootstrap not complete yet — check ${LOG_FILE}"
        show_recent_log_tail "$LOG_FILE"
    fi
done

if [[ "$RAG_ENABLED" = true ]]; then
    echo "Starting rag-service (native) ..."
    echo "  Log:       ${RAG_LOG_FILE}"
    echo "  PID file:  ${RAG_PID_FILE}"

    OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        nohup python -m umrs.api.rag_service \
        >> "$RAG_LOG_FILE" 2>&1 &

    RAG_PID=$!
    echo "$RAG_PID" > "$RAG_PID_FILE"

    echo -n "Waiting for RAG service readiness..."
    for i in $(seq 1 120); do
        if check_rag_ready; then
            echo " ready (PID $RAG_PID)."
            break
        fi
        echo -n "."
        sleep 1
        [[ "$i" -eq 120 ]] && echo " NOTE: RAG service not ready yet — check ${RAG_LOG_FILE}"
    done
fi

echo ""
echo "=== UMRS stack running ==="
echo "  postgres:    localhost:5432      (container)"
echo "  minio:       localhost:9000      (container, console: localhost:9001)"
echo "  umrs-server: localhost:5555      (native, PID $SERVER_PID)"
echo "  umrs-http:   localhost:${UMRS_HTTP_PORT:-8082}      (sidecar for endpoint discovery)"
if [[ "$RAG_ENABLED" = true ]]; then
    echo "  rag-service: localhost:8081      (native, PID $RAG_PID)"
else
    echo "  rag-service: disabled            (set UMRS_RAG_ENABLE=true in .env to enable)"
fi
echo ""
echo "  Presigned URL endpoint: ${S3_PRESIGN_ENDPOINT}"
echo "  Model root:  ${MODEL_DIR}"
echo "  HF cache:    ${HF_CACHE_DIR}"
echo ""
echo "Check logs:  tail -f ${LOG_FILE}"
if [[ "$RAG_ENABLED" = true ]]; then
    echo "Check RAG:   tail -f ${RAG_LOG_FILE}"
fi
echo "Stop:        ./stop.sh"
