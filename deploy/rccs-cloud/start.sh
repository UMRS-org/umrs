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

SCRIPT_DIR="$(pwd)"
REPO_ROOT="$(cd ../.. && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PID_FILE="${SCRIPT_DIR}/server.pid"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/server.log"

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
    echo "*** Edit .env to adjust MODEL_DIR, passwords, etc. ***"
    cp env.example .env
fi

# shellcheck disable=SC1091
set -o allexport
source .env
set +o allexport

# ─── Create model directory (skip if path is not writable) ───────────────────
MODEL_DIR="${MODEL_DIR:-/lvs0/rccs-dmp/mnagaso/umrs_models}"
if mkdir -p "${MODEL_DIR}/sentence_transformers" "${MODEL_DIR}/hf" 2>/dev/null; then
    echo "Model cache: ${MODEL_DIR}"
else
    echo "WARNING: Could not create MODEL_DIR '${MODEL_DIR}' — ensure it exists."
fi

# ─── Start infrastructure containers ─────────────────────────────────────────
if [[ "${1:-}" != "--server" ]]; then
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

# ─── Stop existing server if running ─────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing server (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
        kill -0 "$OLD_PID" 2>/dev/null && kill -9 "$OLD_PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
fi

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

# ─── Environment for the native server ───────────────────────────────────────
# Ensure ~/.local/bin is in PATH (mc, s3cmd and other user-installed tools).
export PATH="${HOME}/.local/bin:${PATH}"

# Database — containers publish postgres on localhost:5432
export DB_HOST="127.0.0.1"
export DB_SUPERUSER="${POSTGRES_USER:-umrs}"
export DB_SUPERUSER_PASSWORD="${POSTGRES_PASSWORD:-umrs_rccs}"

# S3 presigned URL endpoint — compute nodes access MinIO via SSH tunnel
# (localhost:9000 on compute → login node MinIO).
export S3_PRESIGN_ENDPOINT="${S3_PRESIGN_ENDPOINT:-http://localhost:9000}"

# Model cache on Lustre — shared between login node and compute nodes.
# Compute nodes mount the same Lustre path, so models downloaded here are
# immediately available to GPU containers (vLLM, BentoML).  These env vars
# also enable the optional CPU-only fallback (sentence-transformers) on the
# login node if model_type is set to TransformerEncoder/SentenceTransformer.
export SENTENCE_TRANSFORMERS_HOME="${MODEL_DIR}/sentence_transformers"
export HF_HOME="${MODEL_DIR}/hf"
export UMRS_MODEL_STAGING_ROOT="${MODEL_DIR}"

# Slurm wrapper script — site-specific (sets RIVAULT_DIR, model paths, profiles).
# Override with RIVAULT_SLURM_SCRIPT in .env if your wrapper is elsewhere.
export RIVAULT_SLURM_SCRIPT="${RIVAULT_SLURM_SCRIPT:-${SCRIPT_DIR}/serve-model.slurm}"

# Python environment
export PYTHONUNBUFFERED=1

# ─── Remove stale per-user s3cfg files ───────────────────────────────────────
# The server regenerates these from the admin .s3cfg on startup.
# Stale files may point to wrong host (e.g. minio:9000 from a previous
# Docker-based deployment).
rm -f "${HOME}/.s3cfg_user"* 2>/dev/null || true

# ─── Start the native UMRS server ───────────────────────────────────────────
mkdir -p "$LOG_DIR"

echo "Starting umrs-server (native) ..."
echo "  Config:    ${SCRIPT_DIR}/conf_umrs.json"
echo "  Log:       ${LOG_FILE}"
echo "  PID file:  ${PID_FILE}"
echo "  Slurm:     ${RIVAULT_SLURM_SCRIPT}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

nohup python "${REPO_ROOT}/src/server_umrs.py" \
    -c "${SCRIPT_DIR}/conf_umrs.json" \
    >> "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# Verify the server started (wait up to 10s for ZMQ port)
echo -n "Waiting for ZMQ port 5555..."
for i in $(seq 1 20); do
    if nc -z 127.0.0.1 5555 2>/dev/null; then
        echo " ready (PID $SERVER_PID)."
        break
    fi
    echo -n "."
    sleep 0.5
    [[ "$i" -eq 20 ]] && echo " NOTE: port not open yet — check ${LOG_FILE}"
done

echo ""
echo "=== UMRS stack running ==="
echo "  postgres:    localhost:5432      (container)"
echo "  minio:       localhost:9000      (container, console: localhost:9001)"
echo "  umrs-server: localhost:5555      (native, PID $SERVER_PID)"
echo ""
echo "  Presigned URL endpoint: ${S3_PRESIGN_ENDPOINT}"
echo "  Model cache: ${MODEL_DIR}"
echo ""
echo "Check logs:  tail -f ${LOG_FILE}"
echo "Stop:        ./stop.sh"
