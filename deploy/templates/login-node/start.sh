#!/usr/bin/env bash
# =============================================================================
# start.sh — bring up the login-node UMRS deployment (TEMPLATE)
# =============================================================================
# 1. Start infra containers (postgres, minio) via podman compose.
# 2. Launch the UMRS server NATIVELY so it can call the Slurm CLI.
#
#   ./start.sh            # infra + native server
#   ./start.sh --server   # restart only the native server (keep containers)
#
# This is a minimal reference. The production ../../rccs-cloud/start.sh adds
# stale-port reaping, readiness probing, and catalog checks — layer those in as
# your deployment hardens.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

export GOMAXPROCS="${GOMAXPROCS:-4}"

SCRIPT_DIR="$(pwd)"
REPO_ROOT="$(cd ../../.. && pwd)"        # repo root (…/umrs)
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
PID_FILE="${SCRIPT_DIR}/server.pid"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/server.log"
START_MODE="${1:-}"

[[ "$(uname -s)" == "Linux" ]] || { echo "ERROR: run this on the login node (Linux)."; exit 1; }

# ─── .env ────────────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    echo "No .env found — copying env.example to .env; edit secrets before rerun."
    cp env.example .env
fi
set -o allexport; source .env; set +o allexport

export UMRS_SITE_CONFIG="${UMRS_SITE_CONFIG:-${SCRIPT_DIR}/site.yaml}"
[[ -f "$UMRS_SITE_CONFIG" ]] || { echo "ERROR: site config not found: $UMRS_SITE_CONFIG"; exit 1; }
mkdir -p "$LOG_DIR"

# ─── Infra containers ────────────────────────────────────────────────────────
if [[ "$START_MODE" != "--server" ]]; then
    echo "Starting infra containers (postgres, minio)..."
    podman compose -f "${SCRIPT_DIR}/docker-compose.yaml" up -d
fi

# ─── Native UMRS server ──────────────────────────────────────────────────────
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Stopping existing UMRS server (PID $(cat "$PID_FILE"))..."
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    sleep 2
fi

[[ -x "${VENV_DIR}/bin/python" ]] || { echo "ERROR: venv missing at ${VENV_DIR}. Create it and install UMRS."; exit 1; }

echo "Launching native UMRS server..."
PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
UMRS_SITE_CONFIG="$UMRS_SITE_CONFIG" \
    nohup "${VENV_DIR}/bin/python" -m umrs.server \
    >>"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "UMRS server started (PID $(cat "$PID_FILE")). Logs: $LOG_FILE"
