#!/usr/bin/env bash
# =============================================================================
# One-time setup for UMRS on umrs-front (data0.ai.r-ccs.riken.jp)
# =============================================================================
# Creates Python 3.11 venv, installs deps, and prepares for dual-cluster
# deployment (rccs-cloud + AI4S).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"
REPO_ROOT="$(cd ../.. && pwd)"
SRC_DIR="${REPO_ROOT}/src"
VENV_DIR="${SCRIPT_DIR}/.venv"

export PATH="$HOME/.local/bin:$PATH"

log() { echo "[setup] $*"; }
die() { echo "[setup] ERROR: $*" >&2; exit 1; }

# Guard: must be on Linux
[[ "$(uname -s)" == "Linux" ]] || die "Must run on umrs-front (Linux)"

# Check uv
command -v uv &>/dev/null || die "uv not found — install: curl -LsSf https://astral.sh/uv/install.sh | sh"

# Create venv with Python 3.11
if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating venv with Python 3.11..."
    uv venv --python 3.11 "$VENV_DIR"
fi

log "Activating venv..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Install UMRS dependencies
log "Installing UMRS server dependencies..."
uv pip install -e "${SRC_DIR}[all]"

# Also install the gateway requirements if needed
GATEWAY_REQ="${REPO_ROOT}/../gateway/requirements.txt"
if [[ -f "$GATEWAY_REQ" ]]; then
    log "Installing Gateway dependencies..."
    uv pip install -r "$GATEWAY_REQ"
fi

# Create config directories
mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/data"

# Create .env if missing
if [[ ! -f "${SCRIPT_DIR}/.env" ]]; then
    log "Creating .env from template..."
    cat > "${SCRIPT_DIR}/.env" <<'ENVEOF'
# ─────────────────────────────────────────────────────────────
# UMRS umrs-front deployment — dual-cluster environment
# ─────────────────────────────────────────────────────────────

# PostgreSQL (shared for both stacks via separate databases)
POSTGRES_USER=umrs
POSTGRES_PASSWORD=CHANGEME_POSTGRES_PASSWORD
POSTGRES_DB_RCCS=front_rccs
POSTGRES_DB_AI4S=front_ai4s

# MinIO (shared instance, buckets per cluster)
MINIO_ROOT_USER=CHANGEME_MINIO_USER
MINIO_ROOT_PASSWORD=CHANGEME_MINIO_PASSWORD
# Set to 127.0.0.1 for tunnel-only deployments; 0.0.0.0 requires host firewalling.
MINIO_BIND=0.0.0.0

# LiteLLM (auth key for endpoints)
LITELLM_MASTER_KEY=sk-ll-CHANGEME

# UMRS Admin token
UMRS_ADMIN_TOKEN=CHANGEME_ADMIN_TOKEN

# Ports
UMRS_RCCS_PORT=8000
UMRS_AI4S_PORT=8002
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001
POSTGRES_PORT=5432

# Network / Tunnel mode
# Set UMRS_TUNNEL_MODE=true to bind ZMQ to 127.0.0.1 (SSH tunnel required).
# Set UMRS_TUNNEL_MODE=false (default) for direct network access (no SSH tunnel).
UMRS_TUNNEL_MODE=false
# External IP/hostname used by compute nodes and clients to reach services.
# Only used when UMRS_TUNNEL_MODE=false.
UMRS_SERVER_HOST=0.0.0.0
# S3 presigned URL endpoint — use the external IP when not tunneling so
# compute nodes can reach MinIO directly.
# S3_PRESIGN_ENDPOINT=http://134.160.189.136:9000
ENVEOF
    log "*** Edit .env to set proper secrets before starting ***"
fi

log ""
log "Setup complete!"
log "  Venv:     ${VENV_DIR}"
log "  Python:   $(python --version)"
log ""
log "Next steps:"
log "  1. Edit .env with proper secrets"
log "  2. ./start.sh"
