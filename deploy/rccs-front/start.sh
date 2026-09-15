#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Start UMRS front-end server deployment
# ─────────────────────────────────────────────────────────────
# Starts all containerized services: Postgres, MinIO, UMRS, Gateway.
# UMRS will establish SSH ControlMaster to rccs-cloud on startup.
#
# Prerequisites:
#   1. Copy env.example to .env and fill in secrets
#   2. Ensure SSH key access to rccs-cloud is configured
#   3. Podman/Docker available
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment
if [[ -f .env ]]; then
    set -o allexport
    source .env
    set +o allexport
else
    echo "ERROR: .env file not found. Copy env.example to .env and configure."
    exit 1
fi

# Verify SSH connectivity to login node before starting services
SSH_TARGET="${SSH_TARGET:-rccs-cloud}"
echo "Verifying SSH connectivity to ${SSH_TARGET}..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${SSH_TARGET}" "echo ok" >/dev/null 2>&1; then
    echo "ERROR: Cannot SSH to ${SSH_TARGET}. Check SSH key and config."
    exit 1
fi
echo "SSH connectivity OK."

# Start services
echo "Starting UMRS front-end services..."
podman-compose up -d

echo ""
echo "Services started:"
echo "  UMRS API:    http://0.0.0.0:8000"
echo "  Gateway:     http://0.0.0.0:8080"
echo "  MinIO:       http://0.0.0.0:9000 (console: http://127.0.0.1:9001)"
echo "  PostgreSQL:  127.0.0.1:5432"
echo ""
echo "UMRS will establish SSH ControlMaster to ${SSH_TARGET} on first job submission."
