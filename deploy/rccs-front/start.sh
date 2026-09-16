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

# SeaweedFS IAM/STS config (ADR-055 fleet wave), rendered from .env.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../_common/render_seaweedfs_iam.sh"
render_seaweedfs_iam "${SCRIPT_DIR}"

# Retire the pre-ADR-055 MinIO container for good (volume kept for operator).
podman rm -f umrs-minio >/dev/null 2>&1 || true

# Start services
echo "Starting UMRS front-end services..."
podman-compose up -d

echo ""
echo "Services started:"
echo "  UMRS API:    http://0.0.0.0:8000"
echo "  Gateway:     http://0.0.0.0:8080"
echo "  SeaweedFS:   http://0.0.0.0:9000 (IAM: http://127.0.0.1:8112, metrics: http://127.0.0.1:9328)"
echo "  PostgreSQL:  127.0.0.1:5432"
echo ""
echo "UMRS will establish SSH ControlMaster to ${SSH_TARGET} on first job submission."
