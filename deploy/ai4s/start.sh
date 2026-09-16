#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Start UMRS AI4S deployment on umrs-front
# ─────────────────────────────────────────────────────────────
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

# Verify SSH connectivity to AI4S login node
AI4S_SSH_TARGET="${AI4S_SSH_TARGET:-134.160.189.24}"
AI4S_SSH_USER="${AI4S_SSH_USER:-ea0022}"
echo "Verifying SSH connectivity to ${AI4S_SSH_USER}@${AI4S_SSH_TARGET}..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${AI4S_SSH_USER}@${AI4S_SSH_TARGET}" "echo ok" >/dev/null 2>&1; then
    echo "ERROR: Cannot SSH to ${AI4S_SSH_USER}@${AI4S_SSH_TARGET}."
    echo "Ensure:"
    echo "  1. SSH key is in ${AI4S_SSH_KEY_PATH:-~/.ssh/ai4s}"
    echo "  2. Key is authorized on AI4S"
    echo "  3. R-CCS VPN is connected (AI4S requires R-CCS network)"
    exit 1
fi
echo "SSH connectivity OK."

# Verify Slurm is accessible
echo "Verifying Slurm access on AI4S..."
if ! ssh -o BatchMode=yes "${AI4S_SSH_USER}@${AI4S_SSH_TARGET}" "sinfo -h -p 1n1gpu -o '%P %a %D'" >/dev/null 2>&1; then
    echo "WARNING: Cannot query Slurm partitions. Jobs may fail."
fi
echo "Slurm access verified."

# SeaweedFS IAM/STS config (ADR-055 fleet wave), rendered from .env.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../_common/render_seaweedfs_iam.sh"
render_seaweedfs_iam "${SCRIPT_DIR}"

# Retire the pre-ADR-055 MinIO container for good (volume kept for operator).
podman rm -f umrs-minio-ai4s >/dev/null 2>&1 || true

# Start services
echo "Starting UMRS AI4S services..."
podman-compose up -d

echo ""
echo "Services started (AI4S target):"
echo "  UMRS API:    http://0.0.0.0:8002"
echo "  Gateway:     http://0.0.0.0:8082"
echo "  SeaweedFS:   http://0.0.0.0:9002 (IAM: http://127.0.0.1:8113, metrics: http://127.0.0.1:9329)"
echo "  PostgreSQL:  127.0.0.1:5433"
echo ""
echo "SSH target: ${AI4S_SSH_USER}@${AI4S_SSH_TARGET}"
echo "UMRS will establish SSH ControlMaster on first job submission."
