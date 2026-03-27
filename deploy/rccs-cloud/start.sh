#!/usr/bin/env bash
# Start the UMRS server stack on rccs-cloud.
#
# Usage (on rccs-cloud):
#   cd deploy/rccs-cloud
#   ./start.sh
#
# This script:
#   1. Copies env.example to .env if .env doesn't exist
#   2. Creates the Lustre model directory if needed
#   3. Starts postgres, minio, and umrs-server via podman compose

set -euo pipefail
cd "$(dirname "$0")"

# ─── Check podman-compose is available ───
if ! command -v podman-compose &>/dev/null; then
    echo "ERROR: podman-compose not found."
    echo "  Install it: pip install --user podman-compose"
    echo "  Then ensure ~/.local/bin is in your PATH."
    exit 1
fi

# ─── Warn if Podman storage is on a network filesystem ───
GRAPHROOT=$(podman info --format '{{.Store.GraphRoot}}' 2>/dev/null || true)
if [[ "$GRAPHROOT" == /home/* || "$GRAPHROOT" == /lvs* || "$GRAPHROOT" == /hs/* ]]; then
    echo "WARNING: Podman graphroot ($GRAPHROOT) appears to be on a network filesystem."
    echo "  Image pulls/builds may fail with 'lsetxattr: operation not supported'."
    echo "  See README.md Step 0 for how to redirect storage to /tmp."
    echo ""
fi

# ─── .env setup ───
if [ ! -f .env ]; then
    echo "No .env found — copying env.example to .env"
    echo "*** Edit .env to adjust S3_PRESIGN_ENDPOINT, MODEL_DIR, etc. ***"
    cp env.example .env
fi

source .env

# ─── Create model directory on Lustre ───
MODEL_DIR="${MODEL_DIR:-/lvs0/rccs-dmp/mnagaso/umrs_models}"
mkdir -p "${MODEL_DIR}/sentence_transformers" "${MODEL_DIR}/hf"
echo "Model cache: ${MODEL_DIR}"

# ─── Start services ───
echo "Starting UMRS services..."
podman compose up -d

echo ""
echo "=== Services started ==="
echo "  postgres:    localhost:5432"
echo "  minio:       localhost:9000 (console: localhost:9001)"
echo "  umrs-server: localhost:5555 (ZMQ), localhost:8000 (HTTP)"
echo ""
echo "Presigned URL endpoint: ${S3_PRESIGN_ENDPOINT:-http://rccs-cloud:9000}"
echo ""
echo "Check logs:  podman compose logs -f umrs-server"
echo "Stop:        ./stop.sh"
