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

select_compose_cmd() {
    if command -v podman-compose >/dev/null 2>&1; then
        COMPOSE_CMD=(podman-compose)
        return 0
    fi

    if podman compose version >/dev/null 2>&1; then
        COMPOSE_CMD=(podman compose)
        return 0
    fi

    if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD=(docker compose)
        return 0
    fi

    cat <<'EOF'
Error: no working compose backend found.

Detected:
- podman is installed, but "podman compose" has no provider on this host.

Install one of the following:
1) pip install --user podman-compose
2) install docker compose plugin (if docker is available)

Then rerun: ./start.sh
EOF
    return 1
}

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
select_compose_cmd

echo "Starting UMRS services..."
"${COMPOSE_CMD[@]}" up -d

echo ""
echo "=== Services started ==="
echo "  postgres:    localhost:5432"
echo "  minio:       localhost:9000 (console: localhost:9001)"
echo "  umrs-server: localhost:5555 (ZMQ), localhost:8000 (HTTP)"
echo ""
echo "Presigned URL endpoint: ${S3_PRESIGN_ENDPOINT:-http://rccs-cloud:9000}"
echo ""
echo "Compose cmd: ${COMPOSE_CMD[*]}"
echo "Check logs:  ${COMPOSE_CMD[*]} logs -f umrs-server"
echo "Stop:        ./stop.sh"
