#!/usr/bin/env bash
# Build the UMRS Assistant Service image locally (ADR-012).
#
# Usage:
#   bash umrs/docker/build-assistant.sh
#
# After building, the image is available as:
#   localhost/umrs-assistant:latest
#
# The serve-model.yaml assistant service references this image by default.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UMRS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="localhost/umrs-assistant"
IMAGE_TAG="${ASSISTANT_DOCKER_TAG:-latest}"

echo "[build] Building ${IMAGE_NAME}:${IMAGE_TAG}"
echo "[build] Context: ${UMRS_ROOT}"
echo "[build] Dockerfile: ${SCRIPT_DIR}/Dockerfile.assistant"

cd "${UMRS_ROOT}"

if command -v podman &>/dev/null; then
    podman build \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        -f docker/Dockerfile.assistant \
        .
elif command -v docker &>/dev/null; then
    docker build \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        -f docker/Dockerfile.assistant \
        .
else
    echo "[error] Neither podman nor docker found."
    exit 1
fi

echo "[build] Done: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "[build] Verify with: podman images | grep umrs-assistant"
