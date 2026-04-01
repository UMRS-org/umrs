#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Demo 06: Model Registration & Staging
# =============================================================================
# Demonstrates the model registry workflow:
#
#   1. List current models
#   2. Register an existing model from the shared filesystem
#   3. Upload a real HMAE model (optional — set HMAE_LOCAL_MODEL_DIR)
#   3b. Upload a small test model
#   4. Verify the registry
#
# On HPC systems the shared filesystem is the transport layer.
# No internet access is needed on compute nodes — models are pre-staged
# and registered via the UMRS control plane.
#
# Usage:
#   ./run_demo.sh
#   HMAE_LOCAL_MODEL_DIR=~/Downloads/hmae_checkpoint ./run_demo.sh
# =============================================================================

UMRS_CLIENT_CONTAINER="${UMRS_CLIENT_CONTAINER:-umrs-client}"
SERVER_CONTAINER="${UMRS_SERVER_CONTAINER:-ai-login}"

# Server-side path where models are stored
STAGING_ROOT="${UMRS_MODEL_STAGING_ROOT:-/shared/models}"
REMOTE_HMAE_MODEL_PATH="${UMRS_REMOTE_HMAE_MODEL_PATH:-/shared/models/hmae}"

run_client() {
    podman exec "$UMRS_CLIENT_CONTAINER" umrs-client "$@"
}

log()  { echo "[model_reg] $*"; }
ok()   { echo "[model_reg]   ✓ $1"; }
die()  { echo "[model_reg] ERROR: $*" >&2; exit 1; }

echo "=== Demo 06: Model Registration & Staging ==="
echo ""

# Step 0: Check server connectivity
echo "--- Step 0: Ping server ---"
run_client ping
echo ""

# Step 1: List models (initially empty or pre-existing)
echo "--- Step 1: List registered models ---"
run_client model list
echo ""

# Step 2: Register a model from the server's shared model directory
SHARED_MODELS="$STAGING_ROOT"
echo "--- Step 2: Register local model from ${SHARED_MODELS} ---"
FIRST_MODEL=$(podman exec "$SERVER_CONTAINER" \
    find "${SHARED_MODELS}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1 || true)
if [ -n "${FIRST_MODEL}" ]; then
    MODEL_NAME=$(basename "${FIRST_MODEL}")
    echo "  Registering '${MODEL_NAME}' at ${FIRST_MODEL} (server path)..."
    run_client model register \
        "${MODEL_NAME}" "${FIRST_MODEL}" \
        --type embedding --scope system \
        --description "Auto-registered from shared models directory" || true
else
    echo "  No models found in ${SHARED_MODELS}, skipping register"
fi
echo ""

# Step 3a: Upload a real HMAE model (if available)
HMAE_LOCAL_MODEL_DIR="${HMAE_LOCAL_MODEL_DIR:-}"

if [[ -n "$HMAE_LOCAL_MODEL_DIR" ]] && [[ -d "$HMAE_LOCAL_MODEL_DIR" ]] \
   && [[ -f "$HMAE_LOCAL_MODEL_DIR/latest_checkpoint.pth" ]]; then
    echo "--- Step 3a: Upload real HMAE model via 'model upload' ---"
    log "  Source (local):      $HMAE_LOCAL_MODEL_DIR"
    log "  Register path (HPC): $REMOTE_HMAE_MODEL_PATH"

    # Copy model into container, then upload via umrs-client
    CONTAINER_MODEL_DIR="/tmp/hmae_model_upload"
    podman exec "$UMRS_CLIENT_CONTAINER" rm -rf "$CONTAINER_MODEL_DIR"
    podman cp "$HMAE_LOCAL_MODEL_DIR/." "$UMRS_CLIENT_CONTAINER:$CONTAINER_MODEL_DIR"
    run_client model upload "$CONTAINER_MODEL_DIR" \
        --name "hmae-pretrained" \
        --type embedding --scope system \
        --description "HMAE ViT pretrained checkpoint (s8d-1k)" \
        --register-path "$REMOTE_HMAE_MODEL_PATH" \
        --force

    ok "HMAE model upload complete"
else
    if [[ -n "$HMAE_LOCAL_MODEL_DIR" ]]; then
        log "Step 3a: SKIP — HMAE_LOCAL_MODEL_DIR set but not found or missing checkpoint:"
        log "   $HMAE_LOCAL_MODEL_DIR"
    else
        log "Step 3a: SKIP — HMAE_LOCAL_MODEL_DIR not set (set it to upload the HMAE checkpoint)"
    fi
fi
echo ""

# Step 3b: Upload a small test model
echo "--- Step 3b: Upload a test model via 'model upload' ---"

# Create a tiny test model inside the container
CONTAINER_TEST_DIR="/tmp/test_model_upload"
podman exec "$UMRS_CLIENT_CONTAINER" rm -rf "$CONTAINER_TEST_DIR"
podman exec "$UMRS_CLIENT_CONTAINER" mkdir -p "$CONTAINER_TEST_DIR"
podman exec "$UMRS_CLIENT_CONTAINER" bash -c \
    "echo '{\"model_type\": \"test\", \"framework\": \"demo\"}' > $CONTAINER_TEST_DIR/config.json && echo 'dummy weights' > $CONTAINER_TEST_DIR/model.bin"

REMOTE_TEST_PATH="$STAGING_ROOT/test-upload-model"
log "  Register path (server): $REMOTE_TEST_PATH"

run_client model upload "$CONTAINER_TEST_DIR" \
    --name "test-upload-model" \
    --type embedding --scope user \
    --description "Demo model uploaded via model upload" \
    --register-path "$REMOTE_TEST_PATH" \
    --force

ok "test model upload complete"
echo ""

# Step 4: List models again to verify
echo "--- Step 4: Verify — list all registered models ---"
run_client model list

# Step 5: Scope filtering
echo ""
echo "--- Step 5: List models by scope ---"
echo "  System models:"
run_client model list --scope system
echo "  User models:"
run_client model list --scope user
echo ""

# Step 6: Download a model from HuggingFace (optional, requires HF access)
if [ "${DEMO_DOWNLOAD:-0}" = "1" ]; then
    echo "--- Step 6: Download model (server-side) ---"
    run_client model download intfloat/e5-base-v2
    echo ""
    echo "--- Models after download ---"
    run_client model list
    echo ""
else
    echo "--- Step 6: Skipping HuggingFace download (set DEMO_DOWNLOAD=1 to enable) ---"
    echo ""
fi

echo "=== Demo 06 Complete ==="
echo ""
log "The air-gapped HPC model workflow:"
log "  1. Download/prepare models on your local machine"
log "  2. 'umrs-client model upload <path> --register-path <server-path>' copies + registers"
log "  3. Compute nodes access models via the shared filesystem — no internet needed"
log ""
log "  Tip: set HMAE_LOCAL_MODEL_DIR to upload the HMAE checkpoint:"
log "    HMAE_LOCAL_MODEL_DIR=/path/to/hmae_checkpoint ./run_demo.sh"
