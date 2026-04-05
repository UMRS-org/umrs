#!/usr/bin/env bash
# =============================================================================
# Demo 07: Model Registration & Staging
# =============================================================================
# Demonstrates the model registry workflow:
#   1. List current models
#   2. Register an existing model from a shared filesystem path
#   3. Upload a test model via model upload
#   4. Verify the registry
#   5. Scope filtering
#
# On HPC systems the shared filesystem is the transport layer.
# No internet access is needed on compute nodes — models are pre-staged
# and registered via the UMRS control plane.
#
# Prerequisites:
#   - SSH tunnels active (ZMQ 5555/5556)
#   - UMRS server running
#
# Optional env vars:
#   HMAE_LOCAL_MODEL_DIR          — local path to HMAE checkpoint
#   UMRS_REMOTE_HMAE_MODEL_PATH  — server-side path for registration
#   DEMO_DOWNLOAD=1               — download a model from HuggingFace
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="model_reg" source "${SCRIPT_DIR}/../lib/common.sh"

# Server-side path where models are stored
STAGING_ROOT="${UMRS_MODEL_STAGING_ROOT:-/shared/models}"
REMOTE_HMAE_MODEL_PATH="${UMRS_REMOTE_HMAE_MODEL_PATH:-/shared/models/hmae}"

banner "Demo 07: Model Registration & Staging"
echo ""

# Step 0: Check server connectivity
log "Step 0: Ping server..."
preflight
echo ""

# Step 1: List models (initially empty or pre-existing)
log "Step 1: List registered models..."
$CLIENT model list || true
echo ""

# Step 2: Register a model from a known server-side path (if available)
log "Step 2: Register model from server-side path..."
SHARED_MODEL_PATH="${UMRS_SHARED_MODEL_PATH:-}"
if [[ -n "$SHARED_MODEL_PATH" && -d "$SHARED_MODEL_PATH" ]]; then
    MODEL_NAME=$(basename "$SHARED_MODEL_PATH")
    log "  Registering '${MODEL_NAME}' at ${SHARED_MODEL_PATH}..."
    $CLIENT model register "${MODEL_NAME}" "${SHARED_MODEL_PATH}" \
        --type embedding --scope system \
        --description "Auto-registered from shared models directory" || true
else
    log "  Skipping — set UMRS_SHARED_MODEL_PATH to a server-side model dir"
fi
echo ""

# Step 3a: Upload a real HMAE model (if available locally)
HMAE_LOCAL_MODEL_DIR="${HMAE_LOCAL_MODEL_DIR:-}"
if [[ -n "$HMAE_LOCAL_MODEL_DIR" ]] && [[ -d "$HMAE_LOCAL_MODEL_DIR" ]]; then
    log "Step 3a: Uploading real HMAE model via 'model upload'..."
    log "  Source (local):       $HMAE_LOCAL_MODEL_DIR"
    log "  Register path (HPC): $REMOTE_HMAE_MODEL_PATH"
    $CLIENT model upload "$HMAE_LOCAL_MODEL_DIR" \
        --name "hmae-pretrained" \
        --type embedding --scope system \
        --description "HMAE ViT pretrained checkpoint" \
        --register-path "$REMOTE_HMAE_MODEL_PATH" \
        --force
    ok "HMAE model upload complete"
else
    log "Step 3a: Skipping HMAE upload (set HMAE_LOCAL_MODEL_DIR to enable)"
fi
echo ""

# Step 3b: Upload a small test model (created locally)
log "Step 3b: Uploading a test model via 'model upload'..."
TEST_MODEL_DIR="${SCRIPT_DIR}/.test_model"
rm -rf "$TEST_MODEL_DIR"
mkdir -p "$TEST_MODEL_DIR"
echo '{"model_type": "test", "framework": "demo"}' > "$TEST_MODEL_DIR/config.json"
echo "dummy weights" > "$TEST_MODEL_DIR/model.bin"

$CLIENT model upload "$TEST_MODEL_DIR" \
    --name "test-upload-model" \
    --type embedding --scope user \
    --description "Demo model uploaded via model upload" \
    --force
ok "Test model upload complete"
rm -rf "$TEST_MODEL_DIR"
echo ""

# Step 4: List models again to verify
log "Step 4: Verify — list all registered models..."
$CLIENT model list
echo ""

# Step 5: Scope filtering
log "Step 5: List models by scope..."
echo "  System models:"
$CLIENT model list --scope system || true
echo "  User models:"
$CLIENT model list --scope user || true
echo ""

# Step 6: Download a model from HuggingFace (optional, requires HF access)
if [ "${DEMO_DOWNLOAD:-0}" = "1" ]; then
    log "Step 6: Download model (server-side)..."
    $CLIENT model download intfloat/e5-base-v2 --scope system --server-side
    echo ""
    log "Models after download:"
    $CLIENT model list
    echo ""
else
    log "Step 6: Skipping HuggingFace download (set DEMO_DOWNLOAD=1 to enable)"
fi

echo ""
banner "Demo 07 Complete"
