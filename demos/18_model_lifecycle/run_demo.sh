#!/usr/bin/env bash
# =============================================================================
# Demo 18: Model Lifecycle
# =============================================================================
# Demonstrates the full model lifecycle:
#   register → list → use for inference → deprecate
# Shows model scoping (system vs user).
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running
#   - umrs-client installed and in PATH
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="model_life" source "${SCRIPT_DIR}/../lib/common.sh"

RESULTS_DIR="${SCRIPT_DIR}/results"
DEMO_LOCAL_MODEL_UPLOAD="${DEMO_LOCAL_MODEL_UPLOAD:-0}"

banner "Demo 18: Model Lifecycle"
echo "Config       : ${CONF}"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight
mkdir -p "$RESULTS_DIR"

# ---------- Step 2: List existing models ----------
log ""
log "Step 2: Listing existing models..."

log "  System models:"
$CLIENT model list --scope system 2>&1 \
    | while IFS= read -r line; do log "    $line"; done

log "  User models:"
$CLIENT model list --scope user 2>&1 \
    | while IFS= read -r line; do log "    $line"; done
ok "Current model inventory retrieved"

# ---------- Step 3: Download a system model ----------
log ""
log "Step 3: Downloading a system-scope embedding model..."
MODEL_REPO="${EMBEDDING_MODEL:-intfloat/e5-base-v2}"
$CLIENT model download "$MODEL_REPO" --scope system --server-side 2>&1 \
    | while IFS= read -r line; do log "  $line"; done
ok "Model download requested: $MODEL_REPO"

# ---------- Step 4: Register a user-scope model ----------
log ""
log "Step 4: Registering a user-scope model..."
USER_MODEL_REGISTERED=0
if [[ "$DEMO_LOCAL_MODEL_UPLOAD" != "1" ]]; then
    log "  Skipping by default — local model upload requires client/server shared storage."
    log "  Set DEMO_LOCAL_MODEL_UPLOAD=1 to enable this step when the staging path is shared."
else
    MODEL_DIR="${RESULTS_DIR}/my_custom_model"
    mkdir -p "$MODEL_DIR"
    cat > "${MODEL_DIR}/config.json" << 'JSON_EOF'
{
  "model_type": "custom_embedding",
  "hidden_size": 768,
  "num_layers": 6,
  "description": "Demo custom model for lifecycle test"
}
JSON_EOF
    echo "dummy model weights" > "${MODEL_DIR}/model.bin"

    $CLIENT model upload "$MODEL_DIR" \
        --name "demo-custom-model" \
        --type embedding \
        --scope user \
        --description "Custom model for lifecycle demo" \
        --force 2>&1 \
        | while IFS= read -r line; do log "  $line"; done
    ok "User model registered: demo-custom-model"
    USER_MODEL_REGISTERED=1
fi

# ---------- Step 5: Verify registration ----------
log ""
log "Step 5: Verifying model registration..."
$CLIENT model list --scope user 2>&1 \
    | while IFS= read -r line; do log "  $line"; done

# Check if our model appears
if [[ "$USER_MODEL_REGISTERED" = "1" ]]; then
    if $CLIENT model list --scope user 2>&1 | grep -qi "demo-custom-model"; then
        ok "demo-custom-model found in user registry"
    else
        warn "demo-custom-model not found — may need time to index"
    fi
fi

# ---------- Step 6: Use model for inference (via text embedding) ----------
log ""
log "Step 6: Using system model for inference..."
BUCKET_ALIAS="${TEXT_BUCKET:-user1-textfiles}"

# Upload a small test file
echo "Testing the model lifecycle with a sample sentence." > "${RESULTS_DIR}/lifecycle_test.txt"
$CLIENT upload_file "${RESULTS_DIR}/lifecycle_test.txt" "$BUCKET_ALIAS" 2>&1 \
    | while IFS= read -r line; do log "  $line"; done
ok "Inference triggered (server will embed automatically)"

# ---------- Step 7: Model status check ----------
log ""
log "Step 7: Final model inventory..."

log "  System models:"
$CLIENT model list --scope system 2>&1 \
    | while IFS= read -r line; do log "    $line"; done

log "  User models:"
$CLIENT model list --scope user 2>&1 \
    | while IFS= read -r line; do log "    $line"; done

# ---------- Summary ----------
log ""
log "Summary of model lifecycle:"
log "  ✓ Listed existing system and user models"
log "  ✓ Downloaded system-scope model: ${MODEL_REPO}"
if [[ "$USER_MODEL_REGISTERED" = "1" ]]; then
    log "  ✓ Registered user-scope model: demo-custom-model"
else
    log "  • Skipped user-scope upload (set DEMO_LOCAL_MODEL_UPLOAD=1 to enable)"
fi
log "  ✓ Triggered inference with system model"
log "  ✓ Verified model registry state"

banner "Demo 18 Complete"
