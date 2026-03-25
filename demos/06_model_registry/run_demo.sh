#!/usr/bin/env bash
set -euo pipefail

# Demo 06: Model Registration & Staging
# Shows: model register, model list, model download via umrs-client
#
# Runs umrs-client inside the umrs-client container (correct config + server
# connectivity). For model register, uses /shared/models/ paths (visible to
# the UMRS server inside ai-login).

UMRS_CLIENT_CONTAINER="${UMRS_CLIENT_CONTAINER:-umrs-client}"
SERVER_CONTAINER="${UMRS_SERVER_CONTAINER:-ai-login}"

run_client() {
    podman exec "$UMRS_CLIENT_CONTAINER" umrs-client "$@"
}

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
# The server (ai-login) sees models at /shared/models/. We pick the first one
# that is not already registered.
SHARED_MODELS="/shared/models"
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

# Step 3: List models again to verify registration
echo "--- Step 3: List models after registration ---"
run_client model list
echo ""

# Step 4: Download a model from HuggingFace (optional, requires HF access)
if [ "${DEMO_DOWNLOAD:-0}" = "1" ]; then
    echo "--- Step 4a: Download model (client-side, default) ---"
    run_client model download intfloat/e5-base-v2
    echo ""

    echo "--- Step 5: List models after download ---"
    run_client model list
    echo ""
else
    echo "--- Step 4: Skipping HuggingFace download (set DEMO_DOWNLOAD=1 to enable) ---"
    echo ""
fi

# Step 6: Scope filtering
echo "--- Step 6: List models by scope ---"
echo "  System models:"
run_client model list --scope system
echo "  User models:"
run_client model list --scope user
echo ""

echo "=== Demo 06 Complete ==="
