#!/usr/bin/env bash
set -euo pipefail

# Demo 08: Model Registration & Staging
# Shows: model register, model list, model download via umrs-client

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT="umrs-client"
SERVER_IP="${UMRS_SERVER_IP:-127.0.0.1}"

echo "=== Demo 08: Model Registration & Staging ==="
echo ""

# Step 0: Check server connectivity
echo "--- Step 0: Ping server ---"
${CLIENT} --server-ip "${SERVER_IP}" ping
echo ""

# Step 1: List models (initially empty or pre-existing)
echo "--- Step 1: List registered models ---"
${CLIENT} --server-ip "${SERVER_IP}" model list
echo ""

# Step 2: Register a local model (use an existing directory if available)
MODEL_DIR="${SCRIPT_DIR}/../../src/umrs/rivault/models"
if [ -d "${MODEL_DIR}" ]; then
    # Find the first model directory
    FIRST_MODEL=$(find "${MODEL_DIR}" -mindepth 1 -maxdepth 1 -type d | head -1 || true)
    if [ -n "${FIRST_MODEL}" ]; then
        MODEL_NAME=$(basename "${FIRST_MODEL}")
        echo "--- Step 2: Register local model '${MODEL_NAME}' ---"
        ${CLIENT} --server-ip "${SERVER_IP}" model register \
            "${MODEL_NAME}" "${FIRST_MODEL}" \
            --type embedding --scope system \
            --description "Auto-registered from rivault models directory"
        echo ""
    else
        echo "--- Step 2: No local models found in ${MODEL_DIR}, skipping register ---"
        echo ""
    fi
else
    echo "--- Step 2: Models directory not found (${MODEL_DIR}), skipping register ---"
    echo ""
fi

# Step 3: List models again to verify registration
echo "--- Step 3: List models after registration ---"
${CLIENT} --server-ip "${SERVER_IP}" model list
echo ""

# Step 4: Download a model from HuggingFace (optional, requires HF access)
if [ "${DEMO_DOWNLOAD:-0}" = "1" ]; then
    echo "--- Step 4a: Download model (client-side, default) ---"
    ${CLIENT} --server-ip "${SERVER_IP}" model download intfloat/e5-base-v2
    echo ""

    # Uncomment to test server-side download instead:
    # echo "--- Step 4b: Download model (server-side) ---"
    # ${CLIENT} --server-ip "${SERVER_IP}" model download intfloat/e5-base-v2 --server-side
    # echo ""

    echo "--- Step 5: List models after download ---"
    ${CLIENT} --server-ip "${SERVER_IP}" model list
    echo ""
else
    echo "--- Step 4: Skipping HuggingFace download (set DEMO_DOWNLOAD=1 to enable) ---"
    echo ""
fi

# Step 6: Scope filtering
echo "--- Step 6: List models by scope ---"
echo "  System models:"
${CLIENT} --server-ip "${SERVER_IP}" model list --scope system
echo "  User models:"
${CLIENT} --server-ip "${SERVER_IP}" model list --scope user
echo ""

echo "=== Demo 08 Complete ==="
