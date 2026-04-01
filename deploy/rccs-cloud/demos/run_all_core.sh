#!/usr/bin/env bash
# Run core demos 00-06, 09 sequentially (excludes jens demos 07/08).
# Requires: SSH tunnels active, umrs-client in PATH.
#
# Optional:
#   HMAE_LOCAL_MODEL_DIR=/path/to/checkpoint  — uploads HMAE model before inference
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/../.venv"
export PATH="${VENV}/bin:${PATH}"

echo "=== Pre-flight ==="
which umrs-client || { echo "ERROR: umrs-client not found"; exit 1; }
nc -z localhost 5555 || { echo "ERROR: ZMQ port 5555 not reachable — run ssh_tunnel.sh"; exit 1; }
nc -z localhost 9000 || { echo "ERROR: MinIO port 9000 not reachable — run ssh_tunnel.sh"; exit 1; }
echo "Pre-flight OK"
echo

echo "=== Demo 00: Health Check ==="
bash "${SCRIPT_DIR}/00_health_check.sh" 2>&1
echo
echo "=== Demo 00 DONE ==="
echo

echo "=== Demo 01: Text Embedding ==="
bash "${SCRIPT_DIR}/01_text_embedding.sh" 2>&1
echo
echo "=== Demo 01 DONE ==="
echo

# Demo 02 requires UMRS_RAG_ENABLE=1 on server, skip if RAG not available
echo "=== Demo 02: RAG Search ==="
if curl -sf http://localhost:8081/docs >/dev/null 2>&1; then
    bash "${SCRIPT_DIR}/02_rag_search.sh" 2>&1
    echo "=== Demo 02 DONE ==="
else
    echo "SKIP: RAG service not available (needs UMRS_RAG_ENABLE=1 on server)"
fi
echo

echo "=== Demo 03: SAM3 Segmentation ==="
bash "${SCRIPT_DIR}/03_sam3_segmentation.sh" 2>&1
echo
echo "=== Demo 03 DONE ==="
echo

echo "=== Demo 04: HMAE Inference ==="
bash "${SCRIPT_DIR}/04_hmae_inference.sh" 2>&1
echo
echo "=== Demo 04 DONE ==="
echo

echo "=== Demo 05: User Model ==="
bash "${SCRIPT_DIR}/05_user_model.sh" 2>&1
echo
echo "=== Demo 05 DONE ==="
echo

echo "=== Demo 06: Pipeline SDK ==="
bash "${SCRIPT_DIR}/06_pipeline_sdk.sh" 2>&1
echo
echo "=== Demo 06 DONE ==="
echo

echo "=== Demo 09: Model Registry ==="
bash "${SCRIPT_DIR}/09_model_registry.sh" 2>&1
echo
echo "=== Demo 09 DONE ==="
echo

echo "=== ALL CORE DEMOS COMPLETE ==="
