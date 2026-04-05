#!/usr/bin/env bash
# =============================================================================
# Demo 02: Text Embedding
# =============================================================================
# Uploads text files → waits for server-side embedding → exports vectors.
#
# Prerequisites:
#   - SSH tunnels active (ZMQ 5555/5556, MinIO 9000)
#   - UMRS server running with Text Embedding experience
#   - umrs-client installed and in PATH
#
# Bucket alias used: user1-textfiles  (from user config)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="text_embed" source "${SCRIPT_DIR}/../lib/common.sh"

BUCKET_ALIAS="${TEXT_BUCKET:-user1-textfiles}"
SAMPLE_DIR="${SCRIPT_DIR}/../sample_data/texts"
RESULTS_DIR="${SCRIPT_DIR}/results"

banner "Demo 02: Text Embedding"
echo "User config  : ${CONF}"
echo "Bucket alias : ${BUCKET_ALIAS}"
echo "Sample data  : ${SAMPLE_DIR}"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

# ---------- Step 2: Prepare embedding model ----------
log ""
log "Step 2: Ensuring embedding model is available..."
MODEL_REPO="${EMBEDDING_MODEL:-intfloat/e5-base-v2}"
$CLIENT model download "$MODEL_REPO" --scope system --server-side 2>&1 \
    | while IFS= read -r line; do log "  $line"; done
ok "Embedding model download requested"

# ---------- Step 3: Upload text data ----------
log ""
log "Step 3: Uploading text files to bucket '${BUCKET_ALIAS}'..."
[[ -d "$SAMPLE_DIR" ]] || die "Sample data not found: $SAMPLE_DIR"
FILE_COUNT=$(find "$SAMPLE_DIR" -maxdepth 1 -type f -name "*.txt" | wc -l)
log "  Found ${FILE_COUNT} text file(s)"
$CLIENT upload_dir "$SAMPLE_DIR" "$BUCKET_ALIAS"
ok "Text files uploaded"

# ---------- Step 4: Wait for embeddings ----------
log ""
log "Step 4: Waiting for embedding tasks (timeout: 10 min)..."
wait_tasks 60 10

# ---------- Step 5: Verify & Export ----------
log ""
log "Step 5: Exporting embeddings to CSV..."
mkdir -p "$RESULTS_DIR"
$CLIENT export_embeddings "$BUCKET_ALIAS" "${RESULTS_DIR}/embeddings.csv" --format csv
ok "Embeddings exported to ${RESULTS_DIR}/embeddings.csv"
if [[ -f "${RESULTS_DIR}/embeddings.csv" ]]; then
    LINES=$(wc -l < "${RESULTS_DIR}/embeddings.csv")
    log "  ${LINES} row(s) in embeddings.csv"
fi
# ---------- Done ----------
log ""
banner "Demo 02 Complete"
log "Results in: ${RESULTS_DIR}/"
