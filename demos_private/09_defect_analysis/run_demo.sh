#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Demo 09 — Structural Defect Analysis
#
# Uploads images to UMRS → server dispatches SAM3 segmentation via
# BentoML on Slurm → downloads results.
#
# This demo reuses the SAM3 experience (user1-sam3 bucket) to demonstrate
# defect-oriented segmentation. For the full multi-step UDM pipeline
# (SAM3 + region extraction + HMAE + clustering), see the Python step
# files in steps/ and workflow.json.
#
# Flow: upload images → server processes → download results
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running with SAM3 Segmentation experience
#   - BentoML vision service available on compute node
# ──────────────────────────────────────────────────────────────────────────
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="defect" source "${SCRIPT_DIR}/../lib/common.sh"

# Reuse SAM3 bucket alias for segmentation
BUCKET_ALIAS="${DEFECT_BUCKET:-user1-sam3}"
MAX_IMAGES="${MAX_IMAGES:-2}"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Test images: use SAM3 demo's test_data_full
IMAGE_DIR="${IMAGE_DIR:-${SCRIPT_DIR}/../03_sam3_segmentation/test_data_full}"

banner "Demo 09: Structural Defect Analysis"
echo "User config  : ${CONF}"
echo "Bucket alias : ${BUCKET_ALIAS}"
echo "Image dir    : ${IMAGE_DIR}"
echo ""

# ──────────────────────────────────────────────────────────────────────────
log "Step 1: Pre-flight checks..."
preflight

# Reload config so the server creates output buckets
$CLIENT reload 2>/dev/null || warn "reload returned non-zero"
sleep 30  # server needs time to restart after reload

# ──────────────────────────────────────────────────────────────────────────
log ""
log "Step 2: Preparing SAM3 model..."
SAM3_REPO="${SAM3_MODEL_REPO:-facebook/sam2-hiera-small}"
$CLIENT model download "$SAM3_REPO" --name sam3 --scope system --server-side 2>&1 \
    | while IFS= read -r line; do log "  $line"; done
ok "SAM3 model download requested"

# ──────────────────────────────────────────────────────────────────────────
log ""
log "Step 3: Checking test images..."
if [[ ! -d "$IMAGE_DIR" ]]; then
    die "Image directory not found: $IMAGE_DIR"
fi
IMAGE_COUNT=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.tiff" \) | wc -l)
if [[ "$IMAGE_COUNT" -eq 0 ]]; then
    die "No image files found in $IMAGE_DIR"
fi

# Stage a subset
UPLOAD_DIR="$IMAGE_DIR"
if [[ "$IMAGE_COUNT" -gt "$MAX_IMAGES" ]]; then
    UPLOAD_DIR="${SCRIPT_DIR}/.upload_staging"
    rm -rf "$UPLOAD_DIR"
    mkdir -p "$UPLOAD_DIR"
    find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.tiff" \) \
        | head -n "$MAX_IMAGES" | while read -r f; do ln -s "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$UPLOAD_DIR/"; done
    log "  Staged ${MAX_IMAGES} of ${IMAGE_COUNT} images for upload"
else
    log "  Using all ${IMAGE_COUNT} images"
fi

# ──────────────────────────────────────────────────────────────────────────
log ""
log "Step 4: Uploading images to bucket '${BUCKET_ALIAS}'..."
$CLIENT upload_dir "$UPLOAD_DIR" "$BUCKET_ALIAS"
ok "Images uploaded — server will process automatically"
UPLOAD_DONE=$SECONDS

# ──────────────────────────────────────────────────────────────────────────
log ""
log "Step 5: Downloading results..."
log "  (This may take several minutes if launching BentoML for the first time)"
mkdir -p "$RESULTS_DIR"
$CLIENT download_results "$BUCKET_ALIAS" "$RESULTS_DIR" --force-overwrite
DOWNLOAD_ELAPSED=$(( SECONDS - UPLOAD_DONE ))
log "  Processing + download took ${DOWNLOAD_ELAPSED}s"
ok "Results saved to ${RESULTS_DIR}/"

# ──────────────────────────────────────────────────────────────────────────
log ""
banner "Demo 09 Complete"
ls -lh "${RESULTS_DIR}/" 2>/dev/null || true
log "Results in: ${RESULTS_DIR}/"
