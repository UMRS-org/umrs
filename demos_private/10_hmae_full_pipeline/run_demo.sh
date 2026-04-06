#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Demo 10 — HMAE Full Pipeline
#
# Uploads volume slice images to UMRS → server runs HMAE via BentoML on
# Slurm → downloads reconstruction/feature/MAE results.
#
# Note: The current user1.json defines a single HMAE bucket (user1-hmae)
# configured for "reconstruction" mode. To run all 3 modes in parallel,
# add additional experiences with separate bucket aliases (see README.md).
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running with HMAE Inference experience
#   - BentoML vision service available on compute node
# ──────────────────────────────────────────────────────────────────────────
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="hmae_full" source "${SCRIPT_DIR}/../lib/common.sh"

BUCKET_ALIAS="${HMAE_BUCKET:-user1-hmae}"
MAX_IMAGES="${MAX_IMAGES:-3}"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Test images: use SAM3 demo's test_data_full
IMAGE_DIR="${IMAGE_DIR:-${SCRIPT_DIR}/../03_sam3_segmentation/test_data_full}"

banner "Demo 10: HMAE Full Pipeline"
echo "User config  : ${CONF}"
echo "Bucket alias : ${BUCKET_ALIAS}"
echo "Image dir    : ${IMAGE_DIR}"
echo ""

# ──────────────────────────────────────────────────────────────────────────
log "Step 1: Pre-flight checks..."
preflight
$CLIENT reload 2>/dev/null || warn "reload returned non-zero"
sleep 3

# ──────────────────────────────────────────────────────────────────────────
log ""
log "Step 2: Checking test images..."
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
        | head -n "$MAX_IMAGES" | while read -r f; do ln -s "$f" "$UPLOAD_DIR/"; done
    log "  Staged ${MAX_IMAGES} of ${IMAGE_COUNT} images for upload"
else
    log "  Using all ${IMAGE_COUNT} images"
fi

# ──────────────────────────────────────────────────────────────────────────
log ""
log "Step 3: Uploading images to bucket '${BUCKET_ALIAS}'..."
$CLIENT upload_dir "$UPLOAD_DIR" "$BUCKET_ALIAS"
ok "Images uploaded — server will process automatically"
UPLOAD_DONE=$SECONDS

# ──────────────────────────────────────────────────────────────────────────
log ""
log "Step 4: Downloading results..."
log "  (This may take several minutes for HMAE model loading + inference)"
mkdir -p "$RESULTS_DIR"
$CLIENT download_results "$BUCKET_ALIAS" "$RESULTS_DIR" --force-overwrite
DOWNLOAD_ELAPSED=$(( SECONDS - UPLOAD_DONE ))
log "  Processing + download took ${DOWNLOAD_ELAPSED}s"
ok "Results saved to ${RESULTS_DIR}/"

# ──────────────────────────────────────────────────────────────────────────
log ""
banner "Demo 10 Complete"
RESULT_COUNT=$(find "$RESULTS_DIR" -type f 2>/dev/null | wc -l)
log "  ${RESULT_COUNT} file(s) in results/"
ls -lh "$RESULTS_DIR/" 2>/dev/null || true
log "Results in: ${RESULTS_DIR}/"
