#!/usr/bin/env bash
# =============================================================================
# Demo 04: Vision Model Inference
# =============================================================================
# Generic vision model inference demo.  Demonstrates uploading images and
# running a BentoML vision model on HPC via the UMRS pipeline.
#
# By default this demo runs SAM3 segmentation (same model as Demo 03) but
# illustrates how ANY vision model registered in the model registry can be
# invoked through the same interface.
#
# To plug in a different model:
#   1. Register it: umrs-client model register <name> <path> --type embedding
#   2. Update your config experience to reference the new model
#   3. Reload the server: umrs-client reload
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - BentoML vision service available on compute node
#
# Usage:
#   bash run_demo.sh                         # sample images
#   bash run_demo.sh /path/to/images         # your own images
#   VISION_BUCKET=my-bucket bash run_demo.sh # custom bucket alias
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="vision_infer" source "${SCRIPT_DIR}/../lib/common.sh"

BUCKET_ALIAS="${VISION_BUCKET:-user1-sam3}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-}"
MAX_IMAGES="${MAX_IMAGES:-3}"
IMAGE_DIR="${1:-${SCRIPT_DIR}/../sample_data/images}"
RESULTS_DIR="${SCRIPT_DIR}/results"

banner "Demo 04: Vision Model Inference"
echo "Config       : ${CONF}"
echo "Bucket alias : ${BUCKET_ALIAS}"
echo "Image dir    : ${IMAGE_DIR}"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

# Reload config to ensure bucket exists
log "Reloading server configuration..."
$CLIENT reload 2>/dev/null || warn "reload returned non-zero (may be OK)"
sleep 3

# ---------- Step 2: Prepare images ----------
log ""
log "Step 2: Checking images..."
if [[ ! -d "$IMAGE_DIR" ]]; then
    die "Image directory not found: $IMAGE_DIR
  Place .jpg/.png/.tiff images in demos/sample_data/images/ or provide a path."
fi
IMAGE_COUNT=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.tiff" \) | wc -l | tr -d ' ')
if [[ "$IMAGE_COUNT" -eq 0 ]]; then
    die "No image files found in $IMAGE_DIR"
fi
log "  Found ${IMAGE_COUNT} image(s)"

# Stage a subset if MAX_IMAGES < total
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

# ---------- Step 3: Upload images ----------
log ""
log "Step 3: Uploading images to bucket '${BUCKET_ALIAS}'..."
$CLIENT upload_dir "$UPLOAD_DIR" "$BUCKET_ALIAS"
ok "Images uploaded — server will dispatch to vision model"

if [[ -n "$FORCE_RECOMPUTE" ]]; then
    log "FORCE_RECOMPUTE set — triggering full recompute..."
    $CLIENT recompute || warn "recompute returned non-zero"
fi
UPLOAD_DONE=$SECONDS

# ---------- Step 4: Wait & Download results ----------
log ""
log "Step 4: Waiting for processing and downloading results..."
log "  (May take several minutes if launching the vision service for the first time)"
mkdir -p "$RESULTS_DIR"
$CLIENT download_results "$BUCKET_ALIAS" "$RESULTS_DIR" --force-overwrite
DOWNLOAD_ELAPSED=$(( SECONDS - UPLOAD_DONE ))
log "  Processing + download took ${DOWNLOAD_ELAPSED}s"
ok "Results saved to ${RESULTS_DIR}/"

# ---------- Summary ----------
log ""
banner "Demo 04 Complete"
ls -lh "$RESULTS_DIR/" 2>/dev/null || true
log "Results in: ${RESULTS_DIR}/"
log ""
log "To use a different vision model:"
log "  1. Register:  umrs-client model register <name> <path> --type embedding"
log "  2. Configure: add an experience referencing the model in your config"
log "  3. Reload:    umrs-client reload"
log "  4. Re-run:    VISION_BUCKET=<alias> bash run_demo.sh"
