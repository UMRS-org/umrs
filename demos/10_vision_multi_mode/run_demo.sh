#!/usr/bin/env bash
# =============================================================================
# Demo 10: Multi-Mode Vision Pipeline
# =============================================================================
# Run multiple vision analysis modes **in parallel** on the same images:
#   1. Segmentation — object detection via BentoML (GPU)
#   2. Features     — embedding extraction via BentoML (GPU)
#   3. Thumbnails   — resize variants (CPU only)
#
# This demo shows how to configure parallel branches in a workflow.json
# and how GPU and CPU steps can run concurrently.
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - BentoML vision service available on compute node
#
# Usage:
#   bash run_demo.sh                         # sample images
#   bash run_demo.sh /path/to/images         # your own images
#   MULTI_BUCKET=my-bucket bash run_demo.sh  # custom bucket
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="vision_multi" source "${SCRIPT_DIR}/../lib/common.sh"

BUCKET_ALIAS="${MULTI_BUCKET:-user1-sam3}"
MAX_IMAGES="${MAX_IMAGES:-3}"
IMAGE_DIR="${1:-${SCRIPT_DIR}/../sample_data/images}"
RESULTS_DIR="${SCRIPT_DIR}/results"
WORKFLOW_DIR="${SCRIPT_DIR}"

banner "Demo 10: Multi-Mode Vision Pipeline"
echo "Config        : ${CONF}"
echo "Bucket alias  : ${BUCKET_ALIAS}"
echo "Image dir     : ${IMAGE_DIR}"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

# ---------- Step 2: Upload workflow ----------
log ""
log "Step 2: Uploading workflow definition..."
if $CLIENT upload-workflow "${WORKFLOW_DIR}" 2>/dev/null; then
    ok "Workflow uploaded"
else
    warn "upload-workflow returned non-zero — continuing with existing workflow"
fi

$CLIENT reload 2>/dev/null || warn "reload returned non-zero (may be OK)"
sleep 3

# ---------- Step 3: Prepare and upload images ----------
log ""
log "Step 3: Checking images..."
if [[ ! -d "$IMAGE_DIR" ]]; then
    die "Image directory not found: $IMAGE_DIR
  Place .jpg/.png images in demos/sample_data/images/ or provide a path."
fi
IMAGE_COUNT=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.tiff" \) | wc -l | tr -d ' ')
if [[ "$IMAGE_COUNT" -eq 0 ]]; then
    die "No image files found in $IMAGE_DIR"
fi
log "  Found ${IMAGE_COUNT} image(s)"

UPLOAD_DIR="$IMAGE_DIR"
if [[ "$IMAGE_COUNT" -gt "$MAX_IMAGES" ]]; then
    UPLOAD_DIR="${SCRIPT_DIR}/.upload_staging"
    rm -rf "$UPLOAD_DIR"
    mkdir -p "$UPLOAD_DIR"
    find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.tiff" \) \
        | head -n "$MAX_IMAGES" | while read -r f; do ln -s "$f" "$UPLOAD_DIR/"; done
    log "  Staged ${MAX_IMAGES} of ${IMAGE_COUNT} images for upload"
fi

log ""
log "Step 3b: Uploading images to '${BUCKET_ALIAS}'..."
$CLIENT upload_dir "$UPLOAD_DIR" "$BUCKET_ALIAS"
ok "Images uploaded — 3 parallel branches will execute"

# ---------- Step 4: Wait and download ----------
log ""
log "Step 4: Waiting for all parallel branches to complete..."
log "  Branches running concurrently:"
log "    • segment   — object detection (GPU)"
log "    • features  — embedding extraction (GPU)"
log "    • thumbnails — image resizing (CPU)"
log ""
START=$SECONDS
mkdir -p "$RESULTS_DIR"
$CLIENT download_results "$BUCKET_ALIAS" "$RESULTS_DIR" --force-overwrite
ELAPSED=$(( SECONDS - START ))
log "  Processing + download took ${ELAPSED}s"
ok "Results saved to ${RESULTS_DIR}/"

# ---------- Summary ----------
log ""
banner "Demo 10 Complete"

log "Output structure:"
for subdir in segment features thumbnails; do
    count=$(find "${RESULTS_DIR}/${subdir}" -type f 2>/dev/null | wc -l | tr -d ' ')
    log "  ${subdir}/ — ${count} files"
done

log ""
log "Key outputs:"
log "  segment/    — *_segment.json  (bounding boxes + scores)"
log "  features/   — *_features.json (embedding vectors)"
log "  thumbnails/ — *_128px.jpg, *_256px.jpg (resized variants)"
