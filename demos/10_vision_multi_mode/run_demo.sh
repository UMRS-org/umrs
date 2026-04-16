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

BUCKET_ALIAS="${MULTI_BUCKET:-user1-multi}"
MAX_IMAGES="${MAX_IMAGES:-3}"
IMAGE_DIR="${1:-${SCRIPT_DIR}/../sample_data/images}"
RESULTS_DIR="${SCRIPT_DIR}/results"
WORKFLOW_SOURCE_DIR="${SCRIPT_DIR}"
WORKFLOW_UPLOAD_DIR="$(prepare_workflow_upload_dir "${WORKFLOW_SOURCE_DIR}")"
trap 'rm -rf "${WORKFLOW_UPLOAD_DIR:-}"' EXIT

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
$CLIENT upload-workflow "${WORKFLOW_UPLOAD_DIR}" --name "vision-multi-mode" 2>&1 | while IFS= read -r line; do log "  $line"; done
ok "Workflow uploaded"

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
        | head -n "$MAX_IMAGES" | while read -r f; do ln -s "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$UPLOAD_DIR/"; done
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
rm -rf "$RESULTS_DIR"
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
    subdir_path="${RESULTS_DIR}/${subdir}"
    if [[ -d "$subdir_path" ]]; then
        count=$(find "$subdir_path" -type f | wc -l | tr -d ' ')
    else
        count=0
    fi
    log "  ${subdir}/ — ${count} files"
done

log ""
log "Key outputs:"
log "  segment/    — *_segment.json  (bounding boxes + scores)"
log "  segment/    — *_segment_masks.npz  (per-object binary masks)"
log "  features/   — *_features.json (embedding vectors)"
log "  thumbnails/ — *_128px.jpg, *_256px.jpg (resized variants)"
