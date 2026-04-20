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
#   - On auth-enforced deployments, set UMRS_ADMIN_TOKEN for reload
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
WORKFLOW_SOURCE_DIR="${SCRIPT_DIR}/workflow"

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
$CLIENT upload-workflow "${WORKFLOW_SOURCE_DIR}" --name "vision-multi-mode" 2>&1 | while IFS= read -r line; do log "  $line"; done
ok "Workflow uploaded"

log "Reloading server configuration..."
if [[ -n "${UMRS_ADMIN_TOKEN:-}" ]]; then
    $CLIENT --admin-token "$UMRS_ADMIN_TOKEN" reload 2>/dev/null || warn "reload returned non-zero"
else
    warn "UMRS_ADMIN_TOKEN not set; reload may be rejected on auth-enforced deployments"
    $CLIENT reload 2>/dev/null || warn "reload returned non-zero (may be OK)"
fi
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

segment_count=$(find "$RESULTS_DIR" -maxdepth 1 -type f -name '*_segment.json' | wc -l | tr -d ' ')
feature_count=$(find "$RESULTS_DIR" -maxdepth 1 -type f -name '*_features.json' | wc -l | tr -d ' ')
thumbnail_count=$(find "$RESULTS_DIR" -maxdepth 1 -type f \( -name '*_128px.jpg' -o -name '*_256px.jpg' \) | wc -l | tr -d ' ')
raw_count=$(find "$RESULTS_DIR" -maxdepth 1 -type f \( -name '*_sam3_results.pkl' -o -name '*_sam3_seg.png' -o -name '*_features_results.pkl' \) | wc -l | tr -d ' ')

log "Local results (flattened download):"
log "  segment JSONs   — ${segment_count} files"
log "  feature JSONs   — ${feature_count} files"
log "  thumbnails      — ${thumbnail_count} files"
log "  raw model files — ${raw_count} files"

log ""
log "Key outputs:"
log "  *_segment.json          — segmentation metadata"
log "  *_features.json         — feature metadata + embedding vectors"
log "  *_{128,256}px.jpg       — thumbnail variants"
log "  *_sam3_seg.png          — segmentation preview image"
log "  *_sam3_results.pkl      — raw SAM3 adapter payload"
log "  *_features_results.pkl  — raw feature extractor payload"
