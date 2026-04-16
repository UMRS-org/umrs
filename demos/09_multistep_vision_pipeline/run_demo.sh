#!/usr/bin/env bash
# =============================================================================
# Demo 09: Multi-Step Vision Pipeline
# =============================================================================
# End-to-end 3-step pipeline: segment images → crop detected regions → analyse
# patches.  Demonstrates the UMRS Step/workflow.json pattern with chained
# processing stages.
#
# The pipeline is model-agnostic — swap the segmentation model by changing
# ``workflow.json → steps[0].config.bentoml_url``.
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - BentoML vision service available on compute node
#   - workflow.json + steps/ uploaded to server
#
# Usage:
#   bash run_demo.sh                          # use sample images
#   bash run_demo.sh /path/to/images          # your own images
#   PIPELINE_BUCKET=my-bucket bash run_demo.sh
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="vision_pipeline" source "${SCRIPT_DIR}/../lib/common.sh"

BUCKET_ALIAS="${PIPELINE_BUCKET:-user1-pipeline}"
MAX_IMAGES="${MAX_IMAGES:-3}"
IMAGE_DIR="${1:-${SCRIPT_DIR}/../sample_data/images}"
RESULTS_DIR="${SCRIPT_DIR}/results"
WORKFLOW_SOURCE_DIR="${SCRIPT_DIR}"
WORKFLOW_UPLOAD_DIR="$(prepare_workflow_upload_dir "${WORKFLOW_SOURCE_DIR}")"
trap 'rm -rf "${WORKFLOW_UPLOAD_DIR:-}"' EXIT

banner "Demo 09: Multi-Step Vision Pipeline"
echo "Config        : ${CONF}"
echo "Bucket alias  : ${BUCKET_ALIAS}"
echo "Image dir     : ${IMAGE_DIR}"
echo "Workflow dir  : ${WORKFLOW_SOURCE_DIR}"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

# ---------- Step 2: Upload workflow ----------
log ""
log "Step 2: Uploading workflow definition..."
$CLIENT upload-workflow "${WORKFLOW_UPLOAD_DIR}" --name "vision-pipeline-demo" 2>&1 | while IFS= read -r line; do log "  $line"; done
ok "Workflow uploaded"

# Reload to pick up the new workflow
log "Reloading server configuration..."
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

# Stage a subset if needed
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
log "Step 3b: Uploading images to bucket '${BUCKET_ALIAS}'..."
$CLIENT upload_dir "$UPLOAD_DIR" "$BUCKET_ALIAS"
ok "Images uploaded"

# ---------- Step 4: Wait and download results ----------
log ""
log "Step 4: Waiting for pipeline to complete..."
log "  The pipeline runs 3 chained steps:"
log "    1. Segment          – detect objects via BentoML vision model"
log "    2. Crop Regions     – extract bounding-box patches"
log "    3. Image Statistics – compute per-patch metrics + aggregate report"
log ""
log "  This may take several minutes on first run."
START=$SECONDS
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
$CLIENT download_results "$BUCKET_ALIAS" "$RESULTS_DIR" --force-overwrite
ELAPSED=$(( SECONDS - START ))
log "  Processing + download took ${ELAPSED}s"
ok "Results saved to ${RESULTS_DIR}/"

# ---------- Step 5: Show summary ----------
log ""
banner "Demo 09 Complete"

REPORT="${RESULTS_DIR}/analysis/report.json"
if [[ -f "$REPORT" ]]; then
    log "Aggregate report:"
    cat "$REPORT"
else
    log "Result files:"
    ls -lh "$RESULTS_DIR/" 2>/dev/null || true
fi

log ""
log "Pipeline outputs:"
log "  - Segmentation metadata: results/*_segment_meta.json"
log "  - Segmentation masks:    results/*_segment_masks.npz"
log "  - Cropped patches:       results/patches/"
log "  - Analysis reports:       results/analysis/"
