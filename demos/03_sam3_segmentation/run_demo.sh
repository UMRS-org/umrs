#!/usr/bin/env bash
# =============================================================================
# Demo 03: SAM3 Segmentation
# =============================================================================
# Uploads images to UMRS → server dispatches SAM3 via BentoML on Slurm →
# downloads segmentation results.
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running with SAM3 Segmentation experience
#   - BentoML vision service available on compute node
#
# Bucket alias: user1-sam3  (from config, need_output_bucket: true)
#
# Force recompute:
#   FORCE_RECOMPUTE=1 bash run_demo.sh
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="sam3_demo" source "${SCRIPT_DIR}/../lib/common.sh"

BUCKET_ALIAS="${SAM3_BUCKET:-user1-sam3}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-}"
MAX_IMAGES="${MAX_IMAGES:-3}"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Sample images: use test_data_full/ in this demo dir
IMAGE_DIR="${IMAGE_DIR:-${SCRIPT_DIR}/test_data_full}"

banner "Demo 03: SAM3 Segmentation"
echo "Config       : ${CONF}"
echo "Bucket alias : ${BUCKET_ALIAS}"
echo "Image dir    : ${IMAGE_DIR}"
echo ""

model_visible() {
    local scope="$1" name="$2"
    $CLIENT model list --scope "$scope" 2>/dev/null | grep -q "\"name\": \"${name}\""
}

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

# Reload config so server creates output buckets
log "Reloading server configuration..."
$CLIENT reload 2>/dev/null || warn "reload returned non-zero (may be OK)"
sleep 3

# ---------- Step 2: Prepare SAM3 model ----------
log ""
log "Step 2: Checking SAM3 model availability..."
SAM3_REPO="${SAM3_MODEL_REPO:-facebook/sam2-hiera-small}"
SAM3_NAME="${SAM3_MODEL_NAME:-sam3}"
SAM3_STAGING_DIR="${SAM3_STAGING_DIR:-}"
if model_visible system "$SAM3_NAME" || model_visible user "$SAM3_NAME"; then
    ok "SAM3 model already visible in the registry: ${SAM3_NAME}"
elif [[ -n "$SAM3_STAGING_DIR" ]]; then
    $CLIENT model download "$SAM3_REPO" --name "$SAM3_NAME" --scope user --server-side --staging-dir "$SAM3_STAGING_DIR" 2>&1 \
        | while IFS= read -r line; do log "  $line"; done
    ok "SAM3 model download requested via SAM3_STAGING_DIR=${SAM3_STAGING_DIR}"
else
    die "SAM3 model '${SAM3_NAME}' is not visible. Stage it as an admin, or set SAM3_STAGING_DIR to a writable, allowed server path."
fi

# ---------- Step 3: Check test images ----------
log ""
log "Step 3: Checking test images..."
if [[ ! -d "$IMAGE_DIR" ]]; then
    die "Image directory not found: $IMAGE_DIR"
fi
IMAGE_COUNT=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.tiff" \) | wc -l)
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
        | head -n "$MAX_IMAGES" | while read -r f; do ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$UPLOAD_DIR/$(basename "$f")"; done
    log "  Staged ${MAX_IMAGES} of ${IMAGE_COUNT} images for upload"
else
    log "  Using all ${IMAGE_COUNT} images"
fi

# ---------- Step 4: Upload images ----------
log ""
log "Step 4: Uploading images to bucket '${BUCKET_ALIAS}'..."
$CLIENT upload_dir "$UPLOAD_DIR" "$BUCKET_ALIAS"
ok "Images uploaded — server will process automatically"

if [[ -n "$FORCE_RECOMPUTE" ]]; then
    log "FORCE_RECOMPUTE set — triggering full recompute..."
    $CLIENT recompute || warn "recompute returned non-zero"
fi
UPLOAD_DONE=$SECONDS

# ---------- Step 5: Wait & Download results ----------
log ""
log "Step 5: Waiting for processing and downloading results..."
log "  (This may take several minutes if launching BentoML for the first time)"
mkdir -p "$RESULTS_DIR"
$CLIENT download_results "$BUCKET_ALIAS" "$RESULTS_DIR" --force-overwrite
DOWNLOAD_ELAPSED=$(( SECONDS - UPLOAD_DONE ))
log "  Processing + download took ${DOWNLOAD_ELAPSED}s"
ok "Results saved to ${RESULTS_DIR}/"

# ---------- Summary ----------
log ""
banner "Demo 03 Complete"
ls -lh "$RESULTS_DIR/" 2>/dev/null || true
log "Results in: ${RESULTS_DIR}/"
