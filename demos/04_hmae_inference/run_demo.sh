#!/usr/bin/env bash
# =============================================================================
# Demo 04: HMAE Inference
# =============================================================================
# Uploads volume slice images → server runs HMAE via BentoML on Slurm →
# downloads reconstruction results.
#
# HMAE inference engine: tmp/hmae-infer/ (see tmp/hmae-infer/README.md)
#
# Prerequisites:
#   - SSH tunnels active (ZMQ 5555/5556, MinIO 9000)
#   - UMRS server running with HMAE Inference experience
#   - BentoML vision service available on compute node
#
# Bucket alias: user1-hmae  (from user config, need_output_bucket: true)
#
# Usage:
#   bash run_demo.sh [image_dir]
#
# Optional env vars:
#   UMRS_USER_CONF        — path to user config (default: auto-discovered)
#   HMAE_LOCAL_MODEL_DIR  — if set, registers the model before inference
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="hmae_demo" source "${SCRIPT_DIR}/../lib/common.sh"

BUCKET_ALIAS="${HMAE_BUCKET:-user1-hmae}"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Default image dir: use SAM3 demo's test_data_full (shared volume slices)
IMAGE_DIR="${1:-${SCRIPT_DIR}/../03_sam3_segmentation/test_data_full}"

banner "Demo 04: HMAE Inference"
echo "User config  : ${CONF}"
echo "Bucket alias : ${BUCKET_ALIAS}"
echo "Image dir    : ${IMAGE_DIR}"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

# ---------- Step 2: Optional model registration ----------
if [[ -n "${HMAE_LOCAL_MODEL_DIR:-}" ]]; then
    log ""
    log "Step 2: Registering HMAE model from ${HMAE_LOCAL_MODEL_DIR}..."
    $CLIENT model register "hmae-pretrained" "${HMAE_LOCAL_MODEL_DIR}" \
        --type embedding --scope system \
        --description "HMAE ViT pretrained checkpoint" || warn "model register returned non-zero"
    ok "Model registered"
else
    log ""
    log "Step 2: Skipping model registration (set HMAE_LOCAL_MODEL_DIR to register)"
fi

# ---------- Step 3: Check sample images ----------
log ""
log "Step 3: Checking test images..."
if [[ ! -d "${IMAGE_DIR}" ]]; then
    die "Image directory not found: ${IMAGE_DIR}
  Place volume slice PNGs in the directory, or pass a path:
    bash run_demo.sh /path/to/volume_slices/"
fi
PNG_COUNT=$(find "${IMAGE_DIR}" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
if [[ "${PNG_COUNT}" -eq 0 ]]; then
    die "No .png files found in ${IMAGE_DIR}"
fi
log "  Found ${PNG_COUNT} PNG file(s)"

# ---------- Step 4: Upload images ----------
log ""
log "Step 4: Uploading volume slices to bucket '${BUCKET_ALIAS}'..."
$CLIENT upload_dir "${IMAGE_DIR}" "$BUCKET_ALIAS"
ok "Images uploaded — server will process automatically"
UPLOAD_DONE=$SECONDS

# ---------- Step 5: Wait & Download results ----------
log ""
log "Step 5: Waiting for HMAE processing and downloading results..."
log "  (This may take several minutes for model loading + inference)"
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
