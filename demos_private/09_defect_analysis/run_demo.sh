#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Demo 09 — Structural Defect Analysis
#
# Uploads images to UMRS → server dispatches SAM3 segmentation via
# BentoML on Slurm → downloads results.
#
# This demo uses its own Defect Analysis experience (demo-defect bucket) for
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

# Ensure Defect Analysis experience is in user config (idempotent).
ensure_experience "${SCRIPT_DIR}/experience.json"

# Reuse defect bucket alias for segmentation
BUCKET_ALIAS="${DEFECT_BUCKET:-demo-defect}"
MAX_IMAGES="${MAX_IMAGES:-2}"
RESULTS_DIR="${SCRIPT_DIR}/results"
RELOAD_SETTLE_SECONDS="${RELOAD_SETTLE_SECONDS:-30}"
SAM3_REQUIRE_VISIBLE_MODEL="${SAM3_REQUIRE_VISIBLE_MODEL:-0}"
SAM3_REQUIRE_RESULTS="${SAM3_REQUIRE_RESULTS:-0}"
HMAE_ASSET_ROOT="${HMAE_ASSET_ROOT:-${SCRIPT_DIR}/../../tmp/hmae}"

# Test images: prefer explicit images, otherwise use HMAE sample slices in umrs/tmp.
IMAGE_DIR="${IMAGE_DIR:-${HMAE_ASSET_ROOT}/images}"

banner "Demo 09: Structural Defect Analysis"
echo "User config  : ${CONF}"
echo "Bucket alias : ${BUCKET_ALIAS}"
echo "Image dir    : ${IMAGE_DIR}"
echo ""

model_visible() {
    local scope="$1" name="$2"
    run_client_bounded "${MODEL_LIST_TIMEOUT:-8}" --output json model list --scope "$scope" 2>/dev/null \
        | grep -q "\"name\": \"${name}\""
}

# ──────────────────────────────────────────────────────────────────────────
log "Step 1: Pre-flight checks..."
preflight

# General-user path: rely on the deployed SAM3 experience instead of admin reload.
log "Using the deployed SAM3 experience as-is (no admin reload in demo mode)."

# ──────────────────────────────────────────────────────────────────────────
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
    if [[ "$SAM3_REQUIRE_VISIBLE_MODEL" = "1" ]]; then
        die "SAM3 model '${SAM3_NAME}' is not visible. Ask the site operator to pre-stage it, or set SAM3_STAGING_DIR to a writable server-side user staging path if your site allows user-scoped downloads."
    fi
    warn "SAM3 model '${SAM3_NAME}' is not visible in registry; continuing so the upload/result path can be exercised"
fi

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
        | head -n "$MAX_IMAGES" | while read -r f; do ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$UPLOAD_DIR/$(basename "$f")"; done
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
if download_results_bounded "$BUCKET_ALIAS" "$RESULTS_DIR" --force-overwrite; then
    DOWNLOAD_ELAPSED=$(( SECONDS - UPLOAD_DONE ))
    log "  Processing + download took ${DOWNLOAD_ELAPSED}s"
    ok "Results saved to ${RESULTS_DIR}/"
else
    if [[ "$SAM3_REQUIRE_RESULTS" = "1" ]]; then
        die "Could not download defect-analysis results"
    fi
    warn "Could not download defect-analysis results — check server logs or set SAM3_REQUIRE_RESULTS=1 for strict validation"
fi

# ──────────────────────────────────────────────────────────────────────────
log ""
banner "Demo 09 Complete"
ls -lh "${RESULTS_DIR}/" 2>/dev/null || true
log "Results in: ${RESULTS_DIR}/"
