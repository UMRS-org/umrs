#!/usr/bin/env bash
# =============================================================================
# SAM3 Segmentation Demo — user_script version
# =============================================================================
# Equivalent to Demo 02 (SAM3 via BentoML/Rivault), but uses the
# user_script:// model type with the Pipeline SDK.  The SAM2 model is
# loaded directly by the Step class — no BentoML service required.
#
# Flow: upload images → server processes via user_script pipeline → download results
#
# Requires:
#   - Services running (demos/00_start_services/)
#   - SAM2 model weights downloadable from HuggingFace (or pre-staged)
#   - workflow.json source path updated to the actual scripts location
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UMRS_CLIENT_CONTAINER="umrs-client"

# Bucket alias — must match the user config
BUCKET_ALIAS="sam3-user-script"
USERNAME="${UMRS_USERNAME:-rivault_user}"

LOG_PREFIX="[sam3_user_script]"
log() { echo "$LOG_PREFIX $*"; }
warn() { echo "$LOG_PREFIX WARNING: $*"; }
die() { echo "$LOG_PREFIX ERROR: $*" >&2; exit 1; }
run_cmd() {
    echo "$LOG_PREFIX \$ $*"
    "$@"
}

# ---------- Step 1: Verify services ----------
log "Step 1: Verifying UMRS server..."
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client ping || \
    die "UMRS server not reachable. Start services first."

# Reload config so the server picks up the user_script model definition
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client reload || warn "reload failed"
# Retry loop — server may take a moment after reload
for i in $(seq 1 6); do
    sleep 3
    run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client ping && break
    warn "Retry $i/6 — waiting for server after reload"
done

# ---------- Step 2: Check for test images ----------
log ""
log "Step 2: Checking test images..."
# Test images are mounted into the container at /opt/test_data/images/ via docker-compose
CONTAINER_TEST_DATA="/opt/test_data/images"
image_count=$(podman exec "$UMRS_CLIENT_CONTAINER" \
    find "$CONTAINER_TEST_DATA" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.tiff' \) 2>/dev/null | wc -l)
if [[ "$image_count" -gt 0 ]]; then
    log "  Found $image_count test image(s) at $CONTAINER_TEST_DATA (inside container)"
else
    warn "No test images found at $CONTAINER_TEST_DATA inside the container"
    warn "Place .jpg/.png/.tiff files in demos/slurm-dev/test_data/images/"
    die "No test images available"
fi

# ---------- Step 3: Upload images via umrs-client ----------
log ""
log "Step 3: Uploading images to UMRS bucket '$BUCKET_ALIAS'..."

run_cmd podman exec "$UMRS_CLIENT_CONTAINER" \
    umrs-client upload_dir "$CONTAINER_TEST_DATA" "$BUCKET_ALIAS" "$USERNAME"

log "  Images uploaded. Server will process via user_script pipeline (SAM3)."
UPLOAD_DONE=$SECONDS

# ---------- Step 4: Download results ----------
log ""
log "Step 4: Downloading results..."
log "  The download_results command will poll the server until processing is complete."
log "  First run may take longer as the SAM2 model is loaded into memory."
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

CONTAINER_RESULTS="/tmp/demo_results"
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" rm -rf "$CONTAINER_RESULTS"
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" mkdir -p "$CONTAINER_RESULTS"
DOWNLOAD_START=$SECONDS
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" \
    umrs-client --verbose download_results "$BUCKET_ALIAS" "$CONTAINER_RESULTS" --force-overwrite
DOWNLOAD_ELAPSED=$(( SECONDS - DOWNLOAD_START ))
log "  Server processing + download took ${DOWNLOAD_ELAPSED}s ($(( DOWNLOAD_ELAPSED / 60 ))m $(( DOWNLOAD_ELAPSED % 60 ))s)"
run_cmd podman cp "$UMRS_CLIENT_CONTAINER:$CONTAINER_RESULTS/." "$RESULTS_DIR/"
log "  Results saved to: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR/"

# ---------- Done ----------
TOTAL_WAIT=$(( SECONDS - UPLOAD_DONE ))
log ""
log "SAM3 segmentation demo (user_script) complete!"
log "  Processing wait : ${DOWNLOAD_ELAPSED}s ($(( DOWNLOAD_ELAPSED / 60 ))m $(( DOWNLOAD_ELAPSED % 60 ))s)"
log "  Total (upload→results): ${TOTAL_WAIT}s ($(( TOTAL_WAIT / 60 ))m $(( TOTAL_WAIT % 60 ))s)"
log "Results are in $RESULTS_DIR/"
