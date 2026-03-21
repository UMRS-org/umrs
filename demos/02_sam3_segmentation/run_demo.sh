#!/usr/bin/env bash
# =============================================================================
# SAM3 Segmentation Demo
# =============================================================================
# Uploads images to UMRS via umrs-client. The UMRS server automatically
# launches the BentoML vision service on Rivault/Slurm, runs SAM3
# segmentation, and writes results to the output S3 bucket.
#
# Flow: upload images → server processes → download results
#
# Requires:
#   - Services running (demos/00_start_services/)
#   - SAM3 model weights pre-staged on the Rivault node
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UMRS_CLIENT_CONTAINER="umrs-client"

# Bucket alias as defined in user_rivault.json
BUCKET_ALIAS="vision-demo"
# Username as defined in user_rivault.json
USERNAME="${UMRS_USERNAME:-rivault_user}"

LOG_PREFIX="[sam3_demo]"
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

# Reload config so the server creates vision output buckets and grants user access
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client reload || warn "reload failed"
sleep 3  # brief pause for server to process reload

# ---------- Step 2: Prepare SAM3 model ----------
log ""
log "Step 2: Preparing SAM3 model weights..."
# Download SAM3 model (facebook/sam2-hiera-small) from HuggingFace.
# Two download modes are available:
#   Client-side (default):  downloads on the client, writes to shared volume
#   Server-side (--server-side): sends request to server, downloads in background
# The UMRS_MODEL_STAGING_ROOT env var points to /shared/models (shared volume).
SAM3_REPO="${SAM3_MODEL_REPO:-facebook/sam2-hiera-small}"
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" \
    umrs-client model download "$SAM3_REPO" --name sam3 \
    --scope system 2>&1 | while IFS= read -r line; do log "  $line"; done
log "SAM3 model ready."

# ---------- Step 3: Check for test images ----------
log ""
log "Step 3: Checking test images..."
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

# ---------- Step 4: Upload images via umrs-client ----------
log ""
log "Step 4: Uploading images to UMRS bucket '$BUCKET_ALIAS'..."

# Upload via umrs-client (server detects new objects and processes them automatically)
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" \
    umrs-client upload_dir "$CONTAINER_TEST_DATA" "$BUCKET_ALIAS" "$USERNAME"

log "  Images uploaded. Server will launch Rivault/BentoML and process automatically."
UPLOAD_DONE=$SECONDS

# ---------- Step 5: Download results ----------
log ""
log "Step 5: Downloading results..."
log "  The download_results command will poll the server until processing is complete."
log "  This may take several minutes if the server is launching Rivault/BentoML for the first time."
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
log "SAM3 segmentation demo complete!"
log "  Processing wait : ${DOWNLOAD_ELAPSED}s ($(( DOWNLOAD_ELAPSED / 60 ))m $(( DOWNLOAD_ELAPSED % 60 ))s)"
log "  Total (upload→results): ${TOTAL_WAIT}s ($(( TOTAL_WAIT / 60 ))m $(( TOTAL_WAIT % 60 ))s)"
log "Results are in $RESULTS_DIR/"
