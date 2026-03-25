#!/usr/bin/env bash
# =============================================================================
# HMAE Inference Demo — Jens's Vision Stack
# =============================================================================
# Same workflow as Demo 03, but uses Jens's standalone BentoML service
# (mini-rivault-vision) via VISION_STACK=jens.
#
# Flow: upload images → server processes via Jens's /vision_task → download results
#
# Requires:
#   - Services running (demos/00_start_services/)
#   - VISION_STACK=jens set in the environment
#   - HMAE model checkpoint pre-staged on the Rivault node
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure Jens's vision stack is selected
export VISION_STACK="${VISION_STACK:-jens}"

UMRS_CLIENT_CONTAINER="umrs-client"

BUCKET_ALIAS="hmae-demo"
USERNAME="${UMRS_USERNAME:-rivault_user}"

LOG_PREFIX="[hmae_jens]"
log() { echo "$LOG_PREFIX $*"; }
warn() { echo "$LOG_PREFIX WARNING: $*"; }
die() { echo "$LOG_PREFIX ERROR: $*" >&2; exit 1; }
run_cmd() {
    echo "$LOG_PREFIX \$ $*"
    "$@"
}

log "Using VISION_STACK=$VISION_STACK (Jens's standalone BentoML service)"

# ---------- Step 1: Verify services ----------
log "Step 1: Verifying UMRS server..."
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client ping || \
    die "UMRS server not reachable. Start services first."

run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client reload || warn "reload failed"
sleep 3

# ---------- Step 2: Check for test images ----------
log ""
log "Step 2: Checking test images..."
CONTAINER_TEST_DATA="/opt/test_data/images"
image_count=$(podman exec "$UMRS_CLIENT_CONTAINER" \
    find "$CONTAINER_TEST_DATA" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.tiff' \) 2>/dev/null | wc -l)
if [[ "$image_count" -gt 0 ]]; then
    log "  Found $image_count test image(s) at $CONTAINER_TEST_DATA (inside container)"
else
    warn "No test images found at $CONTAINER_TEST_DATA inside the container"
    warn "Place X-ray CT slice images (.png/.tiff) in demos/slurm-dev/test_data/images/"
    die "No test images available"
fi

# ---------- Step 3: Upload images via umrs-client ----------
log ""
log "Step 3: Uploading images to UMRS bucket '$BUCKET_ALIAS'..."

run_cmd podman exec "$UMRS_CLIENT_CONTAINER" \
    umrs-client upload_dir "$CONTAINER_TEST_DATA" "$BUCKET_ALIAS" "$USERNAME"

log "  Images uploaded. Server will launch Jens's BentoML stack and process automatically."

# ---------- Step 4: Download results ----------
log ""
log "Step 4: Downloading results..."
log "  The download_results command will poll the server until processing is complete."
log "  This may take several minutes on first launch (image build + model load)."
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
log ""
log "HMAE inference demo (Jens's stack) complete!"
log "Results are in $RESULTS_DIR/"
