#!/usr/bin/env bash
# =============================================================================
# HMAE Inference Demo
# =============================================================================
# Uploads images to UMRS via umrs-client. The UMRS server automatically
# launches the BentoML vision service on Rivault/Slurm, runs HMAE inference,
# and writes results to the output S3 bucket.
#
# Flow: (optional model upload) → upload images → server processes → download results
#
# Requires:
#   - Services running (demos/00_start_services/)
#   - HMAE model checkpoint pre-staged on the Rivault node
#
# Model upload:
#   Set HMAE_LOCAL_MODEL_DIR to upload & register the HMAE checkpoint first:
#     HMAE_LOCAL_MODEL_DIR=/path/to/hmae_checkpoint ./run_demo.sh
#   The checkpoint directory must contain latest_checkpoint.pth.
#
# Force recompute:
#   By default, the server skips BentoML inference when results already exist
#   in the output bucket. Set FORCE_RECOMPUTE=1 to force re-processing:
#     FORCE_RECOMPUTE=1 ./run_demo.sh
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UMRS_CLIENT_CONTAINER="umrs-client"

# Bucket alias as defined in user_rivault.json
BUCKET_ALIAS="hmae-demo"
# Username as defined in user_rivault.json
USERNAME="${UMRS_USERNAME:-rivault_user}"
HMAE_LOCAL_MODEL_DIR="${HMAE_LOCAL_MODEL_DIR:-}"
REMOTE_HMAE_MODEL_PATH="${UMRS_REMOTE_HMAE_MODEL_PATH:-/shared/models/hmae}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-}"

LOG_PREFIX="[hmae_demo]"
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
log "Reloading server config..."
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client reload 2>/dev/null || true

# Wait for server to finish reload (synchronous: blocks ZMQ loop while
# re-creating S3 users/buckets via mc subprocess calls).
log "Waiting for server to complete reload..."
for _try in 1 2 3 4 5 6; do
    sleep 3
    if podman exec "$UMRS_CLIENT_CONTAINER" umrs-client --no-check-server ping >/dev/null 2>&1; then
        log "Server ready after reload."
        break
    fi
    if [[ $_try -eq 6 ]]; then
        die "Server did not recover after reload. Check server logs."
    fi
    log "  retry $_try/6 ..."
done

# ---------- Step 1b: Upload HMAE model (optional) ----------
if [[ -n "$HMAE_LOCAL_MODEL_DIR" ]]; then
    log ""
    log "Step 1b: Upload HMAE model via 'umrs-client model upload'"
    if [[ ! -d "$HMAE_LOCAL_MODEL_DIR" ]]; then
        die "HMAE_LOCAL_MODEL_DIR not found: $HMAE_LOCAL_MODEL_DIR"
    fi
    if [[ ! -f "$HMAE_LOCAL_MODEL_DIR/latest_checkpoint.pth" ]]; then
        die "No latest_checkpoint.pth in $HMAE_LOCAL_MODEL_DIR"
    fi
    log "  Source (local):      $HMAE_LOCAL_MODEL_DIR"
    log "  Register path (HPC): $REMOTE_HMAE_MODEL_PATH"

    # Copy model into container, then upload via umrs-client
    CONTAINER_MODEL_DIR="/tmp/hmae_model_upload"
    run_cmd podman exec "$UMRS_CLIENT_CONTAINER" rm -rf "$CONTAINER_MODEL_DIR"
    run_cmd podman cp "$HMAE_LOCAL_MODEL_DIR/." "$UMRS_CLIENT_CONTAINER:$CONTAINER_MODEL_DIR"
    run_cmd podman exec "$UMRS_CLIENT_CONTAINER" \
        umrs-client model upload "$CONTAINER_MODEL_DIR" \
            --name "hmae-pretrained" \
            --type embedding --scope system \
            --description "HMAE ViT pretrained checkpoint (s8d-1k)" \
            --register-path "$REMOTE_HMAE_MODEL_PATH" \
            --force
    log "  Model uploaded and registered."
    log "  Current model registry:"
    run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client model list
fi

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
    warn "Place X-ray CT slice images (.png/.tiff) in demos/slurm-dev/test_data/images/"
    die "No test images available"
fi

# ---------- Step 3: Upload images via umrs-client ----------
log ""
log "Step 3: Uploading images to UMRS bucket '$BUCKET_ALIAS'..."

# Create a temp directory inside the container with only image files (skip README.txt, etc.)
CONTAINER_UPLOAD_DIR="/tmp/demo_upload"
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" rm -rf "$CONTAINER_UPLOAD_DIR"
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" mkdir -p "$CONTAINER_UPLOAD_DIR"
podman exec "$UMRS_CLIENT_CONTAINER" bash -c \
    "find $CONTAINER_TEST_DATA -maxdepth 1 -type f \\( -name '*.jpg' -o -name '*.png' -o -name '*.tiff' \\) -exec ln -s {} $CONTAINER_UPLOAD_DIR/ \\;"
log "  Staging $image_count image(s) for upload (non-image files excluded)."

# Upload via umrs-client (server detects new objects and processes them automatically)
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" \
    umrs-client upload_dir "$CONTAINER_UPLOAD_DIR" "$BUCKET_ALIAS" "$USERNAME"

log "  Images uploaded. Server will launch Rivault/BentoML and process automatically."

# Force recompute: re-process all objects even if results already exist
if [[ -n "$FORCE_RECOMPUTE" ]]; then
    log "FORCE_RECOMPUTE is set — triggering full recompute..."
    podman exec "$UMRS_CLIENT_CONTAINER" umrs-client recompute || warn "recompute returned non-zero (check server logs)"
fi

# ---------- Step 4: Download results ----------
log ""
log "Step 4: Downloading results..."
log "  The download_results command will poll the server until processing is complete."
log "  This may take several minutes if the server is launching Rivault/BentoML for the first time."
log "  HMAE produces per-image feature vectors and/or reconstructed images."
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
log "HMAE inference demo complete!"
log "  Processing wait: ${DOWNLOAD_ELAPSED}s ($(( DOWNLOAD_ELAPSED / 60 ))m $(( DOWNLOAD_ELAPSED % 60 ))s)"
log "Results are in $RESULTS_DIR/"
