#!/usr/bin/env bash
# =============================================================================
# HMAE Inference Demo — rccs-cloud deployment
# =============================================================================
# Uploads HMAE model weights (optional) → uploads images → server launches
# mini-rivault HMAE on Slurm → download results.
# Run from your local Mac with SSH tunnels active.
#
# Usage:
#   ./04_hmae_inference.sh                      # uses bundled sample_data/images/
#   ./04_hmae_inference.sh /path/to/images      # your own images
#
# HMAE operates on grayscale industrial/CT images (.png/.tiff).
# Override UMRS_IMAGE_BUCKET to use a different bucket alias.
#
# Model upload:
#   Set HMAE_LOCAL_MODEL_DIR to upload & register the HMAE checkpoint first:
#     HMAE_LOCAL_MODEL_DIR=/path/to/hmae_checkpoint ./04_hmae_inference.sh
#   The checkpoint directory must contain latest_checkpoint.pth.
#
# Force recompute:
#   By default, the server skips BentoML inference when results already exist
#   in the output bucket. Set FORCE_RECOMPUTE=1 to force re-processing:
#     FORCE_RECOMPUTE=1 ./04_hmae_inference.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_CONF="${UMRS_USER_CONF:-$HOME/.config/umrs/users/user1.json}"
IMAGE_BUCKET="${UMRS_IMAGE_BUCKET:-user1-hmae}"
DATA_DIR="${1:-$SCRIPT_DIR/sample_data/images}"
RESULTS_DIR="${SCRIPT_DIR}/results/hmae"
HMAE_LOCAL_MODEL_DIR="${HMAE_LOCAL_MODEL_DIR:-}"
REMOTE_HMAE_MODEL_PATH="${UMRS_REMOTE_HMAE_MODEL_PATH:-/lvs0/dne1/rccs-dmp/mnagaso/umrs_models/hmae}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-}"

log()  { echo "[hmae] $*"; }
warn() { echo "[hmae] WARNING: $*"; }
die()  { echo "[hmae] ERROR: $*" >&2; exit 1; }

CLIENT="umrs-client -c $USER_CONF"

# ─── Preflight ───────────────────────────────────────────────────────────────
log "Config:  $USER_CONF"
log "Bucket:  $IMAGE_BUCKET"
log "Data:    $DATA_DIR"
[[ -f "$USER_CONF" ]] || die "User config not found. Run setup_client.sh first."
[[ -d "$DATA_DIR" ]]  || die "Image directory not found: $DATA_DIR"

img_count=$(find "$DATA_DIR" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.tiff' \) | wc -l | tr -d ' ')
[[ "$img_count" -gt 0 ]] || die "No .jpg/.png/.tiff images found in $DATA_DIR"
log "Found $img_count image(s)."

echo
log "Step 1: Ping server"
$CLIENT ping || die "Server not reachable. Is ssh_tunnel.sh running?"

# ─── Step 1b: Upload HMAE model to HPC (optional) ─────────────────────────────
if [[ -n "$HMAE_LOCAL_MODEL_DIR" ]]; then
    echo
    log "Step 1b: Upload HMAE model via 'umrs-client model upload'"
    if [[ ! -d "$HMAE_LOCAL_MODEL_DIR" ]]; then
        die "HMAE_LOCAL_MODEL_DIR not found: $HMAE_LOCAL_MODEL_DIR"
    fi
    if [[ ! -f "$HMAE_LOCAL_MODEL_DIR/latest_checkpoint.pth" ]]; then
        die "No latest_checkpoint.pth in $HMAE_LOCAL_MODEL_DIR"
    fi
    log "  Source (local):      $HMAE_LOCAL_MODEL_DIR"
    log "  Register path (HPC): $REMOTE_HMAE_MODEL_PATH"

    $CLIENT model upload "$HMAE_LOCAL_MODEL_DIR" \
        --name "hmae-pretrained" \
        --type embedding --scope system \
        --description "HMAE ViT pretrained checkpoint (s8d-1k)" \
        --register-path "$REMOTE_HMAE_MODEL_PATH" \
        --force
    log "  Model uploaded and registered."

    # Show registry
    log "  Current model registry:"
    $CLIENT model list
fi

# Reload server config so it picks up the HMAE experience/buckets
log "Reloading server config..."
$CLIENT reload 2>/dev/null || true

# Wait for server to finish reload (synchronous: blocks ZMQ loop while
# re-creating S3 users/buckets via mc subprocess calls).
log "Waiting for server to complete reload..."
for _try in 1 2 3 4 5 6; do
    sleep 3
    if $CLIENT --no-check-server ping >/dev/null 2>&1; then
        log "Server ready after reload."
        break
    fi
    if [[ $_try -eq 6 ]]; then
        die "Server did not recover after reload. Check server logs."
    fi
    log "  retry $_try/6 ..."
done

# ─── Step 2: Upload images ────────────────────────────────────────────────────
echo
log "Step 2: Upload images to bucket '$IMAGE_BUCKET'"

# Create a temp directory with only image files (skip README.txt, etc.)
_upload_dir="$(mktemp -d)"
trap 'rm -rf "$_upload_dir"' EXIT
find "$DATA_DIR" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.tiff' \) \
    -exec ln -s {} "$_upload_dir/" \;
log "  Staging $img_count image(s) for upload (non-image files excluded)."

$CLIENT upload_dir "$_upload_dir" "$IMAGE_BUCKET"
log "Upload done. Server will launch HMAE on a Slurm node."

# Force recompute: re-process all objects even if results already exist
if [[ -n "$FORCE_RECOMPUTE" ]]; then
    log "FORCE_RECOMPUTE is set — triggering full recompute..."
    $CLIENT recompute || warn "recompute returned non-zero (check server logs)"
fi
UPLOAD_TIME=$SECONDS

# ─── Step 3: Download results (polls until done) ──────────────────────────────
echo
log "Step 3: Download results (may take several minutes for Slurm job + model load)"
log "  HMAE produces per-image feature vectors and/or reconstructed images."
mkdir -p "$RESULTS_DIR"
$CLIENT --verbose download_results "$IMAGE_BUCKET" "$RESULTS_DIR" --force-overwrite
ELAPSED=$(( SECONDS - UPLOAD_TIME ))
log "Done in ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)."
log "Results in: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR/"
