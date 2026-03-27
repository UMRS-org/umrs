#!/usr/bin/env bash
# =============================================================================
# SAM3 Segmentation Demo — Jens's vision stack (rccs-cloud)
# =============================================================================
# Same workflow as 03_sam3_segmentation.sh but targets Jens's standalone
# BentoML service (mini-rivault-vision, VISION_STACK=jens).
#
# Usage:
#   ./07_sam3_jens.sh                     # uses bundled sample_data/images/
#   ./07_sam3_jens.sh /path/to/images     # your own images
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_CONF="${UMRS_USER_CONF:-$HOME/.config/umrs/users/user1.json}"
IMAGE_BUCKET="${UMRS_IMAGE_BUCKET:-user1-sam3}"
DATA_DIR="${1:-$SCRIPT_DIR/sample_data/images}"
RESULTS_DIR="${SCRIPT_DIR}/results/sam3_jens"

export VISION_STACK="${VISION_STACK:-jens}"

log()  { echo "[sam3_jens] $*"; }
warn() { echo "[sam3_jens] WARNING: $*"; }
die()  { echo "[sam3_jens] ERROR: $*" >&2; exit 1; }

CLIENT="umrs-client -c $USER_CONF"

log "Using VISION_STACK=$VISION_STACK (Jens's standalone BentoML service)"
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

echo
log "Step 2: Upload images to bucket '$IMAGE_BUCKET'"
$CLIENT upload_dir "$DATA_DIR" "$IMAGE_BUCKET"
log "Upload done. Server will launch Jens's BentoML stack on a Slurm node."
UPLOAD_TIME=$SECONDS

echo
log "Step 3: Download results (may take several minutes on first launch)"
mkdir -p "$RESULTS_DIR"
$CLIENT --verbose download_results "$IMAGE_BUCKET" "$RESULTS_DIR" --force-overwrite
ELAPSED=$(( SECONDS - UPLOAD_TIME ))
log "Done in ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)."
log "Results in: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR/"
