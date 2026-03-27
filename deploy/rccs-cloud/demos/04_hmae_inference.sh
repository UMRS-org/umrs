#!/usr/bin/env bash
# =============================================================================
# HMAE Inference Demo — rccs-cloud deployment
# =============================================================================
# Uploads images → server launches mini-rivault HMAE on Slurm → download results.
# Run from your local Mac with SSH tunnels active.
#
# Usage:
#   ./04_hmae_inference.sh                      # uses bundled sample_data/images/
#   ./04_hmae_inference.sh /path/to/images      # your own images
#
# HMAE operates on grayscale industrial/CT images (.png/.tiff).
# Override UMRS_IMAGE_BUCKET to use a different bucket alias.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_CONF="${UMRS_USER_CONF:-$HOME/.config/umrs/users/user1.json}"
IMAGE_BUCKET="${UMRS_IMAGE_BUCKET:-user1-hmae}"
DATA_DIR="${1:-$SCRIPT_DIR/sample_data/images}"
RESULTS_DIR="${SCRIPT_DIR}/results/hmae"

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

# Reload server config so it picks up the HMAE experience/buckets
log "Reloading server config..."
$CLIENT reload 2>/dev/null || true
sleep 2

# ─── Step 2: Upload images ────────────────────────────────────────────────────
echo
log "Step 2: Upload images to bucket '$IMAGE_BUCKET'"
$CLIENT upload_dir "$DATA_DIR" "$IMAGE_BUCKET"
log "Upload done. Server will launch HMAE on a Slurm node."
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
