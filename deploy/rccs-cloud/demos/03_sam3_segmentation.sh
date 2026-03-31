#!/usr/bin/env bash
# =============================================================================
# SAM3 Segmentation Demo — rccs-cloud deployment
# =============================================================================
# Uploads images → server launches mini-rivault SAM3 on Slurm → download results.
# Run from your local Mac with SSH tunnels active.
#
# Usage:
#   ./03_sam3_segmentation.sh                     # uses bundled sample_data/images/
#   ./03_sam3_segmentation.sh /path/to/images     # your own images
#
# The second positional arg overrides the bucket alias (default: user1-images).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_CONF="${UMRS_USER_CONF:-$HOME/.config/umrs/users/user1.json}"
IMAGE_BUCKET="${UMRS_IMAGE_BUCKET:-user1-sam3}"
DATA_DIR="${1:-$SCRIPT_DIR/sample_data/images}"
RESULTS_DIR="${SCRIPT_DIR}/results/sam3"

log()  { echo "[sam3] $*"; }
warn() { echo "[sam3] WARNING: $*"; }
die()  { echo "[sam3] ERROR: $*" >&2; exit 1; }

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

# Reload server config so it picks up the SAM3 experience/buckets
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
$CLIENT upload_dir "$DATA_DIR" "$IMAGE_BUCKET"
log "Upload done. Server will launch SAM3 on a Slurm node."
UPLOAD_TIME=$SECONDS

# ─── Step 3: Download results (polls until done) ──────────────────────────────
echo
log "Step 3: Download results (may take several minutes for Slurm job + model load)"
mkdir -p "$RESULTS_DIR"
$CLIENT --verbose download_results "$IMAGE_BUCKET" "$RESULTS_DIR" --force-overwrite
ELAPSED=$(( SECONDS - UPLOAD_TIME ))
log "Done in ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)."
log "Results in: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR/"
