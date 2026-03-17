#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Demo 07 — HMAE Full Pipeline (all 3 modes)
#
# Uploads the same images to three bucket aliases — one per HMAE mode.
# The UMRS server processes each bucket with the corresponding mode
# (reconstruct, mae, features) and writes results to output buckets.
#
# Flow: upload images ×3 → server runs 3 modes in parallel → download all
#
# Prerequisites:
#   - Services running (demos/00_start_services/)
#   - HMAE model checkpoint pre-staged on the Rivault node
# ──────────────────────────────────────────────────────────────────────────
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UMRS_CLIENT_CONTAINER="umrs-client"

# Three bucket aliases — one per HMAE mode (defined in user_rivault.json)
BUCKET_RECON="hmae-full-recon"
BUCKET_MAE="hmae-full-mae"
BUCKET_FEAT="hmae-full-feat"

# Username as defined in user_rivault.json
USERNAME="${UMRS_USERNAME:-rivault_user}"

# Test images are mounted into the container at /opt/test_data/images/ via docker-compose
CONTAINER_TEST_DATA="/opt/test_data/images"
MAX_IMAGES="${MAX_IMAGES:-3}"

RESULTS_DIR="${SCRIPT_DIR}/results"

# ── Colour helpers ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }
step()  { echo -e "\n${CYAN}═══ $* ═══${NC}"; }
run_cmd() {
    echo -e "${BLUE}[INFO]${NC}  \$ $*"
    "$@"
}

# ──────────────────────────────────────────────────────────────────────────
step "1/5  Preflight checks"
# ──────────────────────────────────────────────────────────────────────────

run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client ping || \
    fail "UMRS server not reachable. Start services first."

# Reload config so the server creates vision output buckets and grants user access
info "Reloading UMRS configuration ..."
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client reload || warn "reload failed"
sleep 3  # brief pause for server to process reload

# ──────────────────────────────────────────────────────────────────────────
step "2/5  Check test images"
# ──────────────────────────────────────────────────────────────────────────

image_count=$(podman exec "$UMRS_CLIENT_CONTAINER" \
    find "$CONTAINER_TEST_DATA" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.tiff' \) 2>/dev/null | wc -l)
if [[ "$image_count" -eq 0 ]]; then
    fail "No test images found at $CONTAINER_TEST_DATA inside the container. Place images in demos/slurm-dev/test_data/images/"
fi

# Select up to MAX_IMAGES images
CONTAINER_UPLOAD="/tmp/upload_images"
podman exec "$UMRS_CLIENT_CONTAINER" rm -rf "$CONTAINER_UPLOAD"
podman exec "$UMRS_CLIENT_CONTAINER" mkdir -p "$CONTAINER_UPLOAD"
podman exec "$UMRS_CLIENT_CONTAINER" bash -c \
    "find '$CONTAINER_TEST_DATA' -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.tiff' \) | head -n $MAX_IMAGES | xargs -I{} cp {} '$CONTAINER_UPLOAD/'"

info "Selected $(podman exec "$UMRS_CLIENT_CONTAINER" ls "$CONTAINER_UPLOAD" | wc -l) of $image_count available test images"

# ──────────────────────────────────────────────────────────────────────────
step "3/5  Upload images to all 3 mode buckets"
# ──────────────────────────────────────────────────────────────────────────

for bucket in "$BUCKET_RECON" "$BUCKET_MAE" "$BUCKET_FEAT"; do
    info "Uploading to bucket '${bucket}' ..."
    run_cmd podman exec "$UMRS_CLIENT_CONTAINER" \
        umrs-client upload_dir "$CONTAINER_UPLOAD" "$bucket" "$USERNAME"
done

ok "Images uploaded to 3 buckets. Server will process each with its HMAE mode."

# ──────────────────────────────────────────────────────────────────────────
step "4/5  Download results from all 3 modes"
# ──────────────────────────────────────────────────────────────────────────

mkdir -p "${RESULTS_DIR}/reconstruct" "${RESULTS_DIR}/mae" "${RESULTS_DIR}/features"

info "Downloading results from each mode bucket (polling until complete)..."
info "This may take several minutes on first run (Rivault + BentoML cold start)."

DOWNLOAD_START=$SECONDS

for pair in "reconstruct:${BUCKET_RECON}" "mae:${BUCKET_MAE}" "features:${BUCKET_FEAT}"; do
    mode="${pair%%:*}"
    bucket="${pair#*:}"
    info "--- Mode: ${mode} (bucket: ${bucket}) ---"

    CONTAINER_RESULTS="/tmp/demo_results_${mode}"
    podman exec "$UMRS_CLIENT_CONTAINER" rm -rf "$CONTAINER_RESULTS"
    podman exec "$UMRS_CLIENT_CONTAINER" mkdir -p "$CONTAINER_RESULTS"
    run_cmd podman exec "$UMRS_CLIENT_CONTAINER" \
        umrs-client --verbose download_results "$bucket" "$CONTAINER_RESULTS" --force-overwrite
    run_cmd podman cp "$UMRS_CLIENT_CONTAINER:$CONTAINER_RESULTS/." "$RESULTS_DIR/$mode/"
    ok "  ${mode} results downloaded."
done

DOWNLOAD_ELAPSED=$(( SECONDS - DOWNLOAD_START ))
info "Total processing + download took ${DOWNLOAD_ELAPSED}s ($(( DOWNLOAD_ELAPSED / 60 ))m $(( DOWNLOAD_ELAPSED % 60 ))s)"

ok "All results saved to ${RESULTS_DIR}/"

# ──────────────────────────────────────────────────────────────────────────
step "5/5  Summary"
# ──────────────────────────────────────────────────────────────────────────

echo ""
info "Pipeline: reconstruct + mae + features (3 independent modes)"
echo ""

# Count outputs per mode
recon_count=$(find "$RESULTS_DIR/reconstruct" -type f 2>/dev/null | wc -l || echo 0)
mae_count=$(find "$RESULTS_DIR/mae" -type f 2>/dev/null | wc -l || echo 0)
feat_count=$(find "$RESULTS_DIR/features" -type f 2>/dev/null | wc -l || echo 0)

echo -e "  Reconstruct  : ${recon_count} file(s)"
echo -e "  MAE          : ${mae_count} file(s)"
echo -e "  Features     : ${feat_count} file(s)"
echo ""

for mode_dir in reconstruct mae features; do
    if [[ -d "$RESULTS_DIR/$mode_dir" ]]; then
        echo "  ${mode_dir}/:"
        ls -1h "$RESULTS_DIR/$mode_dir/" 2>/dev/null | sed 's/^/    /'
        echo ""
    fi
done

ok "Demo 07 complete. Results in: ${RESULTS_DIR}/"
