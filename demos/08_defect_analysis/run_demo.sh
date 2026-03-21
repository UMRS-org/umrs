#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Demo 08 — Structural Defect Analysis
#
# Uploads images to UMRS via umrs-client. The UMRS server automatically
# launches the BentoML vision service on Rivault/Slurm, runs defect
# analysis (SAM3 segmentation), and writes results to the output S3 bucket.
#
# Flow: upload images → server processes → download results
#
# Prerequisites:
#   - Services running (demos/00_start_services/)
#   - Model weights pre-staged on the Rivault node
# ──────────────────────────────────────────────────────────────────────────
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UMRS_CLIENT_CONTAINER="umrs-client"

# Bucket alias as defined in user_rivault.json
BUCKET_ALIAS="defect-analysis"
# Username as defined in user_rivault.json
USERNAME="${UMRS_USERNAME:-rivault_user}"

# Test images are mounted into the container at /opt/test_data/images/ via docker-compose
# (same images used by demos 02 and 03)
CONTAINER_TEST_DATA="/opt/test_data/images"
MAX_IMAGES=2

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

# Reload config so the server creates output buckets and grants user access
info "Reloading UMRS configuration ..."
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client reload || warn "reload failed"
sleep 3  # brief pause for server to process reload

# ──────────────────────────────────────────────────────────────────────────
step "2/5  Check test images"
# ──────────────────────────────────────────────────────────────────────────

# Use test images already mounted in the container (shared with demos 02/03)
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
step "3/5  Upload images via umrs-client"
# ──────────────────────────────────────────────────────────────────────────

# Upload via umrs-client (server detects new objects and processes them automatically)
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" \
    umrs-client upload_dir "$CONTAINER_UPLOAD" "$BUCKET_ALIAS" "$USERNAME"

ok "Images uploaded. Server will launch Rivault/BentoML and process automatically."

# ──────────────────────────────────────────────────────────────────────────
step "4/5  Download results"
# ──────────────────────────────────────────────────────────────────────────

mkdir -p "${RESULTS_DIR}"

info "The download_results command will poll the server until processing is complete."
info "This may take several minutes if the server is launching Rivault/BentoML for the first time."

CONTAINER_RESULTS="/tmp/demo_results"
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" rm -rf "$CONTAINER_RESULTS"
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" mkdir -p "$CONTAINER_RESULTS"
DOWNLOAD_START=$SECONDS
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" \
    umrs-client --verbose download_results "$BUCKET_ALIAS" "$CONTAINER_RESULTS" --force-overwrite
DOWNLOAD_ELAPSED=$(( SECONDS - DOWNLOAD_START ))
info "Server processing + download took ${DOWNLOAD_ELAPSED}s ($(( DOWNLOAD_ELAPSED / 60 ))m $(( DOWNLOAD_ELAPSED % 60 ))s)"
run_cmd podman cp "$UMRS_CLIENT_CONTAINER:$CONTAINER_RESULTS/." "$RESULTS_DIR/"

ok "Results saved to ${RESULTS_DIR}/"

# ──────────────────────────────────────────────────────────────────────────
step "5/5  Summary"
# ──────────────────────────────────────────────────────────────────────────

ls -lh "${RESULTS_DIR}/"
ok "Demo 08 complete. Results in: ${RESULTS_DIR}/"
