#!/usr/bin/env bash
# =============================================================================
# Demo 12: HMAE Inference — Jens Direct Mode
# =============================================================================
# Calls Jens's VisionFrontend BentoML endpoint directly for HMAE inference.
# Supports all 3 HMAE modes: reconstruct, mae, features.
#
# Protocol:
#   POST /vision_task  →  {image, model="hmae", task=<mode>, out}
#   Response: pickled dict with prediction tensors, metadata etc.
#
# Prerequisites:
#   - Jens VisionFrontend running with HMAE model loaded
#   - Volume slice images (PNG) accessible by the service
#   - Python 3 with bentoml package installed
#
# Usage:
#   bash run_demo.sh                          # default: reconstruct mode
#   HMAE_MODE=mae bash run_demo.sh            # masked autoencoder mode
#   HMAE_MODE=features bash run_demo.sh       # feature extraction mode
#   HMAE_MODE=all bash run_demo.sh            # run all 3 modes sequentially
#   bash run_demo.sh /path/to/volume_slices   # custom image directory
#
# Environment:
#   JENS_SERVER   — VisionFrontend host (default: http://localhost)
#   JENS_PORT     — VisionFrontend port (default: 3000)
#   JENS_TOKEN    — Bearer token (optional)
#   HMAE_MODE     — reconstruct | mae | features | all (default: reconstruct)
#   MAX_IMAGES    — Limit number of images to process (default: 3)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="hmae_jens" source "${SCRIPT_DIR}/../../demos/lib/common.sh"

JENS_CLIENT="${SCRIPT_DIR}/../lib/jens_client.py"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Server defaults
JENS_SERVER="${JENS_SERVER:-http://localhost}"
JENS_PORT="${JENS_PORT:-3000}"
JENS_TOKEN="${JENS_TOKEN:-}"
HMAE_MODE="${HMAE_MODE:-reconstruct}"
MAX_IMAGES="${MAX_IMAGES:-3}"
IMAGE_MODE="${IMAGE_MODE:-presigned}"   # presigned | local | s3

# Image directory
IMAGE_DIR="${1:-${SCRIPT_DIR}/../../demos/03_sam3_segmentation/test_data_full}"

banner "Demo 12: HMAE Inference — Jens Direct"
echo "Server       : ${JENS_SERVER}:${JENS_PORT}"
echo "HMAE mode    : ${HMAE_MODE}"
echo "Image mode   : ${IMAGE_MODE}"
echo "Image dir    : ${IMAGE_DIR}"
echo "Results      : ${RESULTS_DIR}"
echo ""

# ---------- Step 1: Check service reachability ----------
log "Step 1: Checking VisionFrontend reachability..."
if ! curl -sf "${JENS_SERVER}:${JENS_PORT}/readyz" >/dev/null 2>&1; then
    die "VisionFrontend not reachable at ${JENS_SERVER}:${JENS_PORT}/readyz"
fi
ok "VisionFrontend is ready"

# ---------- Step 2: Check Python + bentoml ----------
log ""
log "Step 2: Checking Python environment..."
if ! python3 -c "import bentoml" 2>/dev/null; then
    die "bentoml not found — install with: pip install bentoml>=1.4"
fi
ok "Python + bentoml available"

# ---------- Step 3: Check/stage images ----------
log ""
log "Step 3: Checking images..."
if [[ ! -d "$IMAGE_DIR" ]]; then
    die "Image directory not found: $IMAGE_DIR
  Place volume slice PNGs in the directory, or pass a path:
    bash run_demo.sh /path/to/volume_slices/"
fi

IMAGES=()
while IFS= read -r img; do
    IMAGES+=("$img")
done < <(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.tiff" \) | sort | head -n "$MAX_IMAGES")

if [[ ${#IMAGES[@]} -eq 0 ]]; then
    die "No image files found in $IMAGE_DIR"
fi
log "  Selected ${#IMAGES[@]} image(s)"

# ---------- Step 4: Generate presigned URLs (if needed) ----------
if [[ "$IMAGE_MODE" == "presigned" ]]; then
    log ""
    log "Step 4: Uploading images and generating presigned URLs..."
    BUCKET_ALIAS="${BUCKET_ALIAS:-user1-hmae}"

    STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hmae-jens.XXXXXX")"
    trap 'rm -rf "$STAGING_DIR"' EXIT

    for img in "${IMAGES[@]}"; do
        ln -sf "$img" "$STAGING_DIR/"
    done
    $CLIENT upload_dir "$STAGING_DIR" "$BUCKET_ALIAS" 2>&1 | while IFS= read -r l; do log "  $l"; done

    PRESIGNED_URLS=()
    for img in "${IMAGES[@]}"; do
        fname="$(basename "$img")"
        url=$($CLIENT presign "$BUCKET_ALIAS" "$fname" 2>/dev/null || echo "")
        if [[ -z "$url" ]]; then
            warn "Could not generate presigned URL for $fname — using s3:// fallback"
            url="s3://${BUCKET_ALIAS}/${fname}"
        fi
        PRESIGNED_URLS+=("$url")
    done
    ok "Generated ${#PRESIGNED_URLS[@]} presigned URL(s)"
else
    log ""
    log "Step 4: Skipped (using ${IMAGE_MODE} mode)"
fi

# ---------- Step 5: Determine modes to run ----------
if [[ "$HMAE_MODE" == "all" ]]; then
    MODES=(reconstruct mae features)
else
    MODES=("$HMAE_MODE")
fi
log ""
log "Running HMAE mode(s): ${MODES[*]}"

# ---------- Step 6: Submit inference tasks ----------
run_mode() {
    local mode="$1"
    log ""
    banner "HMAE mode: ${mode}"
    local mode_dir="${RESULTS_DIR}/${mode}"
    mkdir -p "$mode_dir"

    local total=${#IMAGES[@]}
    local passed=0
    local failed=0

    for i in "${!IMAGES[@]}"; do
        local img="${IMAGES[$i]}"
        local fname
        fname="$(basename "$img")"
        log "  [$(( i + 1 ))/$total] $fname → ${mode}"

        # Determine image reference
        local img_ref
        case "$IMAGE_MODE" in
            presigned)
                img_ref="${PRESIGNED_URLS[$i]}"
                ;;
            s3)
                img_ref="s3://${BUCKET_ALIAS:-user1-hmae}/${fname}"
                ;;
            local)
                img_ref="$img"
                ;;
            *)
                die "Unknown IMAGE_MODE: $IMAGE_MODE"
                ;;
        esac

        local token_args=()
        if [[ -n "$JENS_TOKEN" ]]; then
            token_args=(--token "$JENS_TOKEN")
        fi

        if python3 "$JENS_CLIENT" \
            --server "$JENS_SERVER" \
            --port "$JENS_PORT" \
            "${token_args[@]}" \
            --model hmae \
            --task "$mode" \
            --image "$img_ref" \
            --output-dir "$mode_dir" \
            --timeout 300; then
            passed=$(( passed + 1 ))
            ok "$fname → done"
        else
            failed=$(( failed + 1 ))
            warn "$fname → FAILED"
        fi
    done

    echo "  ${mode}: ${passed}/${total} passed, ${failed}/${total} failed"
    return $failed
}

TOTAL_FAILED=0
for mode in "${MODES[@]}"; do
    if ! run_mode "$mode"; then
        TOTAL_FAILED=$(( TOTAL_FAILED + 1 ))
    fi
done

# ---------- Summary ----------
log ""
banner "Results"
log "  Modes run : ${MODES[*]}"
log "  Output    : ${RESULTS_DIR}/"
if [[ $TOTAL_FAILED -gt 0 ]]; then
    warn "Some modes had failures — check VisionFrontend logs"
    exit 1
fi
ok "All HMAE tasks completed successfully"
