#!/usr/bin/env bash
# =============================================================================
# Demo 11: SAM3 Segmentation — Jens Direct Mode
# =============================================================================
# Calls Jens's VisionFrontend BentoML endpoint directly (bypassing the
# UMRS generic vision pipeline).  This is the fastest path for SAM3
# segmentation when the VisionFrontend service is already running.
#
# Protocol:
#   POST /vision_task  →  {image, model="sam3", task=<prompt>, out}
#   Response: pickled dict with masks, boxes, scores etc.
#
# Prerequisites:
#   - Jens VisionFrontend running on a BentoML port (default: 3000)
#   - Images accessible by the service (local, HTTP presigned, or s3://)
#   - Python 3 with bentoml package installed
#
# Usage:
#   bash run_demo.sh                    # default: presigned S3 via UMRS
#   JENS_SERVER=http://compute1 JENS_PORT=3000 bash run_demo.sh
#   bash run_demo.sh /path/to/images    # local image directory
#
# Environment:
#   JENS_SERVER   — VisionFrontend host (default: http://localhost)
#   JENS_PORT     — VisionFrontend port (default: 3000)
#   JENS_TOKEN    — Bearer token (optional; set NO_AUTH=1 on service to skip)
#   SAM3_PROMPT   — Object to segment (default: "all")
#   MAX_IMAGES    — Limit number of images to process (default: 3)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="sam3_jens" source "${SCRIPT_DIR}/../../demos/lib/common.sh"

JENS_CLIENT="${SCRIPT_DIR}/../lib/jens_client.py"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Server defaults
JENS_SERVER="${JENS_SERVER:-http://localhost}"
JENS_PORT="${JENS_PORT:-3000}"
JENS_TOKEN="${JENS_TOKEN:-}"
SAM3_PROMPT="${SAM3_PROMPT:-all}"
MAX_IMAGES="${MAX_IMAGES:-3}"
IMAGE_MODE="${IMAGE_MODE:-presigned}"   # presigned | local | s3

# Image directory (local files for presigned or local mode)
IMAGE_DIR="${1:-${SCRIPT_DIR}/../../demos/03_sam3_segmentation/test_data_full}"

banner "Demo 11: SAM3 Segmentation — Jens Direct"
echo "Server       : ${JENS_SERVER}:${JENS_PORT}"
echo "Prompt       : ${SAM3_PROMPT}"
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
    die "Image directory not found: $IMAGE_DIR"
fi

IMAGES=()
while IFS= read -r img; do
    IMAGES+=("$img")
done < <(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.tiff" \) | sort | head -n "$MAX_IMAGES")

if [[ ${#IMAGES[@]} -eq 0 ]]; then
    die "No image files found in $IMAGE_DIR"
fi
log "  Selected ${#IMAGES[@]} image(s)"

# ---------- Step 4: Generate presigned URLs (if needed) ----------
if [[ "$IMAGE_MODE" == "presigned" ]]; then
    log ""
    log "Step 4: Uploading images and generating presigned URLs..."
    BUCKET_ALIAS="${BUCKET_ALIAS:-user1-sam3}"

    # Upload images via UMRS client
    STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sam3-jens.XXXXXX")"
    trap 'rm -rf "$STAGING_DIR"' EXIT

    for img in "${IMAGES[@]}"; do
        ln -sf "$img" "$STAGING_DIR/"
    done
    $CLIENT upload_dir "$STAGING_DIR" "$BUCKET_ALIAS" 2>&1 | while IFS= read -r l; do log "  $l"; done

    # Generate presigned URLs for each uploaded file
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

# ---------- Step 5: Submit inference tasks ----------
log ""
log "Step 5: Submitting SAM3 tasks to VisionFrontend..."
mkdir -p "$RESULTS_DIR"

TOTAL=${#IMAGES[@]}
PASSED=0
FAILED=0

for i in "${!IMAGES[@]}"; do
    img="${IMAGES[$i]}"
    fname="$(basename "$img")"
    log "  [$(( i + 1 ))/$TOTAL] $fname"

    # Determine image reference for the service
    case "$IMAGE_MODE" in
        presigned)
            img_ref="${PRESIGNED_URLS[$i]}"
            ;;
        s3)
            img_ref="s3://${BUCKET_ALIAS:-user1-sam3}/${fname}"
            ;;
        local)
            img_ref="$img"
            ;;
        *)
            die "Unknown IMAGE_MODE: $IMAGE_MODE"
            ;;
    esac

    token_args=()
    if [[ -n "$JENS_TOKEN" ]]; then
        token_args=(--token "$JENS_TOKEN")
    fi

    if python3 "$JENS_CLIENT" \
        --server "$JENS_SERVER" \
        --port "$JENS_PORT" \
        "${token_args[@]}" \
        --model sam3 \
        --task "$SAM3_PROMPT" \
        --image "$img_ref" \
        --output-dir "$RESULTS_DIR" \
        --timeout 300; then
        PASSED=$(( PASSED + 1 ))
        ok "$fname → done"
    else
        FAILED=$(( FAILED + 1 ))
        warn "$fname → FAILED"
    fi
done

# ---------- Summary ----------
log ""
banner "Results"
log "  Passed: ${PASSED}/${TOTAL}"
log "  Failed: ${FAILED}/${TOTAL}"
log "  Output: ${RESULTS_DIR}/"
if [[ $FAILED -gt 0 ]]; then
    warn "Some tasks failed — check VisionFrontend logs"
    exit 1
fi
ok "All SAM3 tasks completed successfully"
