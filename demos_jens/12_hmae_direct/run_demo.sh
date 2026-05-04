#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="hmae_jens" source "${SCRIPT_DIR}/../../demos/lib/common.sh"

JENS_CLIENT="${SCRIPT_DIR}/../lib/jens_client.py"
RESULTS_DIR="${SCRIPT_DIR}/results"

JENS_SERVER="${JENS_SERVER:-http://localhost}"
JENS_PORT="${JENS_PORT:-3000}"
JENS_TOKEN="${JENS_TOKEN:-}"
HMAE_MODE="${HMAE_MODE:-reconstruct}"
MAX_IMAGES="${MAX_IMAGES:-3}"
IMAGE_MODE="${IMAGE_MODE:-s3}"

IMAGE_DIR="${1:-${IMAGE_DIR:-${SCRIPT_DIR}/../../tmp/hmae/images}}"

banner "Jens Demo 12: HMAE Direct"
echo "Server       : ${JENS_SERVER}:${JENS_PORT}"
echo "HMAE mode    : ${HMAE_MODE}"
echo "Image mode   : ${IMAGE_MODE}"
echo "Image dir    : ${IMAGE_DIR}"
echo "Results      : ${RESULTS_DIR}"
echo ""

log "Step 1: Checking VisionFrontend reachability..."
if ! curl -sf "${JENS_SERVER}:${JENS_PORT}/readyz" >/dev/null 2>&1; then
    die "VisionFrontend not reachable at ${JENS_SERVER}:${JENS_PORT}/readyz"
fi
ok "VisionFrontend is ready"

log ""
log "Step 2: Checking Python environment..."
if ! python3 -c "import bentoml" 2>/dev/null; then
    die "bentoml not found — install with: pip install bentoml>=1.4"
fi
ok "Python + bentoml available"

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

if [[ "$IMAGE_MODE" == "presigned" || "$IMAGE_MODE" == "s3" ]]; then
    log ""
    log "Step 4: Staging images through UMRS..."
    preflight
    BUCKET_ALIAS="${BUCKET_ALIAS:-user1-hmae}"

    STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hmae-jens.XXXXXX")"
    trap 'rm -rf "$STAGING_DIR"' EXIT

    for img in "${IMAGES[@]}"; do
        ln -sf "$img" "$STAGING_DIR/"
    done
    $CLIENT upload_dir "$STAGING_DIR" "$BUCKET_ALIAS" 2>&1 | while IFS= read -r line; do log "  $line"; done

    if [[ "$IMAGE_MODE" == "presigned" ]]; then
        warn "umrs-client does not expose a dedicated presign command in this environment; using s3:// references instead."
    fi
    ok "Uploaded ${#IMAGES[@]} image(s); using s3://${BUCKET_ALIAS}/... references"
else
    log ""
    log "Step 4: Skipped (using local image paths)"
fi

if [[ "$HMAE_MODE" == "all" ]]; then
    MODES=(reconstruct mae features)
else
    MODES=("$HMAE_MODE")
fi
log ""
log "Running HMAE mode(s): ${MODES[*]}"

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
        log "  [$(( i + 1 ))/$total] $fname -> ${mode}"

        local img_ref
        case "$IMAGE_MODE" in
            presigned|s3)
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
            ok "$fname -> done"
        else
            failed=$(( failed + 1 ))
            warn "$fname -> FAILED"
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

log ""
banner "Results"
log "  Modes run : ${MODES[*]}"
log "  Output    : ${RESULTS_DIR}/"
if [[ $TOTAL_FAILED -gt 0 ]]; then
    warn "Some modes had failures — check VisionFrontend logs"
    exit 1
fi
ok "All HMAE tasks completed successfully"