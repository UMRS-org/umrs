#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Demo 10 — HMAE Full Pipeline
#
# Runs all 3 HMAE tasks (reconstruction, mae_reconstruction, feature_extraction)
# sequentially, each with its own bucket alias.  Uploads volume slice images
# to UMRS → server runs HMAE via BentoML on Slurm → downloads results
# (PNG + pkl for each task, plus .pt for feature_extraction).
#
# Bucket aliases (configured in user1.json):
#   user1-hmae           → reconstruction
#   user1-hmae-mae       → mae_reconstruction
#   user1-hmae-features  → feature_extraction
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running with HMAE Inference experiences (all 3 tasks)
#   - BentoML vision service available on compute node
#
# Note: Running all 3 tasks sequentially may cause CUDA out-of-memory on
# the 3rd task if the GPU does not release memory between tasks.  If this
# happens, re-run with fewer tasks or restart the BentoML service.
# ──────────────────────────────────────────────────────────────────────────
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="hmae_full" source "${SCRIPT_DIR}/../lib/common.sh"

MAX_IMAGES="${MAX_IMAGES:-3}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-}"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Test images: use SAM3 demo's test_data_full
IMAGE_DIR="${IMAGE_DIR:-${SCRIPT_DIR}/../../demos/03_sam3_segmentation/test_data_full}"

# All 3 HMAE tasks and their bucket aliases
HMAE_TASKS=("reconstruction" "mae_reconstruction" "feature_extraction")
HMAE_BUCKETS=("user1-hmae" "user1-hmae-mae" "user1-hmae-features")

# S3 output bucket names (derived from user1.json naming convention)
# Format: <username>.<xp_name>.<bucket_alias>-output  (lowercase, spaces→hyphens)
HMAE_OUTPUT_BUCKETS=(
    "user1.hmae-inference.user1-hmae-output"
    "user1.hmae-inference.user1-hmae-mae-output"
    "user1.hmae-inference.user1-hmae-features-output"
)
HMAE_INPUT_BUCKETS=(
    "user1.hmae-inference.user1-hmae"
    "user1.hmae-inference.user1-hmae-mae"
    "user1.hmae-inference.user1-hmae-features"
)

# Helper: clear all objects from an S3 bucket (via local boto3 + s3cfg)
clear_s3_bucket() {
    local bucket="$1"
    python3 - "$bucket" <<'PY'
import sys, configparser, os, boto3
bucket = sys.argv[1]
cfg = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
s3cfg = os.path.expanduser("~/.config/umrs/s3cfg")
cfg.read(s3cfg)
protocol = "https" if cfg.getboolean("default", "use_https", fallback=False) else "http"
endpoint = f"{protocol}://{cfg.get('default', 'host_base')}"
s3 = boto3.client("s3", endpoint_url=endpoint,
    aws_access_key_id=cfg.get("default", "access_key"),
    aws_secret_access_key=cfg.get("default", "secret_key"))
try:
    objs = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        objs.extend(page.get("Contents", []))
    if objs:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": o["Key"]} for o in objs]})
    print(f"Cleared {len(objs)} object(s) from {bucket}")
except s3.exceptions.NoSuchBucket:
    print(f"Bucket {bucket} does not exist yet (will be created on first upload)")
except Exception as e:
    print(f"Warning: could not clear {bucket}: {e}")
PY
}

banner "Demo 10: HMAE Full Pipeline (all 3 tasks)"
echo "User config  : ${CONF}"
echo "Image dir    : ${IMAGE_DIR}"
echo "Tasks        : ${HMAE_TASKS[*]}"
echo "Force recomp : ${FORCE_RECOMPUTE:-no}"
echo ""

# ──────────────────────────────────────────────────────────────────────────
log "Step 1: Pre-flight checks..."
preflight
$CLIENT reload 2>/dev/null || warn "reload returned non-zero"
sleep 30  # server needs time to restart after reload

# ──────────────────────────────────────────────────────────────────────────
log ""
log "Step 2: Checking test images..."
if [[ ! -d "$IMAGE_DIR" ]]; then
    die "Image directory not found: $IMAGE_DIR"
fi
IMAGE_COUNT=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.tiff" \) | wc -l)
if [[ "$IMAGE_COUNT" -eq 0 ]]; then
    die "No image files found in $IMAGE_DIR"
fi

# Stage a subset
UPLOAD_DIR="$IMAGE_DIR"
if [[ "$IMAGE_COUNT" -gt "$MAX_IMAGES" ]]; then
    UPLOAD_DIR="${SCRIPT_DIR}/.upload_staging"
    rm -rf "$UPLOAD_DIR"
    mkdir -p "$UPLOAD_DIR"
    find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.tiff" \) \
        | head -n "$MAX_IMAGES" | while read -r f; do ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$UPLOAD_DIR/$(basename "$f")"; done
    log "  Staged ${MAX_IMAGES} of ${IMAGE_COUNT} images for upload"
else
    log "  Using all ${IMAGE_COUNT} images"
fi

# ──────────────────────────────────────────────────────────────────────────
# Run each HMAE task sequentially
for i in "${!HMAE_TASKS[@]}"; do
    TASK="${HMAE_TASKS[$i]}"
    BUCKET="${HMAE_BUCKETS[$i]}"
    OUTPUT_BUCKET="${HMAE_OUTPUT_BUCKETS[$i]}"
    INPUT_BUCKET="${HMAE_INPUT_BUCKETS[$i]}"
    TASK_RESULTS_DIR="${RESULTS_DIR}/${TASK}"

    log ""
    banner "Task ${TASK} (bucket: ${BUCKET})"

    # Clear S3 buckets to ensure fresh results (no stale outputs)
    log "Step 3.${i}a: Clearing S3 buckets for fresh computation..."
    clear_s3_bucket "$OUTPUT_BUCKET"
    clear_s3_bucket "$INPUT_BUCKET"

    log "Step 3.${i}b: Uploading images to bucket '${BUCKET}'..."
    $CLIENT upload_dir "$UPLOAD_DIR" "$BUCKET"
    ok "Images uploaded for ${TASK} — server will process automatically"

    if [[ -n "$FORCE_RECOMPUTE" ]]; then
        log "FORCE_RECOMPUTE set — triggering full recompute..."
        $CLIENT recompute || warn "recompute returned non-zero"
    fi
    UPLOAD_DONE=$SECONDS

    log "Step 4.${i}: Downloading ${TASK} results..."
    log "  (This may take several minutes for HMAE model loading + inference)"
    rm -rf "$TASK_RESULTS_DIR"
    mkdir -p "$TASK_RESULTS_DIR"
    $CLIENT download_results "$BUCKET" "$TASK_RESULTS_DIR" --force-overwrite
    DOWNLOAD_ELAPSED=$(( SECONDS - UPLOAD_DONE ))
    log "  ${TASK} processing + download took ${DOWNLOAD_ELAPSED}s"

    RESULT_COUNT=$(find "$TASK_RESULTS_DIR" -type f 2>/dev/null | wc -l)
    ok "${TASK}: ${RESULT_COUNT} file(s) downloaded to ${TASK_RESULTS_DIR}"
    ls -lh "$TASK_RESULTS_DIR/" 2>/dev/null || true
    log ""
done

# ──────────────────────────────────────────────────────────────────────────
log ""
banner "Demo 10 Complete"
TOTAL_COUNT=$(find "$RESULTS_DIR" -type f 2>/dev/null | wc -l)
log "  ${TOTAL_COUNT} total file(s) across all tasks in results/"
for TASK in "${HMAE_TASKS[@]}"; do
    TASK_DIR="${RESULTS_DIR}/${TASK}"
    if [[ -d "$TASK_DIR" ]]; then
        TASK_COUNT=$(find "$TASK_DIR" -type f 2>/dev/null | wc -l)
        log "  ${TASK}: ${TASK_COUNT} file(s)"
    fi
done
log "Results in: ${RESULTS_DIR}/"
