#!/usr/bin/env bash
# =============================================================================
# Demo 04: HMAE Inference
# =============================================================================
# Uploads volume slice images → server runs HMAE via BentoML on Slurm →
# downloads reconstruction results.
#
# HMAE inference engine: tmp/jens_stack/hmae-infer/ (see tmp/jens_stack/hmae-infer/README.md)
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running with HMAE Inference experience
#   - BentoML vision service available on compute node
#
# Bucket alias: user1-hmae  (from user config, need_output_bucket: true)
#
# Usage:
#   bash run_demo.sh [image_dir]
#
# Optional env vars:
#   UMRS_USER_CONF        — path to user config (default: auto-discovered)
#   HMAE_LOCAL_MODEL_DIR  — if set, registers the model before inference
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="hmae_demo" source "${SCRIPT_DIR}/../lib/common.sh"

BUCKET_ALIAS="${HMAE_BUCKET:-user1-hmae}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-}"
RESULTS_DIR="${SCRIPT_DIR}/results"
REMOTE_HMAE_MODEL_PATH="${UMRS_REMOTE_HMAE_MODEL_PATH:-}"

# S3 bucket names (derived from user1.json naming convention)
INPUT_S3_BUCKET="user1.hmae-inference.${BUCKET_ALIAS}"
OUTPUT_S3_BUCKET="user1.hmae-inference.${BUCKET_ALIAS}-output"

# Default image dir: use SAM3 demo's test_data_full (shared volume slices)
IMAGE_DIR="${1:-${SCRIPT_DIR}/../../demos/03_sam3_segmentation/test_data_full}"

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

banner "Demo 04: HMAE Inference"
echo "User config  : ${CONF}"
echo "Bucket alias : ${BUCKET_ALIAS}"
echo "Image dir    : ${IMAGE_DIR}"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

# ---------- Step 2: Optional model registration ----------
if [[ -n "${HMAE_LOCAL_MODEL_DIR:-}" ]]; then
    log ""
    if [[ -n "$REMOTE_HMAE_MODEL_PATH" ]]; then
        log "Step 2: Uploading HMAE model from ${HMAE_LOCAL_MODEL_DIR}..."
        $CLIENT model upload "${HMAE_LOCAL_MODEL_DIR}" \
            --name "hmae-pretrained" \
            --type embedding --scope user \
            --description "HMAE ViT pretrained checkpoint" \
            --register-path "$REMOTE_HMAE_MODEL_PATH" \
            --force || warn "model upload returned non-zero"
        ok "Model uploaded"
    else
        warn "Skipping model upload — set UMRS_REMOTE_HMAE_MODEL_PATH to a server-visible register path"
    fi
else
    log ""
    log "Step 2: Skipping model registration (set HMAE_LOCAL_MODEL_DIR to register)"
fi

# ---------- Step 3: Check sample images ----------
log ""
log "Step 3: Checking test images..."
if [[ ! -d "${IMAGE_DIR}" ]]; then
    die "Image directory not found: ${IMAGE_DIR}
  Place volume slice PNGs in the directory, or pass a path:
    bash run_demo.sh /path/to/volume_slices/"
fi
PNG_COUNT=$(find "${IMAGE_DIR}" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
if [[ "${PNG_COUNT}" -eq 0 ]]; then
    die "No .png files found in ${IMAGE_DIR}"
fi
log "  Found ${PNG_COUNT} PNG file(s)"

# ---------- Step 4: Clear stale results & Upload images ----------
log ""
log "Step 4a: Clearing S3 buckets for fresh computation..."
clear_s3_bucket "$OUTPUT_S3_BUCKET"
clear_s3_bucket "$INPUT_S3_BUCKET"

log "Step 4b: Uploading volume slices to bucket '${BUCKET_ALIAS}'..."
$CLIENT upload_dir "${IMAGE_DIR}" "$BUCKET_ALIAS"
ok "Images uploaded — server will process automatically"

if [[ -n "$FORCE_RECOMPUTE" ]]; then
    log "FORCE_RECOMPUTE set — triggering full recompute..."
    admin_recompute || warn "recompute returned non-zero"
fi
UPLOAD_DONE=$SECONDS

# ---------- Step 5: Wait & Download results ----------
log ""
log "Step 5: Waiting for HMAE processing and downloading results..."
log "  (This may take several minutes for model loading + inference)"
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
$CLIENT download_results "$BUCKET_ALIAS" "$RESULTS_DIR" --force-overwrite
DOWNLOAD_ELAPSED=$(( SECONDS - UPLOAD_DONE ))
log "  Processing + download took ${DOWNLOAD_ELAPSED}s"
ok "Results saved to ${RESULTS_DIR}/"

# ---------- Summary ----------
log ""
banner "Demo 04 Complete"
ls -lh "$RESULTS_DIR/" 2>/dev/null || true
log "Results in: ${RESULTS_DIR}/"
