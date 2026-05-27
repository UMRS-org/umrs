#!/usr/bin/env bash
# =============================================================================
# Demo 04: Vision Model Inference
# =============================================================================
# Generic vision model inference demo.  Demonstrates uploading images and
# running a BentoML vision model on HPC via the UMRS pipeline.
#
# By default this demo runs SAM3 segmentation (same model as Demo 03) but
# illustrates how ANY vision model registered in the model registry can be
# invoked through the same interface.
#
# To plug in a different model:
#   1. Register it in a user-visible scope or ask the site operator to pre-stage it
#   2. Use a user bucket / experience already configured for that model
#   3. Run this demo against that bucket alias
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - BentoML vision service available on compute node
#
# Usage:
#   bash run_demo.sh                         # sample images
#   bash run_demo.sh /path/to/images         # your own images
#   VISION_BUCKET=my-bucket bash run_demo.sh # custom bucket alias
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="vision_infer" source "${SCRIPT_DIR}/../lib/common.sh"

# Ensure Vision Inference experience is in user config (idempotent).
ensure_experience "${SCRIPT_DIR}/experience.json" --update

BUCKET_ALIAS="${VISION_BUCKET:-demo-images}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-}"
MAX_IMAGES="${MAX_IMAGES:-3}"
IMAGE_DIR="${1:-${SCRIPT_DIR}/../sample_data/images}"
RESULTS_DIR="${SCRIPT_DIR}/results"
VISION_MODEL_NAME="${VISION_MODEL_NAME:-sam3.1}"
REQUIRE_VISIBLE_MODEL="${REQUIRE_VISIBLE_MODEL:-1}"
VISION_REQUIRE_RESULTS="${VISION_REQUIRE_RESULTS:-0}"
VISION_S3CFG="${VISION_S3CFG:-$(CONF="${CONF}" python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["CONF"]).expanduser()
if path.exists():
    print(json.loads(path.read_text()).get("s3_configuration_file", ""))
PY
)}"
VISION_S3CFG="${VISION_S3CFG:-${HOME}/.config/umrs/s3cfg}"

read -r INPUT_S3_BUCKET OUTPUT_S3_BUCKET < <(CONF="${CONF}" BUCKET_ALIAS="${BUCKET_ALIAS}" python3 - <<'PY'
import json
import os
from pathlib import Path

def bucket_name(username, xp_name, alias, output=False):
    value = f"{username}.{xp_name}.{alias}"
    if output:
        value += "-output"
    return value.lower().replace(" ", "-").replace("_", "-")

conf = json.loads(Path(os.environ["CONF"]).expanduser().read_text())
alias = os.environ["BUCKET_ALIAS"]
username = conf.get("username", "demo_user")
for xp in conf.get("experiences", []):
    for modality in xp.get("modalities", []):
        if (modality.get("s3") or {}).get("bucket_alias") == alias:
            xp_name = xp.get("xp_name", "Vision Inference")
            print(bucket_name(username, xp_name, alias), bucket_name(username, xp_name, alias, output=True))
            raise SystemExit(0)
raise SystemExit(f"bucket alias not found in {os.environ['CONF']}: {alias}")
PY
)

clear_s3_bucket() {
    local bucket="$1"
    python3 - "$bucket" "$VISION_S3CFG" <<'PY'
import sys, configparser, os, boto3
bucket = sys.argv[1]
s3cfg = os.path.expanduser(sys.argv[2])
cfg = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
read_files = cfg.read(s3cfg)
if not read_files:
    raise SystemExit(f"S3 config not found: {s3cfg}")
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

banner "Demo 04: Vision Model Inference"
echo "Config       : ${CONF}"
echo "Bucket alias : ${BUCKET_ALIAS}"
echo "Image dir    : ${IMAGE_DIR}"
echo ""

model_visible() {
    local scope="$1" name="$2"
    $CLIENT --output json model list --scope "$scope" 2>/dev/null | grep -q "\"name\": \"${name}\""
}

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

# General-user path: rely on the deployed site configuration and user-visible buckets.
log "Using the deployed vision experience as-is (no admin reload in demo mode)."

# ---------- Step 2: Check model availability ----------
if [[ "$REQUIRE_VISIBLE_MODEL" != "0" ]]; then
    log ""
    log "Step 2: Checking vision model availability..."
    if model_visible system "$VISION_MODEL_NAME" || model_visible user "$VISION_MODEL_NAME"; then
        ok "Vision model already visible in the registry: ${VISION_MODEL_NAME}"
    else
        die "Vision model '${VISION_MODEL_NAME}' is not visible. Ask the site operator to pre-stage it, switch VISION_MODEL_NAME, or set REQUIRE_VISIBLE_MODEL=0 when you are targeting a site-specific custom bucket that is already wired server-side."
    fi
fi

# ---------- Step 3: Prepare images ----------
log ""
log "Step 3: Checking images..."
if [[ ! -d "$IMAGE_DIR" ]]; then
    die "Image directory not found: $IMAGE_DIR
  Place .jpg/.png/.tiff images in demos/sample_data/images/ or provide a path."
fi
IMAGE_COUNT=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.tiff" \) | wc -l | tr -d ' ')
if [[ "$IMAGE_COUNT" -eq 0 ]]; then
    die "No image files found in $IMAGE_DIR"
fi
log "  Found ${IMAGE_COUNT} image(s)"

# Stage a subset if MAX_IMAGES < total
UPLOAD_DIR="$IMAGE_DIR"
if [[ "$IMAGE_COUNT" -gt "$MAX_IMAGES" ]]; then
    UPLOAD_DIR="${SCRIPT_DIR}/.upload_staging"
    rm -rf "$UPLOAD_DIR"
    mkdir -p "$UPLOAD_DIR"
    find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.tiff" \) | sort \
        | head -n "$MAX_IMAGES" | while read -r f; do ln -s "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$UPLOAD_DIR/"; done
    log "  Staged ${MAX_IMAGES} of ${IMAGE_COUNT} images for upload"
else
    log "  Using all ${IMAGE_COUNT} images"
fi

# ---------- Step 4: Clear stale data & Upload images ----------
log ""
log "Step 4a: Clearing S3 buckets for fresh vision computation..."
clear_s3_bucket "$OUTPUT_S3_BUCKET"
clear_s3_bucket "$INPUT_S3_BUCKET"

log "Step 4b: Uploading images to bucket '${BUCKET_ALIAS}'..."
$CLIENT upload_dir "$UPLOAD_DIR" "$BUCKET_ALIAS"
ok "Images uploaded — server will dispatch to vision model"

if [[ -n "$FORCE_RECOMPUTE" ]]; then
    log "FORCE_RECOMPUTE set — triggering full recompute..."
    maybe_admin_recompute || warn "recompute returned non-zero"
fi
UPLOAD_DONE=$SECONDS

# ---------- Step 5: Wait & Download results ----------
log ""
log "Step 5: Waiting for processing and downloading results..."
log "  (May take several minutes if launching the vision service for the first time)"
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
if download_results_bounded "$BUCKET_ALIAS" "$RESULTS_DIR" --force-overwrite; then
    DOWNLOAD_ELAPSED=$(( SECONDS - UPLOAD_DONE ))
    log "  Processing + download took ${DOWNLOAD_ELAPSED}s"
    if check_result_files "$RESULTS_DIR" "$VISION_REQUIRE_RESULTS" "Vision results"; then
        case "$VISION_MODEL_NAME" in
            sam3*|SAM3*)
                check_sam3_result_objects "$RESULTS_DIR" "$VISION_REQUIRE_RESULTS" "SAM3 detections"
                ;;
        esac
    fi
else
    if [[ "$VISION_REQUIRE_RESULTS" = "1" ]]; then
        die "Could not download vision results"
    fi
    warn "Could not download vision results — check server logs or set VISION_REQUIRE_RESULTS=1 for strict validation"
fi

# ---------- Summary ----------
log ""
banner "Demo 04 Complete"
ls -lh "$RESULTS_DIR/" 2>/dev/null || true
log "Results in: ${RESULTS_DIR}/"
log ""
log "To use a different vision model:"
log "  1. Register or stage the model in a user-visible scope supported by your site"
log "  2. Use a bucket/experience that already points at that model"
log "  3. Re-run: VISION_BUCKET=<alias> VISION_MODEL_NAME=<name> bash run_demo.sh"
