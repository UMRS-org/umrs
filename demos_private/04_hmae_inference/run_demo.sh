#!/usr/bin/env bash
# =============================================================================
# Demo 04: HMAE Inference
# =============================================================================
# Uploads volume slice images → server runs HMAE via BentoML on Slurm →
# downloads reconstruction results.
#
# HMAE inference engine: tmp/hmae-infer/ (see tmp/hmae-infer/README.md)
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running with HMAE Inference experience
#   - BentoML vision service available on compute node
#
# Bucket alias: demo-hmae  (from experience.json fragment, need_output_bucket: true)
#
# Usage:
#   bash run_demo.sh [image_dir]
#
# Optional env vars:
#   UMRS_USER_CONF        — path to user config (default: auto-discovered)
#   HMAE_ASSET_ROOT       — local HMAE data/checkpoint root (default: umrs/tmp/hmae)
#   HMAE_INFER_DIR        — local hmae-infer checkout (default: umrs/tmp/hmae-infer)
#   HMAE_REGISTER_MODEL   — set to 1 to register HMAE_LOCAL_MODEL_DIR
#   FORCE_RECOMPUTE       — default: 1; set 0 to skip admin recompute attempt
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="hmae_demo" source "${SCRIPT_DIR}/../lib/common.sh"

# Ensure HMAE Inference experience is in user config (idempotent).
ensure_experience "${SCRIPT_DIR}/experience.json" --update

BUCKET_ALIAS="${HMAE_BUCKET:-demo-hmae}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-1}"
MAX_IMAGES="${MAX_IMAGES:-3}"
RESULTS_DIR="${SCRIPT_DIR}/results"
REMOTE_HMAE_MODEL_PATH="${UMRS_REMOTE_HMAE_MODEL_PATH:-}"
HMAE_ASSET_ROOT="${HMAE_ASSET_ROOT:-${SCRIPT_DIR}/../../tmp/hmae}"
HMAE_INFER_DIR="${HMAE_INFER_DIR:-${SCRIPT_DIR}/../../tmp/hmae-infer}"
HMAE_LOCAL_MODEL_DIR="${HMAE_LOCAL_MODEL_DIR:-${HMAE_ASSET_ROOT}}"
HMAE_LOCAL_CHECKPOINT="${HMAE_LOCAL_CHECKPOINT:-${HMAE_ASSET_ROOT}/latest_checkpoint.pth}"
HMAE_LOCAL_CONFIG="${HMAE_LOCAL_CONFIG:-${HMAE_INFER_DIR}/configs/infer_hmae_vit_b.yaml}"
HMAE_REGISTER_MODEL="${HMAE_REGISTER_MODEL:-0}"
HMAE_REQUIRE_RESULTS="${HMAE_REQUIRE_RESULTS:-0}"
HMAE_S3CFG="${HMAE_S3CFG:-$(CONF="${CONF}" python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["CONF"]).expanduser()
if path.exists():
    print(json.loads(path.read_text()).get("s3_configuration_file", ""))
PY
)}"
HMAE_S3CFG="${HMAE_S3CFG:-${HOME}/.config/umrs/s3cfg}"

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
            xp_name = xp.get("xp_name", "HMAE Inference")
            print(bucket_name(username, xp_name, alias), bucket_name(username, xp_name, alias, output=True))
            raise SystemExit(0)
raise SystemExit(f"bucket alias not found in {os.environ['CONF']}: {alias}")
PY
)

# Default image dir: use the Spring-8 concrete volume slices for this demo.
IMAGE_DIR="${1:-${IMAGE_DIR:-${HMAE_ASSET_ROOT}/spring8concrete}}"

# Helper: clear all objects from an S3 bucket (via local boto3 + s3cfg)
clear_s3_bucket() {
    local bucket="$1"
    python3 - "$bucket" "$HMAE_S3CFG" <<'PY'
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

banner "Demo 04: HMAE Inference"
echo "User config  : ${CONF}"
echo "Bucket alias : ${BUCKET_ALIAS}"
echo "Image dir    : ${IMAGE_DIR}"
echo "Max images   : ${MAX_IMAGES}"
echo "HMAE assets  : ${HMAE_ASSET_ROOT}"
echo "HMAE config  : ${HMAE_LOCAL_CONFIG}"
echo "Checkpoint   : ${HMAE_LOCAL_CHECKPOINT}"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

# ---------- Step 2: Local HMAE assets / optional model registration ----------
log ""
log "Step 2: Checking local HMAE assets..."
[[ -d "$HMAE_INFER_DIR" ]] || die "HMAE inference checkout not found: ${HMAE_INFER_DIR}"
[[ -f "$HMAE_LOCAL_CONFIG" ]] || die "HMAE config not found: ${HMAE_LOCAL_CONFIG}"
[[ -f "$HMAE_LOCAL_CHECKPOINT" ]] || die "HMAE checkpoint not found: ${HMAE_LOCAL_CHECKPOINT}"
ok "Using hmae-infer config and checkpoint from umrs/tmp"

if [[ "$HMAE_REGISTER_MODEL" = "1" ]]; then
    if [[ -n "$REMOTE_HMAE_MODEL_PATH" ]]; then
        log "Registering HMAE model from ${HMAE_LOCAL_MODEL_DIR}..."
        $CLIENT model upload "${HMAE_LOCAL_MODEL_DIR}" \
            --name "hmae-pretrained" \
            --type vision --scope user \
            --description "HMAE ViT pretrained checkpoint" \
            --register-path "$REMOTE_HMAE_MODEL_PATH" \
            --force || warn "model upload returned non-zero"
        ok "Model uploaded"
    else
        warn "Skipping model upload — set UMRS_REMOTE_HMAE_MODEL_PATH to a server-visible register path"
    fi
else
    log "Model registration skipped (set HMAE_REGISTER_MODEL=1 to register ${HMAE_LOCAL_MODEL_DIR})"
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

UPLOAD_DIR="$IMAGE_DIR"
if [[ "$PNG_COUNT" -gt "$MAX_IMAGES" ]]; then
    UPLOAD_DIR="${SCRIPT_DIR}/.upload_staging"
    rm -rf "$UPLOAD_DIR"
    mkdir -p "$UPLOAD_DIR"
    find "$IMAGE_DIR" -maxdepth 1 -type f -name "*.png" | sort \
        | head -n "$MAX_IMAGES" | while read -r f; do ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$UPLOAD_DIR/$(basename "$f")"; done
    log "  Staged ${MAX_IMAGES} of ${PNG_COUNT} images for upload"
else
    log "  Using all ${PNG_COUNT} images"
fi

# ---------- Step 4: Clear stale results & Upload images ----------
log ""
log "Step 4a: Clearing S3 buckets for fresh computation..."
clear_s3_bucket "$OUTPUT_S3_BUCKET"
clear_s3_bucket "$INPUT_S3_BUCKET"

log "Step 4b: Uploading volume slices to bucket '${BUCKET_ALIAS}'..."
$CLIENT upload_dir "${UPLOAD_DIR}" "$BUCKET_ALIAS"
ok "Images uploaded — server will process automatically"

if [[ "$FORCE_RECOMPUTE" != "0" ]]; then
    log "FORCE_RECOMPUTE set — triggering full recompute..."
    maybe_admin_recompute || warn "recompute returned non-zero"
fi
UPLOAD_DONE=$SECONDS

# ---------- Step 5: Wait & Download results ----------
log ""
log "Step 5: Waiting for HMAE processing and downloading results..."
log "  (This may take several minutes for model loading + inference)"
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
if download_results_bounded "$BUCKET_ALIAS" "$RESULTS_DIR" --force-overwrite; then
    DOWNLOAD_ELAPSED=$(( SECONDS - UPLOAD_DONE ))
    log "  Processing + download took ${DOWNLOAD_ELAPSED}s"
    check_result_files "$RESULTS_DIR" "$HMAE_REQUIRE_RESULTS" "HMAE results"
else
    if [[ "$HMAE_REQUIRE_RESULTS" = "1" ]]; then
        die "Could not download HMAE results"
    fi
    warn "Could not download HMAE results — check server logs or set HMAE_REQUIRE_RESULTS=1 for strict validation"
fi

# ---------- Summary ----------
log ""
banner "Demo 04 Complete"
ls -lh "$RESULTS_DIR/" 2>/dev/null || true
log "Results in: ${RESULTS_DIR}/"
