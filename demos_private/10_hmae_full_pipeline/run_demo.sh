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
#   demo-hmae           → reconstruction
#   demo-hmae-mae       → mae_reconstruction
#   demo-hmae-features  → feature_extraction
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

# Ensure HMAE Inference experience is in user config (idempotent).
ensure_experience "${SCRIPT_DIR}/experience.json" --update

MAX_IMAGES="${MAX_IMAGES:-3}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-1}"
RELOAD_SETTLE_SECONDS="${RELOAD_SETTLE_SECONDS:-30}"
RESULTS_DIR="${SCRIPT_DIR}/results"
HMAE_ASSET_ROOT="${HMAE_ASSET_ROOT:-${SCRIPT_DIR}/../../tmp/hmae}"
HMAE_INFER_DIR="${HMAE_INFER_DIR:-${SCRIPT_DIR}/../../tmp/hmae-infer}"
HMAE_LOCAL_CHECKPOINT="${HMAE_LOCAL_CHECKPOINT:-${HMAE_ASSET_ROOT}/latest_checkpoint.pth}"
HMAE_LOCAL_CONFIG="${HMAE_LOCAL_CONFIG:-${HMAE_INFER_DIR}/configs/infer_hmae_vit_b.yaml}"
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

# Test images: use the Spring-8 concrete volume slices for this demo.
IMAGE_DIR="${IMAGE_DIR:-${HMAE_ASSET_ROOT}/spring8concrete}"

# All 3 HMAE tasks and their bucket aliases
HMAE_TASKS=("reconstruction" "mae_reconstruction" "feature_extraction")
HMAE_BUCKETS=("demo-hmae" "demo-hmae-mae" "demo-hmae-features")

bucket_names_for_alias() {
    local alias="$1"
    CONF="${CONF}" BUCKET_ALIAS="${alias}" python3 - <<'PY'
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
}

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

banner "Demo 10: HMAE Full Pipeline (all 3 tasks)"
echo "User config  : ${CONF}"
echo "Image dir    : ${IMAGE_DIR}"
echo "HMAE assets  : ${HMAE_ASSET_ROOT}"
echo "HMAE config  : ${HMAE_LOCAL_CONFIG}"
echo "Checkpoint   : ${HMAE_LOCAL_CHECKPOINT}"
echo "Tasks        : ${HMAE_TASKS[*]}"
echo "Force recomp : ${FORCE_RECOMPUTE}"
echo ""

# ──────────────────────────────────────────────────────────────────────────
log "Step 1: Pre-flight checks..."
preflight
[[ -d "$HMAE_INFER_DIR" ]] || die "HMAE inference checkout not found: ${HMAE_INFER_DIR}"
[[ -f "$HMAE_LOCAL_CONFIG" ]] || die "HMAE config not found: ${HMAE_LOCAL_CONFIG}"
[[ -f "$HMAE_LOCAL_CHECKPOINT" ]] || die "HMAE checkpoint not found: ${HMAE_LOCAL_CHECKPOINT}"
ok "Using hmae-infer config and checkpoint from umrs/tmp"
log "Using the deployed HMAE experiences as-is (no admin reload in demo mode)."

# ──────────────────────────────────────────────────────────────────────────
log ""
log "Step 2: Checking test images..."
if [[ ! -d "$IMAGE_DIR" ]]; then
    die "Image directory not found: $IMAGE_DIR"
fi
IMAGE_COUNT=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.tiff" \) | wc -l)
if [[ "$IMAGE_COUNT" -eq 0 ]]; then
    die "No image files found in $IMAGE_DIR"
fi

# Stage a subset
UPLOAD_DIR="$IMAGE_DIR"
if [[ "$IMAGE_COUNT" -gt "$MAX_IMAGES" ]]; then
    UPLOAD_DIR="${SCRIPT_DIR}/.upload_staging"
    rm -rf "$UPLOAD_DIR"
    mkdir -p "$UPLOAD_DIR"
    find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.tiff" \) | sort \
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
    read -r INPUT_BUCKET OUTPUT_BUCKET < <(bucket_names_for_alias "$BUCKET")
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

    if [[ "$FORCE_RECOMPUTE" != "0" ]]; then
        log "FORCE_RECOMPUTE set — triggering full recompute..."
        maybe_admin_recompute || warn "recompute returned non-zero"
    fi
    UPLOAD_DONE=$SECONDS

    log "Step 4.${i}: Downloading ${TASK} results..."
    log "  (This may take several minutes for HMAE model loading + inference)"
    rm -rf "$TASK_RESULTS_DIR"
    mkdir -p "$TASK_RESULTS_DIR"
    if download_results_bounded "$BUCKET" "$TASK_RESULTS_DIR" --force-overwrite; then
        DOWNLOAD_ELAPSED=$(( SECONDS - UPLOAD_DONE ))
        log "  ${TASK} processing + download took ${DOWNLOAD_ELAPSED}s"
        check_result_files "$TASK_RESULTS_DIR" "$HMAE_REQUIRE_RESULTS" "$TASK"
    else
        if [[ "$HMAE_REQUIRE_RESULTS" = "1" ]]; then
            die "Could not download ${TASK} results"
        fi
        warn "Could not download ${TASK} results — check server logs or set HMAE_REQUIRE_RESULTS=1 for strict validation"
    fi
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
