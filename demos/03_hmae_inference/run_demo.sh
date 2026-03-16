#!/usr/bin/env bash
# =============================================================================
# HMAE Inference Demo via Mini-Rivault
# =============================================================================
# Launches Mini-Rivault with the vision-bentoml profile, uploads images,
# submits HMAE inference through BentoML vision service, retrieves results.
#
# Supports three modes: mae (masked reconstruction), full (full reconstruction),
# and features (feature extraction). Default: all three.
#
# Requires:
#   - Services running (demos/00_start_services/)
#   - HMAE model checkpoint pre-staged at $HMAE_MODEL_PATH
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"
SLURM_DEV_DIR="$DEMO_DIR/slurm-dev"

AI_LOGIN_CONTAINER="ai-login"
UMRS_CLIENT_CONTAINER="umrs-client"

# Inference mode: mae, full, features, or all (default)
HMAE_MODE="${HMAE_MODE:-all}"

# NGINX port must match what serve-model.slurm uses on the host.
NGINX_PORT="${NGINX_LITELLM_DOCKER_PORT:-11434}"
BENTOML_URL="http://localhost:${NGINX_PORT}/vision"

LOG_PREFIX="[hmae_demo]"
log() { echo "$LOG_PREFIX $*"; }
warn() { echo "$LOG_PREFIX WARNING: $*"; }
die() { echo "$LOG_PREFIX ERROR: $*" >&2; exit 1; }
run_cmd() {
    echo "$LOG_PREFIX \$ $*"
    "$@"
}

# ---------- Step 1: Verify services ----------
log "Step 1: Verifying UMRS server..."
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client ping || \
    die "UMRS server not reachable. Start services first."

# ---------- Step 2: Check for test images ----------
log ""
log "Step 2: Checking test images..."
TEST_DATA_DIR="$SCRIPT_DIR/test_data"
image_count=$(find -L "$TEST_DATA_DIR" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.tiff' \) 2>/dev/null | wc -l)
if [[ -d "$TEST_DATA_DIR" ]] && [[ "$image_count" -gt 0 ]]; then
    log "  Found $image_count test image(s) in $TEST_DATA_DIR"
else
    warn "No test images found in $TEST_DATA_DIR"
    warn "Place X-ray CT slice images (.png/.tiff) in test_data/"
    die "No test images available"
fi

# ---------- Step 3: Launch Mini-Rivault with vision-bentoml ----------
log ""
log "Step 3: Launching Mini-Rivault with vision-bentoml profile..."
log "  HMAE checkpoint must be pre-staged at: \${HMAE_MODEL_PATH:-./rivault/models/hmae}"

# Check for an existing Rivault job
existing_job="$(podman exec "$AI_LOGIN_CONTAINER" \
    squeue -h -o '%i %j' 2>/dev/null | grep -i 'mini-rivault' | awk '{print $1}' || true)"

if [[ -n "$existing_job" ]]; then
    # Verify the vision-bentoml profile is actually running (NGINX returns 502
    # when the upstream BentoML container is missing).
    if curl -sf "${BENTOML_URL}/healthz" >/dev/null 2>&1; then
        log "  Rivault job already running (job $existing_job) with healthy vision service — reusing."
    else
        log "  Rivault job $existing_job exists but vision-bentoml is NOT healthy."
        log "  Cancelling job $existing_job and relaunching with vision-bentoml profile..."
        run_cmd podman exec "$AI_LOGIN_CONTAINER" scancel "$existing_job"
        sleep 3  # brief pause for cleanup
        existing_job=""
    fi
fi

if [[ -z "$existing_job" ]]; then
    log "  Submitting Slurm job with RIVAULT_PROFILES=\"harness vllm vision-bentoml\"..."
    run_cmd podman exec "$AI_LOGIN_CONTAINER" \
        sbatch --export=ALL,RIVAULT_PROFILES="harness vllm vision-bentoml" \
        /root/.config/umrs/rivault_data/serve-model.slurm

    # Wait for job to start running
    log "  Waiting for Slurm job to start..."
    for i in $(seq 1 60); do
        job_state="$(podman exec "$AI_LOGIN_CONTAINER" \
            squeue -h -o '%T' -n mini-rivault 2>/dev/null | head -1 || true)"
        if [[ "$job_state" == "RUNNING" ]]; then
            log "  Slurm job is running."
            break
        fi
        [[ "$i" -eq 60 ]] && die "Timeout waiting for Slurm job to start (state: ${job_state:-UNKNOWN})"
        sleep 5
    done
fi

# ---------- Step 4: Wait for BentoML to be healthy ----------
log ""
log "Step 4: Waiting for BentoML service at $BENTOML_URL ..."

for i in $(seq 1 90); do
    if curl -sf "${BENTOML_URL}/healthz" >/dev/null 2>&1; then
        log "  BentoML vision service is healthy!"
        break
    fi
    [[ "$i" -eq 90 ]] && die "Timeout: BentoML not healthy after 450s at $BENTOML_URL"
    [[ $((i % 6)) -eq 0 ]] && log "  Still waiting... (${i}×5s)"
    sleep 5
done

# ---------- Step 5: Run HMAE demo script ----------
log ""
log "Step 5: Running HMAE inference (mode: '$HMAE_MODE')..."
log ""

DEMO_SCRIPT="$SLURM_DEV_DIR/scripts/demo_hmae_inference.py"
if [[ ! -f "$DEMO_SCRIPT" ]]; then
    die "Demo script not found: $DEMO_SCRIPT"
fi

S3_ENDPOINT="http://localhost:${MINIO_PORT:-19000}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-minioadmin}"
S3_SECRET_KEY="${S3_SECRET_KEY:-minioadmin}"
S3_BUCKET="hmae-demo"

# Build mode flag
mode_flag=""
if [[ "$HMAE_MODE" == "all" ]]; then
    mode_flag="--all"
else
    mode_flag="--mode $HMAE_MODE"
fi

# Pick first image only (demo needs just one)
TEST_IMAGE="$(find -L "$TEST_DATA_DIR" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.tiff' \) | head -1)"
log "  Image: $(basename "$TEST_IMAGE")"

run_cmd python3 "$DEMO_SCRIPT" \
    --bentoml-url "$BENTOML_URL" \
    --s3-endpoint "$S3_ENDPOINT" \
    --image-path "$TEST_IMAGE" \
    $mode_flag || die "HMAE inference failed."

# ---------- Step 6: Download results to client side ----------
log ""
log "Step 6: Downloading results from S3..."
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

python3 -c "
import boto3, os
s3 = boto3.client('s3',
    endpoint_url='${S3_ENDPOINT}',
    aws_access_key_id='${S3_ACCESS_KEY}',
    aws_secret_access_key='${S3_SECRET_KEY}')
bucket = '${S3_BUCKET}'
out = '${RESULTS_DIR}'
resp = s3.list_objects_v2(Bucket=bucket)
for obj in resp.get('Contents', []):
    key = obj['Key']
    dest = os.path.join(out, key)
    s3.download_file(bucket, key, dest)
    print(f'  Downloaded: {key} -> {dest}')
"
log "  Results saved to: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR/"

# ---------- Done ----------
log ""
log "HMAE inference demo complete!"
log "Results are in $RESULTS_DIR/"
