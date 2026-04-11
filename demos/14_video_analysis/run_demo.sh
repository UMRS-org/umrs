#!/usr/bin/env bash
# =============================================================================
# Demo 14: Video Motion Analysis
# =============================================================================
# Detect motion events in video files using OpenCV frame differencing.
# No external model or GPU required — runs entirely on CPU.
#
# The demo can use:
#   a) Synthetic test videos (generated automatically)
#   b) Your own video files passed as an argument
#
# Workflow submitted to UMRS:
#   video_workflow/workflow.json → MotionDetectStep
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - opencv-python-headless and numpy installed (for test video generation)
#   - A bucket alias with 'movie' modality and 'need_output_bucket: true'
#     configured in your user config (user1-video by default).
#
# Usage:
#   bash run_demo.sh                           # generate + analyse test videos
#   bash run_demo.sh /path/to/videos           # your own MP4 files
#   VIDEO_BUCKET=my-bucket bash run_demo.sh
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="video_analysis" source "${SCRIPT_DIR}/../lib/common.sh"

BUCKET_ALIAS="${VIDEO_BUCKET:-user1-video}"
MAX_VIDEOS="${MAX_VIDEOS:-6}"
VIDEO_DIR="${1:-}"
RESULTS_DIR="${SCRIPT_DIR}/results"
WORKFLOW_SOURCE_DIR="${SCRIPT_DIR}/video_workflow"
WORKFLOW_UPLOAD_DIR="$(prepare_workflow_upload_dir "${WORKFLOW_SOURCE_DIR}")"
trap 'rm -rf "${WORKFLOW_UPLOAD_DIR:-}"' EXIT

banner "Demo 14: Video Motion Analysis"
echo "Config        : ${CONF}"
echo "Bucket alias  : ${BUCKET_ALIAS}"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

# ---------- Step 2: Generate or locate videos ----------
log ""
log "Step 2: Preparing video data..."
if [[ -z "$VIDEO_DIR" ]]; then
    # Generate synthetic test videos
    VIDEO_DIR="${SCRIPT_DIR}/sample_videos"
    EXISTING_COUNT=$(find "$VIDEO_DIR" -maxdepth 1 -name "*.mp4" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$EXISTING_COUNT" -lt 2 ]]; then
        log "  Generating synthetic test videos..."
        python3 "${SCRIPT_DIR}/generate_test_videos.py" -o "$VIDEO_DIR" --count 4 --duration 10
        ok "Test videos generated in ${VIDEO_DIR}/"
    else
        log "  Using existing test videos (${EXISTING_COUNT} files)"
    fi
else
    if [[ ! -d "$VIDEO_DIR" ]]; then
        die "Video directory not found: $VIDEO_DIR"
    fi
fi

VIDEO_COUNT=$(find "$VIDEO_DIR" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | wc -l | tr -d ' ')
if [[ "$VIDEO_COUNT" -eq 0 ]]; then
    die "No video files found in $VIDEO_DIR"
fi
log "  Found ${VIDEO_COUNT} video(s)"

# Stage subset if needed
UPLOAD_DIR="$VIDEO_DIR"
if [[ "$VIDEO_COUNT" -gt "$MAX_VIDEOS" ]]; then
    UPLOAD_DIR="${SCRIPT_DIR}/.upload_staging"
    rm -rf "$UPLOAD_DIR"
    mkdir -p "$UPLOAD_DIR"
    find "$VIDEO_DIR" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) \
        | head -n "$MAX_VIDEOS" | while read -r f; do ln -s "$f" "$UPLOAD_DIR/"; done
    log "  Staged ${MAX_VIDEOS} of ${VIDEO_COUNT} videos"
fi

# ---------- Step 3: Upload workflow ----------
log ""
log "Step 3: Uploading video analysis workflow..."
$CLIENT upload-workflow "${WORKFLOW_UPLOAD_DIR}" --name "video-motion-analysis" 2>&1 | while IFS= read -r line; do log "  $line"; done
ok "Workflow uploaded"
$CLIENT reload 2>/dev/null || warn "reload returned non-zero (may be OK)"
sleep 3

# ---------- Step 4: Upload videos ----------
log ""
log "Step 4: Uploading videos to '${BUCKET_ALIAS}'..."
$CLIENT upload_dir "$UPLOAD_DIR" "$BUCKET_ALIAS"
ok "Videos uploaded"

# ---------- Step 5: Wait and download results ----------
log ""
log "Step 5: Waiting for motion analysis to complete..."
log "  (Frame differencing — CPU only, should be fast)"
START=$SECONDS
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
$CLIENT download_results "$BUCKET_ALIAS" "$RESULTS_DIR" --force-overwrite
ELAPSED=$(( SECONDS - START ))
log "  Processing + download took ${ELAPSED}s"
ok "Results saved to ${RESULTS_DIR}/"

# ---------- Summary ----------
log ""
banner "Demo 14 Complete"

SUMMARY="${RESULTS_DIR}/results/summary.json"
if [[ -f "$SUMMARY" ]]; then
    log "Analysis summary:"
    cat "$SUMMARY"
else
    # Try alternative path
    SUMMARY2="${RESULTS_DIR}/summary.json"
    if [[ -f "$SUMMARY2" ]]; then
        log "Analysis summary:"
        cat "$SUMMARY2"
    else
        log "Result files:"
        find "$RESULTS_DIR" -type f -name "*.json" | head -20
    fi
fi

log ""
log "Per-video motion reports: ${RESULTS_DIR}/results/*_motion.json"
log "Aggregate summary:        ${RESULTS_DIR}/results/summary.json"
