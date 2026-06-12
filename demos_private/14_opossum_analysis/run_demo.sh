#!/usr/bin/env bash
# =============================================================================
# Demo 14: Opossum Video Motion Analysis (Production Pipeline)
# =============================================================================
# Demonstrates GPU-accelerated video motion analysis using the UMRS Pipeline SDK.
#
# Production pipeline stages:
#   1. VFR→CFR conversion (ffmpeg) for real trail-camera footage
#   2. Frame-difference motion detection (GPU-accelerated cv2.cuda)
#   3. Per-camera incremental tracking via checkpoints
#   4. Per-pair TSV output + aggregated motion_events.tsv
#
# Real data source:
#   tmp/opossum-video-analysis/data/opossum_*/opossum*/Schedule/YYYYMMDD/HH/*.mp4
#   Multi-camera layout: opossum_215_A/opossumA, opossum_215_B/opossumB, ...
#   Hourly subdirs: 00, 01, 02, ..., 23
#   Filenames encode datetime: YYYYMMDD_HHMMSS.mp4
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running
#   - python3 with opencv-python-headless and numpy available
#   - ffmpeg/ffprobe available for VFR→CFR conversion
#
# Usage:
#   bash run_demo.sh [video_dir]
#
# Optional env vars:
#   UMRS_USER_CONF     — path to user config
#   GEN_TEST_VIDEOS    — "1" to generate synthetic test videos instead
#   VIDEO_BUCKET       — bucket alias for video uploads (default: demo-opossum)
#   MAX_VIDEOS         — limit videos per camera (default: all)
#   GPU_ENABLED        — "1" to request GPU node for server-side execution
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="opossum" source "${SCRIPT_DIR}/../lib/common.sh"

# Ensure Opossum Analysis experience is in user config (idempotent).
ensure_experience "${SCRIPT_DIR}/experience.json"

# Use the configured movie bucket from the active user config.
BUCKET_ALIAS="${VIDEO_BUCKET:-demo-opossum}"

# Default to real observation data from the production pipeline.
REAL_DATA_BASE="${SCRIPT_DIR}/../../tmp/opossum-video-analysis/data"
VIDEO_DIR="${1:-${REAL_DATA_BASE}}"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Opossum engine source (to bundle with workflow upload).
OPOSSUM_ENGINE="${SCRIPT_DIR}/../../tmp/opossum-video-analysis/src/opossum"

extract_status_from_cli_output() {
    python3 -c 'import json, sys
payload = sys.stdin.read().splitlines()
for line in reversed(payload):
    line = line.strip()
    if not line or not line.startswith("{"):
        continue
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        continue
    print(data.get("status", ""))
    sys.exit(0)
sys.exit(1)'
}

banner "Demo 14: Opossum Video Motion Analysis (Production Pipeline)"
echo "User config  : ${CONF}"
echo "Bucket alias : ${BUCKET_ALIAS}"
echo "Video dir    : ${VIDEO_DIR}"
echo "GPU mode     : ${GPU_ENABLED:-0}"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

# ---------- Step 2: Locate or generate test videos ----------
log ""
log "Step 2: Preparing video data..."

if [[ "${GEN_TEST_VIDEOS:-0}" == "1" ]]; then
    VIDEO_DIR="${SCRIPT_DIR}/sample_videos"
    log "  Generating synthetic test videos..."
    mkdir -p "${VIDEO_DIR}"
    python3 "${SCRIPT_DIR}/generate_test_videos.py" --output-dir "${VIDEO_DIR}" --num-videos 4
    ok "Generated test videos in ${VIDEO_DIR}"
elif [[ -z "${1:-}" && ! -d "${VIDEO_DIR}" ]]; then
    warn "Default real data directory not found; generating synthetic test videos instead"
    VIDEO_DIR="${SCRIPT_DIR}/sample_videos"
    mkdir -p "${VIDEO_DIR}"
    python3 "${SCRIPT_DIR}/generate_test_videos.py" --output-dir "${VIDEO_DIR}" --num-videos 4
    ok "Generated test videos in ${VIDEO_DIR}"
fi

if [[ ! -d "${VIDEO_DIR}" ]]; then
    die "Video directory not found: ${VIDEO_DIR}
  Options:
    1. Provide real data:       bash run_demo.sh /path/to/data/
    2. Use production data:     bash run_demo.sh ../../tmp/opossum-video-analysis/data
    3. Generate test videos:    GEN_TEST_VIDEOS=1 bash run_demo.sh"
fi

# Recursively find videos (real data has camera/Schedule/YYYYMMDD/HH/*.mp4 structure)
VIDEO_COUNT=$(find "${VIDEO_DIR}" \( -name "*.mp4" -o -name "*.avi" \) 2>/dev/null | wc -l)
if [[ "${VIDEO_COUNT}" -eq 0 ]]; then
    die "No .mp4 or .avi files found in ${VIDEO_DIR}"
fi
log "  Found ${VIDEO_COUNT} video file(s) (recursive)"

# Count cameras
CAMERA_COUNT=$(find "${VIDEO_DIR}" -maxdepth 1 -type d -name "opossum_*" 2>/dev/null | wc -l)
if [[ "${CAMERA_COUNT}" -gt 0 ]]; then
    log "  Detected ${CAMERA_COUNT} camera(s)"
fi

# ---------- Step 3: Bundle opossum engine with workflow ----------
log ""
log "Step 3: Preparing workflow bundle..."

WORKFLOW_DIR="${SCRIPT_DIR}/workflow"
OPOSSUM_DEST="${WORKFLOW_DIR}/opossum"

if [[ -d "${OPOSSUM_ENGINE}" ]]; then
    # Copy opossum package into workflow dir for upload
    log "  Bundling opossum engine into workflow..."
    rm -rf "${OPOSSUM_DEST}"
    cp -r "${OPOSSUM_ENGINE}" "${OPOSSUM_DEST}"
    ok "Opossum engine bundled"
else
    warn "Opossum engine not found at ${OPOSSUM_ENGINE} — workflow will lack motion analysis"
    warn "  Expected: tmp/opossum-video-analysis/src/opossum/"
fi

# ---------- Step 4: Upload opossum workflow ----------
log ""
log "Step 4: Uploading opossum workflow to server..."
if [[ -d "${WORKFLOW_DIR}" ]] && [[ -f "${WORKFLOW_DIR}/workflow.json" ]]; then
    $CLIENT upload-workflow "${WORKFLOW_DIR}" --name opossum-analysis || warn "workflow upload returned non-zero"
    ok "Workflow uploaded (includes bundled opossum engine)"
else
    warn "Workflow directory not found at ${WORKFLOW_DIR} — skipping"
fi

# ---------- Step 5: Upload videos (preserving camera directory structure) ----------
log ""
log "Step 5: Uploading videos to bucket '${BUCKET_ALIAS}'..."
log "  (using recursive upload_dir — preserves camera/Schedule date hierarchy)"
UPLOAD_JSON="$($CLIENT --output json upload_dir "${VIDEO_DIR}" "$BUCKET_ALIAS" 2>&1 || true)"
printf '%s\n' "$UPLOAD_JSON"
if [[ "$(printf '%s\n' "$UPLOAD_JSON" | extract_status_from_cli_output || true)" == "ok" ]]; then
    ok "Videos uploaded — server will process via Slurm GPU job"
else
    warn "Video upload was not accepted by the server for bucket '${BUCKET_ALIAS}'"
fi
UPLOAD_DONE=$SECONDS

# ---------- Step 6: Run local analysis (fallback with real pipeline) ----------
log ""
log "Step 6: Running local opossum analysis (production pipeline)..."
# Uses the same production logic as the server-side step:
#   multi-camera grouping, VFR→CFR, frame-difference, per-pair TSV
if [[ -d "${OPOSSUM_ENGINE}" ]]; then
    mkdir -p "$RESULTS_DIR"

    GPU_FLAG=""
    if [[ "${GPU_ENABLED:-0}" == "1" ]]; then
        GPU_FLAG="--gpu"
    fi

    python3 - <<PYEOF
import sys, os, json, subprocess
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "${SCRIPT_DIR}/../../tmp/opossum-video-analysis/src")

import cv2
from opossum.utils import (
    binarize_image,
    remove_white_noise,
    merge_adjacent_boxes,
    calculate_mask_percentage_in_rectangle,
)

video_dir = Path("${VIDEO_DIR}")
results_dir = Path("${RESULTS_DIR}")
dt = 0.05   # distance threshold
tt = 15.0   # time threshold (seconds)
max_videos = int("${MAX_VIDEOS:-0}")  # 0 = no limit

# Check GPU availability
use_cuda = False
if "${GPU_ENABLED:-0}" == "1":
    try:
        use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        print(f"  GPU mode: {'CUDA available (' + str(cv2.cuda.getCudaEnabledDeviceCount()) + ' devices)' if use_cuda else 'CUDA not available, falling back to CPU'}")
    except:
        print("  GPU mode: cv2.cuda not available, using CPU")

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_datetime_from_filename(filename):
    date_str = filename.replace("Record-", "")
    date_str = date_str.replace(".mp4", "").replace(".avi", "")
    date_str = date_str.replace("-", "")
    return datetime.strptime(date_str, "%Y%m%d_%H%M%S")

def vfr2cfr(file_path, frame_rate=5):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate",
           "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
    r_frame_rate = subprocess.check_output(cmd, text=True).strip()
    if r_frame_rate == "0/0":
        temp_path = Path("/tmp") / f"{file_path.stem}_cfr.mp4"
        subprocess.run(["ffmpeg", "-i", str(file_path), "-vf", f"fps={frame_rate}",
                        "-c:v", "libx264", "-c:a", "copy", "-y", str(temp_path)],
                       check=True, capture_output=True)
        return temp_path
    return file_path

def find_continuous_regions(rates, min_gap_frames):
    regions = []
    in_region = False
    start = 0
    n = len(rates)
    for i in range(n):
        if rates[i] > 0:
            if not in_region:
                start = i
                in_region = True
        else:
            if in_region:
                gap_end = min(i + int(min_gap_frames) + 1, n)
                if any(rates[j] > 0 for j in range(i + 1, gap_end)):
                    continue
                regions.append((start, i - 1))
                in_region = False
    if in_region:
        regions.append((start, n - 1))
    return regions

def seconds_to_hms(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def analyze_pair(v1, v2):
    """Production frame-difference analysis on consecutive video pair."""
    path1 = vfr2cfr(v1)
    path2 = vfr2cfr(v2)

    cap1 = cv2.VideoCapture(str(path1))
    cap2 = cv2.VideoCapture(str(path2))

    try:
        frame_rate = cap1.get(cv2.CAP_PROP_FPS)
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        merge_threshold = width * dt

        ret1, prev_frame = cap1.read()
        if not ret1:
            return []
        ret1, frame = cap1.read()

        masked_pixel_rate = []

        # Phase 1: first video
        while ret1:
            if use_cuda:
                gpu_f = cv2.cuda_GpuMat()
                gpu_p = cv2.cuda_GpuMat()
                gpu_f.upload(frame)
                gpu_p.upload(prev_frame)
                gpu_bin = cv2.cuda.cvtColor(gpu_f, cv2.COLOR_BGR2GRAY)
                cv2.cuda.threshold(gpu_bin, 128, 255, cv2.THRESH_BINARY, dst=gpu_bin)
                gpu_pbin = cv2.cuda.cvtColor(gpu_p, cv2.COLOR_BGR2GRAY)
                cv2.cuda.threshold(gpu_pbin, 128, 255, cv2.THRESH_BINARY, dst=gpu_pbin)
                gpu_diff = cv2.cuda.absdiff(gpu_bin, gpu_pbin)
                bdiff = gpu_diff.download()
            else:
                binarized = binarize_image(frame)
                bdiff = cv2.absdiff(binarized, binarize_image(prev_frame))

            without_noise = remove_white_noise(bdiff, 5)
            merged = merge_adjacent_boxes(without_noise, threshold=merge_threshold)
            rate = calculate_mask_percentage_in_rectangle(
                merged, {"x": 0, "y": 0, "width": width, "height": height})
            masked_pixel_rate.append(rate)
            prev_frame = frame.copy()
            ret1, frame = cap1.read()

        # Phase 2: second video (up to time_threshold)
        ret2, frame = cap2.read()
        if ret2:
            prev_frame = frame.copy()
            ret2, frame = cap2.read()

        while ret2:
            sec = round(cap2.get(cv2.CAP_PROP_POS_FRAMES) / frame_rate)
            if sec > tt:
                break
            if use_cuda:
                gpu_f = cv2.cuda_GpuMat()
                gpu_p = cv2.cuda_GpuMat()
                gpu_f.upload(frame)
                gpu_p.upload(prev_frame)
                gpu_bin = cv2.cuda.cvtColor(gpu_f, cv2.COLOR_BGR2GRAY)
                cv2.cuda.threshold(gpu_bin, 128, 255, cv2.THRESH_BINARY, dst=gpu_bin)
                gpu_pbin = cv2.cuda.cvtColor(gpu_p, cv2.COLOR_BGR2GRAY)
                cv2.cuda.threshold(gpu_pbin, 128, 255, cv2.THRESH_BINARY, dst=gpu_pbin)
                gpu_diff = cv2.cuda.absdiff(gpu_bin, gpu_pbin)
                bdiff = gpu_diff.download()
            else:
                binarized = binarize_image(frame)
                bdiff = cv2.absdiff(binarized, binarize_image(prev_frame))

            without_noise = remove_white_noise(bdiff, 5)
            merged = merge_adjacent_boxes(without_noise, threshold=merge_threshold)
            rate = calculate_mask_percentage_in_rectangle(
                merged, {"x": 0, "y": 0, "width": width, "height": height})
            masked_pixel_rate.append(rate)
            prev_frame = frame.copy()
            ret2, frame = cap2.read()

        events = find_continuous_regions(masked_pixel_rate, frame_rate * tt)
        return [[seconds_to_hms(start), seconds_to_hms(end)] for start, end in events]
    finally:
        cap1.release()
        cap2.release()
        if path1 != v1 and path1.exists():
            path1.unlink()
        if path2 != v2 and path2.exists():
            path2.unlink()

# ── Main: multi-camera analysis ──

# Group videos by camera directory
cameras = defaultdict(list)
for p in video_dir.rglob("*.mp4"):
    if p.name.startswith("._") or p.name.endswith("_concat.mp4"):
        continue
    parts = list(p.relative_to(video_dir).parts)
    if len(parts) >= 2 and parts[0].startswith("opossum_"):
        camera = f"{parts[0]}/{parts[1]}"
    else:
        camera = parts[0] if len(parts) > 1 else "default"
    cameras[camera].append(p)

# Also find .avi files
for p in video_dir.rglob("*.avi"):
    if p.name.startswith("._"):
        continue
    parts = list(p.relative_to(video_dir).parts)
    if len(parts) >= 2 and parts[0].startswith("opossum_"):
        camera = f"{parts[0]}/{parts[1]}"
    else:
        camera = parts[0] if len(parts) > 1 else "default"
    cameras[camera].append(p)

print(f"  Cameras detected: {len(cameras)} ({list(cameras.keys())})")

all_events = []
for camera_name, videos in sorted(cameras.items()):
    # Sort by filename datetime
    videos.sort(key=lambda p: get_datetime_from_filename(p.name))
    if max_videos > 0:
        videos = videos[:max_videos]

    print(f"\n  Camera: {camera_name} ({len(videos)} videos)")
    if len(videos) < 2:
        print(f"    Skipping — need at least 2 videos")
        continue

    camera_events = 0
    for i in range(len(videos) - 1):
        v1, v2 = videos[i], videos[i + 1]
        pair_name = f"{v1.stem}_{v2.stem}"
        print(f"    [{i+1}/{len(videos)-1}] {v1.name} + {v2.name}", end="")
        try:
            events = analyze_pair(v1, v2)
            if events:
                print(f" -> {len(events)} motion event(s)")
                camera_events += len(events)
                for ev in events:
                    all_events.append({
                        "camera": camera_name, "pair": pair_name,
                        "start": ev[0], "end": ev[1],
                    })
                # Per-pair TSV
                out_dir = results_dir / camera_name
                out_dir.mkdir(parents=True, exist_ok=True)
                tsv_path = out_dir / f"{v1.stem}.tsv"
                with open(tsv_path, 'w') as f:
                    f.write("start_time\tend_time\n")
                    for ev in events:
                        f.write(f"{ev[0]}\t{ev[1]}\n")
            else:
                print(" -> No motion events")
        except Exception as e:
            print(f" -> Error: {e}")

    print(f"    Total: {camera_events} motion events")

# Aggregated TSV
tsv_path = results_dir / "motion_events.tsv"
with open(tsv_path, 'w') as f:
    f.write("camera\tvideo_pair\tstart_time\tend_time\n")
    if all_events:
        for ev in all_events:
            f.write(f"{ev['camera']}\t{ev['pair']}\t{ev['start']}\t{ev['end']}\n")
        print(f"\n  {len(all_events)} total motion event(s) across {len(cameras)} cameras")
        print(f"  Aggregated results: {tsv_path}")
    else:
        f.write("# No motion events detected\n")
        print(f"\n  No motion events detected across all cameras")

# JSON summary
summary = {
    "gpu_accelerated": use_cuda,
    "cameras_analyzed": list(cameras.keys()),
    "total_cameras": len(cameras),
    "total_events": len(all_events),
    "distance_threshold": dt,
    "time_threshold": tt,
}
summary_path = results_dir / "analysis_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"  Summary: {summary_path}")
print("  Local analysis complete")
PYEOF
    ok "Local analysis complete"
else
    warn "Opossum engine not found at ${OPOSSUM_ENGINE} — skipping local analysis"
fi

# ---------- Step 7: Download server-side results ----------
log ""
log "Step 7: Downloading server-side results..."
DOWNLOAD_JSON="$($CLIENT --output json download_results "$BUCKET_ALIAS" "$RESULTS_DIR" --force-overwrite 2>/dev/null || true)"
if [[ -n "$DOWNLOAD_JSON" ]] && [[ "$(printf '%s\n' "$DOWNLOAD_JSON" | extract_status_from_cli_output || true)" == "ok" ]]; then
    printf '%s\n' "$DOWNLOAD_JSON"
else
    warn "Server-side download was not available (workflow may not be wired to '${BUCKET_ALIAS}')"
    log "  Local analysis results are in: ${RESULTS_DIR}/"
fi

# ---------- Summary ----------
log ""
banner "Demo 14 Complete"
if [[ -f "${RESULTS_DIR}/motion_events.tsv" ]]; then
    log "Motion events (first 10 lines):"
    head -10 "${RESULTS_DIR}/motion_events.tsv"
fi
echo ""
ls -lh "$RESULTS_DIR/" 2>/dev/null || true
if [[ -d "${RESULTS_DIR}" ]]; then
    # Show per-camera results
    find "$RESULTS_DIR" -name "*.tsv" -type f 2>/dev/null | while read -r f; do
        echo "  $(wc -l < "$f" | tr -d ' ') lines  ${f#$RESULTS_DIR/}"
    done
fi
log ""
log "Results in: ${RESULTS_DIR}/"
log ""
log "Tip: For GPU-accelerated server execution, run:"
log "  GPU_ENABLED=1 bash run_demo.sh"
