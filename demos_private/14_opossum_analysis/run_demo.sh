#!/usr/bin/env bash
# =============================================================================
# Demo 14: Opossum Video Motion Analysis
# =============================================================================
# Demonstrates video-based motion analysis using the UMRS Pipeline SDK.
#
# The opossum engine (tmp/opossum/) performs frame-difference motion detection
# on consecutive video files. This demo wraps it as a UDM workflow:
#
#   1. Locate real video data (or generate synthetic test videos)
#   2. Upload the opossum workflow to the server
#   3. Upload videos to the designated bucket
#   4. Server processes via Slurm (runs opossum analysis on HPC)
#   5. Download TSV motion event results
#
# The default data source is the real observation footage in
#   tmp/opossum/data/20260324/  (19 MP4 files, ~99 MB each)
# organised in hourly subdirectories (20/, 21/, 22/, 23/).
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running
#   - Python with opencv-python-headless and numpy available
#
# Usage:
#   bash run_demo.sh [video_dir]
#
# Optional env vars:
#   UMRS_USER_CONF   — path to user config
#   GEN_TEST_VIDEOS  — "1" to generate synthetic test videos instead
#   VIDEO_BUCKET     — bucket alias for video uploads (default: demo-opossum)
#   MAX_VIDEOS       — limit the number of videos to process (default: all)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="opossum" source "${SCRIPT_DIR}/../lib/common.sh"

# Ensure Opossum Analysis experience is in user config (idempotent).
ensure_experience "${SCRIPT_DIR}/experience.json"

# Use the configured movie bucket from the active user config.
BUCKET_ALIAS="${VIDEO_BUCKET:-demo-opossum}"

# Default to the real observation data from 2026-03-24
REAL_DATA_DIR="${SCRIPT_DIR}/../../tmp/opossum/data/20260324"
VIDEO_DIR="${1:-${REAL_DATA_DIR}}"
RESULTS_DIR="${SCRIPT_DIR}/results"

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

banner "Demo 14: Opossum Video Motion Analysis"
echo "User config  : ${CONF}"
echo "Bucket alias : ${BUCKET_ALIAS}"
echo "Video dir    : ${VIDEO_DIR}"
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
    1. Provide video directory:  bash run_demo.sh /path/to/videos/
    2. Use real data (default):  bash run_demo.sh ../../tmp/opossum/data/20260324
    3. Generate test videos:     GEN_TEST_VIDEOS=1 bash run_demo.sh"
fi

# Recursively find videos (real data has hourly subdirs 20/ 21/ 22/ 23/)
VIDEO_COUNT=$(find "${VIDEO_DIR}" \( -name "*.mp4" -o -name "*.avi" \) 2>/dev/null | wc -l)
if [[ "${VIDEO_COUNT}" -eq 0 ]]; then
    die "No .mp4 or .avi files found in ${VIDEO_DIR}"
fi
log "  Found ${VIDEO_COUNT} video file(s) (recursive)"

# ---------- Step 3: Upload opossum workflow ----------
log ""
log "Step 3: Uploading opossum workflow to server..."
WORKFLOW_DIR="${SCRIPT_DIR}/workflow"
if [[ -d "${WORKFLOW_DIR}" ]] && [[ -f "${WORKFLOW_DIR}/workflow.json" ]]; then
    $CLIENT upload-workflow "${WORKFLOW_DIR}" --name opossum-analysis || warn "workflow upload returned non-zero"
    ok "Workflow uploaded"
else
    warn "Workflow directory not found at ${WORKFLOW_DIR} — skipping"
fi

# ---------- Step 4: Upload videos ----------
log ""
log "Step 4: Uploading videos to bucket '${BUCKET_ALIAS}'..."
UPLOAD_JSON="$($CLIENT --output json upload_dir "${VIDEO_DIR}" "$BUCKET_ALIAS" 2>&1 || true)"
printf '%s\n' "$UPLOAD_JSON"
if [[ "$(printf '%s\n' "$UPLOAD_JSON" | extract_status_from_cli_output || true)" == "ok" ]]; then
    ok "Videos uploaded — server will process automatically"
else
    warn "Video upload was not accepted by the server for bucket '${BUCKET_ALIAS}'"
fi
UPLOAD_DONE=$SECONDS

# ---------- Step 5: Run local analysis (fallback if server-side not configured) ----------
log ""
log "Step 5: Running local opossum analysis as demonstration..."
# Since the opossum experience may not be configured server-side,
# we demonstrate the analysis engine locally using the same files.
OPOSSUM_SRC="${SCRIPT_DIR}/../../tmp/opossum/src"
if [[ -d "${OPOSSUM_SRC}" ]]; then
    mkdir -p "$RESULTS_DIR"
    # Run the opossum analysis on consecutive video pairs
    python3 - <<PYEOF
import sys, os
from pathlib import Path
sys.path.insert(0, "${OPOSSUM_SRC}")
from opossum.analyze import get_duration

video_dir = Path("${VIDEO_DIR}")
results_dir = Path("${RESULTS_DIR}")
max_videos = int("${MAX_VIDEOS:-0}")  # 0 = no limit
dt = 0.05   # distance threshold
tt = 15.0   # time threshold (seconds)

# Recursively collect and sort video files (handles hourly subdirs)
videos = sorted(
    p for p in video_dir.rglob("*")
    if p.suffix.lower() in ('.mp4', '.avi')
    and not p.name.startswith("._")
    and not p.name.endswith("_concat.mp4")
)

if max_videos > 0:
    videos = videos[:max_videos]
    print(f"  Limited to {max_videos} videos")

print(f"  Processing {len(videos)} videos ({len(videos)-1} consecutive pairs)")
all_events = []

# Process consecutive pairs
for i in range(len(videos) - 1):
    v1, v2 = videos[i], videos[i+1]
    print(f"  [{i+1}/{len(videos)-1}] {v1.name} + {v2.name}")
    try:
        events = get_duration(v1, v2,
                              distance_threshold=dt, time_threshold=tt)
        # First element is header ["start_time", "end_time"], skip it
        data_events = events[1:] if events and len(events) > 1 else []
        if data_events:
            pair_name = f"{v1.stem}_{v2.stem}"
            for ev in data_events:
                all_events.append([pair_name] + list(ev))
            print(f"         -> {len(data_events)} motion event(s)")
        else:
            print(f"         -> No motion events")
    except Exception as e:
        print(f"         -> Error: {e}")

# Save results as TSV
tsv_path = results_dir / "motion_events.tsv"
with open(tsv_path, 'w') as f:
    f.write("video_pair\tstart_time\tend_time\n")
    if all_events:
        for ev in all_events:
            f.write('\t'.join(str(x) for x in ev) + '\n')
        print(f"\n  {len(all_events)} total motion event(s) saved to {tsv_path}")
    else:
        f.write("# No motion events detected\n")
        print(f"\n  No motion events detected across all pairs")
print("  Analysis complete")
PYEOF
    ok "Local analysis complete"
else
    warn "Opossum engine not found at ${OPOSSUM_SRC} — skipping local analysis"
fi

# ---------- Step 6: Try downloading server-side results ----------
log ""
log "Step 6: Attempting to download server-side results..."
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
    log "Motion events:"
    cat "${RESULTS_DIR}/motion_events.tsv"
fi
ls -lh "$RESULTS_DIR/" 2>/dev/null || true
log "Results in: ${RESULTS_DIR}/"
