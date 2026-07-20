# Demo 14: Video Motion Analysis

Detect motion events in video files using **OpenCV frame differencing** —
no external model or GPU required.

## Overview

```
┌──────────────┐     ┌───────────────────┐     ┌────────────────┐
│ Test videos  │────▶│ MotionDetectStep  │────▶│ Motion reports │
│  or your own │     │  (frame diff, CPU) │     │ JSON per video │
└──────────────┘     └───────────────────┘     └────────────────┘
```

The pipeline analyses consecutive frames for pixel intensity changes,
clusters significant changes into discrete events, and reports start/end
timestamps plus duration for each event.

## Quick Start

```bash
# See "Running Manually" and "Generate Test Videos" sections below for step-by-step commands
cd "$(git rev-parse --show-toplevel)/umrs/demos/14_video_analysis"

# Generate synthetic test videos + analyse them
GEN_TEST_VIDEOS=1 bash ./run_demo.sh

# Use your own videos
VIDEO_DIR=/path/to/videos bash ./run_demo.sh

# Custom bucket
VIDEO_BUCKET=my-project bash ./run_demo.sh
```

> **Note:** The inline script `run_demo.sh` does not exist yet. Use the
> step-by-step commands in the "Running Manually" section below instead.

## Synthetic Test Videos

If no video directory is supplied, the demo auto-generates 4 short
(10-second) MP4 files with a moving white rectangle. Every 3rd video is
static (no motion), giving the detector both positive and negative cases.

Generate manually:

```bash
python generate_test_videos.py -o sample_videos/ --count 8 --fps 30
```

## What the Script Does

1. **Pre-flight** — verify server connectivity
2. **Prepare videos** — generate synthetic data or locate user-supplied files
3. **Upload workflow** — push `workflow/` to the server
4. **Upload videos** — send MP4 files to the configured bucket
5. **Wait & download** — poll until analysis completes, download results
6. **Show summary** — print aggregate motion report

## Running Manually

```bash
# 1. Verify server connectivity
umrs-client ping

# 2. (Optional) Generate synthetic test videos
#    Creates short MP4 files with a moving white rectangle.
#    Every 3rd video is static (no motion) for negative test cases.
python3 generate_test_videos.py -o sample_videos/ --count 4 --fps 15

# 3. Upload the motion detection workflow to the server
#    The workflow contains a MotionDetectStep that analyzes
#    consecutive frames for pixel intensity changes and reports
#    motion events with timestamps.
umrs-client workflow upload workflow/ --name "video-motion-analysis"

# 4. Upload videos to the analysis bucket
#    Each video will be processed by the MotionDetectStep,
#    producing a per-video JSON report with motion events.
umrs-client data dir-upload ./sample_videos user1-video

# 5. Download analysis results
#    Results include per-video motion reports and an aggregate
#    summary.json with total events across all videos.
umrs-client data download user1-video ./results --force-overwrite
```

This demo shows the normal user path: upload the workflow bundle, upload data,
and collect outputs. If your site requires extra server-side activation, that is
an operator concern rather than part of the user demo.

If you rerun the demo against a shared bucket, clear the corresponding
`*-output` bucket first or use unique filenames so old reports do not mix into
the next download.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_BUCKET` | `user1-video` | S3 bucket alias |
| `MAX_VIDEOS` | `6` | Max videos to upload |
| `UMRS_CONF` | `~/.config/umrs/conf_umrs.json` | Server config |

### workflow/workflow.json Tuning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `diff_threshold` | 30 | Pixel intensity change needed to flag motion |
| `area_threshold` | 500 | Minimum contour area (px²) to count |
| `min_event_frames` | 3 | Min consecutive frames for a motion event |
| `sample_fps` | 5 | Sampling rate (0 = every frame) |

Lower `diff_threshold` for subtle motion; raise for noisy footage.

## Output

```
results/
├── test_video_000_motion_motion.json
├── test_video_001_motion_motion.json
├── test_video_002_static_motion.json  # should have 0 events
├── test_video_003_motion_motion.json
└── summary.json                       # aggregate
```

Each `*_motion.json` contains:

```json
{
  "source_key": "test_video_000_motion.mp4",
  "total_frames": 150,
  "video_fps": 15.0,
  "duration_seconds": 10.0,
  "motion_events": 3,
  "events": [
    {
      "start_frame": 0, "end_frame": 45,
      "start_sec": 0.0, "end_sec": 3.0, "duration_sec": 3.0
    }
  ]
}
```

## Extending

To add a second analysis pass (e.g. optical flow, object tracking):

1. Create `workflow/optical_flow_step.py`
2. Add a second entry to `workflow/workflow.json → steps[]`
3. Re-upload: `umrs-client workflow upload workflow/`

## See Also

- **Demo 08** — Step/workflow.json basics
- **Demo 09** — Multi-step image pipeline
- **Demo 12** — Pipeline SDK (6-phase lifecycle)
