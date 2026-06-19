# Demo 14: Opossum Motion Analysis (Production Pipeline)

GPU-accelerated video motion analysis using the same frame-difference pipeline
as the production `tmp/opossum-video-analysis/` engine.

## Pipeline Stages

1. **VFR→CFR conversion** (ffmpeg) — real trail-camera footage is VFR
2. **Frame-difference motion detection** — binarize → absdiff → morphology → merge
3. **Per-camera incremental tracking** — checkpoint files skip already-processed videos
4. **Per-pair TSV output** — one TSV per video pair, plus aggregated `motion_events.tsv`

GPU acceleration uses `cv2.cuda` (OpenCV CUDA module) when available.

## Directory Layout

```
workflow/
  opossum_step.py      — Pipeline SDK step (production logic)
  workflow.json         — workflow spec (GPU, 8 CPU, 16 GiB)
  opossum/              — bundled opossum engine (auto-copied by README.md (step-by-step commands))
generate_test_videos.py — synthetic test-video generator
README.md (step-by-step commands)             — shell entrypoint (upload, workflow, results)
experience.json         — UMRS experience definition
results/                — output directory
sample_videos/          — synthetic test videos (GEN_TEST_VIDEOS=1)
```

## Quick Start

```bash
# Production pipeline with real data
# Follow the step-by-step commands in README.md ../../tmp/opossum-video-analysis/data

# GPU-accelerated server execution
GPU_ENABLED=1 # Follow the step-by-step commands in README.md ../../tmp/opossum-video-analysis/data

# Synthetic test videos (no real data needed)
GEN_TEST_VIDEOS=1 # Follow the step-by-step commands in README.md

# Limit videos per camera
MAX_VIDEOS=10 # Follow the step-by-step commands in README.md ../../tmp/opossum-video-analysis/data
```

## Data Source

Real data from `tmp/opossum-video-analysis/data/`:
```
opossum_215_A/
  opossumA/Schedule/20260330/00/20260330_000001.mp4
                                 20260330_001132.mp4
                                 ...
                            01/...
                            02/...
```

Multi-camera layout: `opossum_NNN_X/opossumX/Schedule/YYYYMMDD/HH/*.mp4`

## Manual Steps

```bash
# 1. Verify server connectivity
umrs-client ping
# Expected: "✓ Server ready | profile=rccs-cloud | data0.ai.r-ccs.riken.jp:50090"

# 2. Register the experience into your user config
#    Merges the "Opossum Analysis" experience (bucket alias "demo-opossum")
#    into ~/.config/umrs/users/<username>.json
umrs-client experience add experience.json --update
# Expected: "✓ Added experience(s): Opossum Analysis"
# Common failure: "Fragment not found" — run from the demo directory.

# 3. Reload so the server provisions the MinIO bucket
umrs-client reload
# Expected: "✓ reload accepted"
# Common failure: "HTTP 503: Service Unavailable" — server reload endpoint
# not initialised; restart the server or check `umrs-server` logs.

# 4. Verify the demo-opossum bucket alias is now configured
umrs-client list_buckets
# Expected: a row for "Opossum Analysis" with alias "demo-opossum"
#           Remote = synced, Objects = 0
# Common failure: "Bucket alias 'demo-opossum' not found" — repeat
#                 steps 2-3; the experience must be reloaded on the server.

# 5. Bundle opossum engine into workflow
cp -r ../../tmp/opossum-video-analysis/src/opossum workflow/opossum

# 6. Upload workflow (includes bundled engine)
umrs-client upload-workflow workflow/ --name opossum-analysis
# Expected: workflow upload success message

# 7. Upload videos (preserves camera/Schedule date hierarchy)
umrs-client upload_dir ../../tmp/opossum-video-analysis/data demo-opossum
# Expected: 47 files uploaded; final JSON with
#           bucket=mnagaso.opossum-analysis.demo-opossum, modality=movie
# Common failure: "Cannot process directory" — fixed in client_umrs.py
#                 (upload_dir now recurses into subdirectories).
# Common failure: "Bucket alias 'demo-opossum' not found" — repeat
#                 steps 2-3 to register and reload the experience.

# 8. Download results
umrs-client download_results demo-opossum results/ --force-overwrite
# Expected: per-camera TSVs, motion_events.tsv, analysis_summary.json
#           written under results/
```

## Dependencies

- `opencv-python-headless` with CUDA support (for GPU acceleration)
- `numpy`
- `ffmpeg` + `ffprobe` (for VFR→CFR conversion)
- Python ≥3.11

## Defaults

| Setting | Default |
|---|---|
| Bucket alias | `demo-opossum` |
| GPU mode | 0 (CPU) — set `GPU_ENABLED=1` for GPU |
| Distance threshold | 0.05 (ratio of frame width) |
| Time threshold | 15 seconds |
| Result dir | `results/` |

## Output

```
results/
  motion_events.tsv           — aggregated: camera, video_pair, start_time, end_time
  analysis_summary.json       — GPU mode, camera counts, event counts
  opossum_215_A/opossumA/     — per-camera per-pair TSVs + checkpoints
    20260330_000001.tsv
    20260330_001132.tsv
    last_target.txt
```

## External Dependency

The opossum engine (`tmp/opossum-video-analysis/src/opossum/`) is bundled into the
workflow directory when `README.md (step-by-step commands)` runs (step 3). It is NOT checked into version
control.

## Public Analogue

The public replacement for a generic video-analysis workflow is:

- [../../demos/14_video_analysis/README.md](../../demos/14_video_analysis/README.md)

## Related Docs

- [../../demos/12_pipeline_sdk/README.md](../../demos/12_pipeline_sdk/README.md)
- [workflow/workflow.json](workflow/workflow.json)
