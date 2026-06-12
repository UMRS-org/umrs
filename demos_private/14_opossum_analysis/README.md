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
  opossum/              — bundled opossum engine (auto-copied by run_demo.sh)
generate_test_videos.py — synthetic test-video generator
run_demo.sh             — shell entrypoint (upload, workflow, results)
experience.json         — UMRS experience definition
results/                — output directory
sample_videos/          — synthetic test videos (GEN_TEST_VIDEOS=1)
```

## Quick Start

```bash
# Production pipeline with real data
bash run_demo.sh ../../tmp/opossum-video-analysis/data

# GPU-accelerated server execution
GPU_ENABLED=1 bash run_demo.sh ../../tmp/opossum-video-analysis/data

# Synthetic test videos (no real data needed)
GEN_TEST_VIDEOS=1 bash run_demo.sh

# Limit videos per camera
MAX_VIDEOS=10 bash run_demo.sh ../../tmp/opossum-video-analysis/data
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

# 2. Bundle opossum engine into workflow
cp -r ../../tmp/opossum-video-analysis/src/opossum workflow/opossum

# 3. Upload workflow (includes bundled engine)
umrs-client upload-workflow workflow/ --name opossum-analysis

# 4. Upload videos (preserves camera/Schedule date hierarchy)
umrs-client upload_dir ../../tmp/opossum-video-analysis/data demo-opossum

# 5. Download results
umrs-client download_results demo-opossum results/ --force-overwrite
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
workflow directory when `run_demo.sh` runs (step 3). It is NOT checked into version
control.

## Public Analogue

The public replacement for a generic video-analysis workflow is:

- [../../demos/14_video_analysis/README.md](../../demos/14_video_analysis/README.md)

## Related Docs

- [../../demos/12_pipeline_sdk/README.md](../../demos/12_pipeline_sdk/README.md)
- [workflow/workflow.json](workflow/workflow.json)
