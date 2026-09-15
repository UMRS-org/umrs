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
umrs-client workflow upload workflow/ --name opossum-analysis
# Expected: workflow upload success message

# 6a. (Optional) Validate the sandbox Dockerfile before running
umrs-client workflow validate-sandbox workflow/ --dry-run
# Expected: ✓ Validation passed

# 6b. (Optional) Visualize the workflow DAG
umrs-client workflow visualize opossum-analysis
# Expected: ASCII art showing the workflow DAG structure

# 7. Upload videos (preserves camera/Schedule date hierarchy)
umrs-client upload_dir ../../tmp/opossum-video-analysis/data demo-opossum
# Expected: 47 files uploaded; final JSON with
#           bucket=mnagaso.opossum-analysis.demo-opossum, modality=movie
# Common failure: "Cannot process directory" — fixed in client_umrs.py
#                 (upload_dir now recurses into subdirectories).
# Common failure: "Bucket alias 'demo-opossum' not found" — repeat
#                 steps 2-3 to register and reload the experience.

# 8. Run the workflow on the uploaded videos
umrs-client workflow run opossum-analysis --bucket demo-opossum
# Expected: "✓ Workflow 'opossum-analysis' execution queued as durable job
#           job-XXXXX (1 steps)"
# The workflow runs server-side: it downloads the videos, runs the
# frame-difference motion detection pipeline, and uploads per-pair TSVs
# and an aggregated motion_events.tsv back to the bucket.

# 8a. (Optional) Check workflow execution status
umrs-client workflow status <execution-id>
# Expected: execution status with progress and step details

# 8b. (Optional) Show error logs if the workflow fails
umrs-client workflow errors <execution-id>
# Expected: error logs for failed Attempts, including per-step errors

# 9. Check job status (optional)
umrs-client job status <job-id-from-step-8>
# Wait until status is "succeeded" or "failed".
umrs-client download_results demo-opossum results/ --force-overwrite
# Expected: per-camera TSVs, motion_events.tsv, analysis_summary.json
#           written under results/
```

## Auto-Trigger

The workflow definition includes `"auto_trigger": true`. This tells the
server to automatically trigger the workflow when data is uploaded to the
bucket, eliminating the need for step 8 (`run-workflow`).

When `auto_trigger` is `true`:
1. Upload data: `umrs-client upload_dir <data-dir> demo-opossum`
2. The server detects the upload and automatically queues the workflow
3. Download results: `umrs-client download_results demo-opossum results/`

When `auto_trigger` is `false` (or omitted):
1. Upload data: `umrs-client upload_dir <data-dir> demo-opossum`
2. Manually trigger: `umrs-client workflow run opossum-analysis --bucket demo-opossum`
3. Download results: `umrs-client download_results demo-opossum results/`

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
| Auto-trigger | `true` — workflow starts automatically when data is uploaded |

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

## Custom Container Images

Each workflow can include a `Dockerfile` alongside `workflow.json`. When
`umrs-client upload-workflow` detects a `Dockerfile`, it:

1. Extracts the workflow zip as before.
2. Runs `podman build -t <workflow-name>:latest` in the extracted directory.
3. Injects `routing_hints.image` into `workflow.json` for all steps that
   do not already declare an image override.

This lets each workflow carry its own dependencies (CUDA-enabled OpenCV,
PyTorch, custom libraries) without polluting the host environment.

### Pre-built Image Reference

If you maintain a private registry, reference a pre-built image directly
in the step definition. This bypasses the server-side build entirely:

```json
{
    "name": "opossum_analyze",
    "entrypoint": "opossum_step:OpossumAnalyzeStep",
    "routing_hints": {
        "image": "registry.example.com/mnagaso/opossum-cuda:latest",
        "gpu": true
    }
}
```

### Mixed Mode

Both mechanisms can coexist in the same workflow. Steps with
`routing_hints.image` use that image; steps without it fall back to the
`Dockerfile`-built image. Steps without any image hint run in-process
(the default `InProcessExecutor`).

- [../../demos/12_pipeline_sdk/README.md](../../demos/12_pipeline_sdk/README.md)
- [workflow/workflow.json](workflow/workflow.json)
