# Demo 10: Multi-Mode Vision Pipeline

Run **three analysis modes in parallel** on the same set of images,
demonstrating UMRS's parallel branch execution.

## Architecture

```
                 ┌─────────────────┐
                 │   Input Images   │
                 └───────┬─────────┘
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ 1. Segment   │ │ 2. Features  │ │ 3. Thumbnails│
│   (GPU)      │ │   (GPU)      │ │   (CPU)      │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       ▼                ▼                ▼
   segment/         features/       thumbnails/
```

All three branches start simultaneously. GPU branches share the BentoML
vision service; the thumbnail branch runs on CPU and finishes quickly.

## Quick Start

```bash
bash 10_vision_multi_mode/run_demo.sh

# Custom image directory
bash 10_vision_multi_mode/run_demo.sh /path/to/images

# Custom bucket
MULTI_BUCKET=my-analysis bash 10_vision_multi_mode/run_demo.sh
```

## Steps

| Branch | Class | Hardware | Output |
|--------|-------|----------|--------|
| `segment` | `SegmentModeStep` | GPU | `segment/*_segment.json` — bounding boxes |
| `features` | `FeatureModeStep` | GPU | `features/*_features.json` — embedding vectors |
| `thumbnails` | `ThumbnailModeStep` | CPU | `thumbnails/*_128px.jpg`, `*_256px.jpg` |

## How Parallel Execution Works

The `workflow.json` includes an `execution.parallel` list:

```json
"execution": {
  "parallel": ["segment", "features", "thumbnails"]
}
```

Steps listed in `parallel` receive the **same input keys** and run
concurrently.  Each branch writes to its own output prefix so there
are no conflicts.

## Running Manually

```bash
# 1. Verify server connectivity
umrs-client ping

# 2. Upload the multi-mode workflow to the server
#    The workflow.json defines 3 parallel branches (segment, features,
#    thumbnails). Each branch processes the same input images independently.
umrs-client upload-workflow . --name "vision-multi-mode"

# 3. Reload server configuration
#    Picks up the newly uploaded workflow definition.
umrs-client reload

# 4. Upload images to the multi-mode bucket
#    All 3 branches receive the same images simultaneously:
#    - segment:    runs GPU-based segmentation (bounding boxes)
#    - features:   runs GPU-based feature extraction (embedding vectors)
#    - thumbnails: runs CPU-based thumbnail generation (128px, 256px)
umrs-client upload_dir ../sample_data/images user1-multi

# 5. Download results from all 3 branches
#    Results are organized by branch in the output:
#    results/segment/   — segmentation JSON files
#    results/features/  — feature vector JSON files
#    results/thumbnails/ — resized JPEG thumbnails
umrs-client download_results user1-multi ./results --force-overwrite
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTI_BUCKET` | `user1-multi` | S3 bucket alias |
| `MAX_IMAGES` | `3` | Maximum images to upload |
| `UMRS_CONF` | `~/.config/umrs/conf_umrs.json` | Server config |

## Using Different Models

Swap the vision model in `workflow.json`:

```json
{
  "model_type": "your-model",
  "bentoml_url": "http://gpu-host:PORT/vision"
}
```

Register the model first via Demo 07 (`umrs-client model register ...`).

## See Also

- **Demo 04** — Single vision model inference
- **Demo 09** — Sequential multi-step pipeline
- **Demo 15** — Complex DAG with dependencies and merging
