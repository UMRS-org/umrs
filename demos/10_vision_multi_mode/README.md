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
