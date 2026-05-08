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
#    The upload-ready bundle lives under workflow/. Its workflow.json defines
#    3 parallel branches (segment, features,
#    thumbnails). Each branch processes the same input images independently.
umrs-client upload-workflow workflow/ --name "vision-multi-mode"

# 3. Upload images to the multi-mode bucket
#    All 3 branches receive the same images simultaneously:
#    - segment:    runs GPU-based segmentation (bounding boxes)
#    - features:   runs GPU-based feature extraction (embedding vectors)
#    - thumbnails: runs CPU-based thumbnail generation (128px, 256px)
umrs-client upload_dir ../sample_data/images user1-multi

# 4. Download results from all 3 branches
#    Results are organized by branch in the output:
#    S3 objects are written under segment/, features/, and thumbnails/.
#    The local client currently flattens those prefixes when downloading.
umrs-client download_results user1-multi ./results --force-overwrite
```

If you rerun the demo against a shared bucket, clear the corresponding
`*-output` bucket first or use unique filenames so old artifacts do not mix
into the new download.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTI_BUCKET` | `user1-multi` | S3 bucket alias |
| `MAX_IMAGES` | `3` | Maximum images to upload |
| `UMRS_CONF` | `~/.config/umrs/conf_umrs.json` | Server config |

## Using Different Models

Swap the vision model in `workflow/workflow.json`:

```json
{
  "model_type": "your-model",
  "bentoml_url": "http://gpu-host:PORT/vision"
}
```

Register the model first via Demo 07 (`umrs-client model register ...`).

This demo is intentionally written from the general-user point of view: the user
uploads the workflow and data, and the server executes the configured branches.
Operator-only reload/maintenance steps are omitted from the normal flow.

## Output

Objects on S3 are grouped by branch prefixes:

```text
segment/demo10_cat_segment.json
features/demo10_cat_features.json
thumbnails/demo10_cat_128px.jpg
```

`download_results` currently flattens those keys into a single local directory,
so a local run typically looks like this:

```text
results/
├── demo10_cat_segment.json
├── demo10_cat_features.json
├── demo10_cat_128px.jpg
├── demo10_cat_256px.jpg
├── demo10_cat_sam3_seg.png
├── demo10_cat_sam3_results.pkl
├── demo10_cat_features_results.pkl
├── demo10_dog_segment.json
├── demo10_dog_features.json
├── demo10_dog_128px.jpg
└── ...
```

The `*_segment.json`, `*_features.json`, and `*_{128,256}px.jpg` files are the
branch outputs. The `*_sam3_*` and `*_features_results.pkl` files are raw model
artifacts produced alongside those branch outputs.

## See Also

- **Demo 04** — Single vision model inference
- **Demo 09** — Sequential multi-step pipeline
- **Demo 15** — Complex DAG with dependencies and merging
