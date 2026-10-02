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
# See "Running Manually" section below for step-by-step commands
cd "$(git rev-parse --show-toplevel)/$(test -d "$(git rev-parse --show-toplevel)/umrs/demos" && echo umrs)/demos/10_vision_multi_mode"

# With your own images
IMAGE_DIR=/path/to/images MULTI_BUCKET=my-analysis bash ./run_demo.sh
```

> **Note:** To navigate to this demo from either repo level, use:
> ```bash
> cd "$(git rev-parse --show-toplevel)/$(test -d "$(git rev-parse --show-toplevel)/umrs/demos" && echo umrs)/demos/10_vision_multi_mode"
> ```

> **Note:** The inline script `run_demo.sh` does not exist yet. Use the
> step-by-step commands in the "Running Manually" section below instead.

## Steps

| Branch | Class | Hardware | Output |
|--------|-------|----------|--------|
| `segment` | `SegmentModeStep` | GPU | `segment/*_segment.json` — bounding boxes |
| `features` | `FeatureModeStep` | GPU | `features/*_features.json` — embedding vectors |
| `thumbnails` | `ThumbnailModeStep` | CPU | `thumbnails/*_128px.jpg`, `*_256px.jpg` |

## How Parallel Execution Works

Each step in `workflow.json` declares the steps it waits for. All three
start the run, so each says `"depends_on": []`:

```json
{"name": "segment",    "entrypoint": "segment_mode:SegmentModeStep",     "depends_on": []},
{"name": "features",   "entrypoint": "feature_mode:FeatureModeStep",     "depends_on": []},
{"name": "thumbnails", "entrypoint": "thumbnail_mode:ThumbnailModeStep", "depends_on": []}
```

Steps with no edge between them receive the **same input keys** and run
concurrently. Each branch writes to its own output prefix so there are no
conflicts. (Until 2026-10-02 this demo used the `execution.parallel`
shorthand; the runner still reads it, but an upload must spell the graph
out in `depends_on`.)

## Running Manually

```bash
# 1. Verify server connectivity
umrs ping

# 2. Load the experience fragment into your user config
#    This registers the multi-mode vision experience (model + bucket + table)
umrs experience add experience.json

# 3. Reload to apply the new experience and provision buckets
umrs reload

# 4. Upload the multi-mode workflow to the server
#    The upload-ready bundle lives under workflow/. Its workflow.json defines
#    3 parallel branches (segment, features,
#    thumbnails). Each branch processes the same input images independently.
umrs workflow upload workflow/ --name "vision-multi-mode"

# 5. Upload images to the multi-mode bucket
#    All 3 branches receive the same images simultaneously:
#    - segment:    runs GPU-based segmentation (bounding boxes)
#    - features:   runs GPU-based feature extraction (embedding vectors)
#    - thumbnails: runs CPU-based thumbnail generation (128px, 256px)
umrs data dir-upload ../sample_data/images user1-multi

# 6. Download results from all 3 branches
#    Results are organized by branch in the output:
#    S3 objects are written under segment/, features/, and thumbnails/.
#    The local client currently flattens those prefixes when downloading.
umrs data download user1-multi ./results --force-overwrite
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

Register the model first via Demo 07 (`umrs model register ...`).

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
