# Demo 04: Vision Model Inference

Generic vision model inference using the UMRS BentoML pipeline.

This demo shows how to upload images to UMRS and have them processed by
a vision model running on HPC (via BentoML + Slurm). By default it uses
SAM3 segmentation, but the same interface works with **any** registered
vision model.

## Prerequisites

- UMRS server running with a vision-capable compute backend
- `umrs-client` installed and in PATH
- Server config with a vision experience (e.g. SAM3 Segmentation)

## Quick Start

```bash
cd demos/04_vision_inference
bash run_demo.sh
```

## Manual Steps

```bash
# 1. Ping server + reload config
umrs-client -c ~/.config/umrs/conf_umrs.json ping
umrs-client -c ~/.config/umrs/conf_umrs.json reload

# 2. Upload images
umrs-client -c ~/.config/umrs/conf_umrs.json \
    upload_dir ../sample_data/images user1-sam3

# 3. Wait for processing and download results
umrs-client -c ~/.config/umrs/conf_umrs.json \
    download_results user1-sam3 results/ --force-overwrite
```

## Plugging in a Different Vision Model

The UMRS BentoML vision service supports multiple model adapters. To use
your own model:

1. **Register** the model with UMRS:
   ```bash
   umrs-client model register my-model /path/to/model --type embedding --scope system
   ```

2. **Add an experience** to your config JSON that references the model.

3. **Reload** the server:
   ```bash
   umrs-client reload
   ```

4. **Run** with a custom bucket:
   ```bash
   VISION_BUCKET=my-bucket bash run_demo.sh /path/to/images
   ```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_BUCKET` | `user1-sam3` | Bucket alias for image uploads |
| `MAX_IMAGES` | `3` | Limit number of images to upload |
| `FORCE_RECOMPUTE` | *(unset)* | Set to `1` to force re-processing |
| `IMAGE_DIR` | `../sample_data/images` | Override via first argument |

## How It Works

1. Images are uploaded to MinIO (S3) via `umrs-client upload_dir`
2. The server detects new objects and dispatches them to the BentoML vision
   service running on HPC (Slurm)
3. The vision service loads the model, runs inference, and writes results
   back to S3
4. `download_results` retrieves the output files to your local machine
