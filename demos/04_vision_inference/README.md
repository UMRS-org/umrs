# Demo 04: Vision Model Inference

Generic vision model inference using the UMRS BentoML pipeline.

This demo shows how to upload images to UMRS and have them processed by
a vision model running on HPC (via BentoML + Slurm). By default it uses
SAM3 segmentation, but the same interface works with **any** registered
vision model.

## Prerequisites

- UMRS server running with a vision-capable compute backend
- `umrs` installed and in PATH (see [demos/README.md](../README.md#prerequisites))
- User onboarded (see [Demo 00](../00_onboarding/))
- Server config with a vision experience (e.g. SAM3 Segmentation)
- A site/user bucket already configured for the vision model you want to exercise

## Quick Start

```bash
# Navigate from any level (auto-detects parent repo vs umrs subdirectory)
cd "$(git rev-parse --show-toplevel)/$(test -d "$(git rev-parse --show-toplevel)/umrs/demos" && echo umrs)/demos/04_vision_inference"
# See the numbered steps below in this README
```

## Running Manually

```bash
# 1. Verify server connectivity
#    Sends a ZMQ ping to confirm the server is alive.
umrs ping

# 2. Load the experience fragment into your user config
#    This registers the vision inference experience (model + bucket + table)
umrs experience add experience.json

# 3. Reload to apply the new experience and provision buckets
umrs reload

# 4. Upload images to the vision bucket
#    Images are stored in S3 and the server dispatches each one
#    to the BentoML vision model running on HPC via Slurm.
#    The first upload may trigger a cold start (Slurm job launch).
umrs data dir-upload ../sample_data/images user1-sam3

# 5. Download inference results
#    Polls the server until processing completes, then retrieves
#    all output files (JSON metadata, overlay images, etc.) to
#    the local results/ directory.
umrs data download user1-sam3 results/ --force-overwrite
```

## Plugging in a Different Vision Model

The UMRS BentoML vision service supports multiple model adapters. To use
your own model:

1. **Register** the model with UMRS:
   ```bash
   umrs model register my-model /path/to/model --type vision --scope user
   ```

   On a local-client / remote-server deployment, use `model upload` with an
   explicit `--register-path` when the model files only exist on your laptop.
   `system` scope is admin-only.

2. **Add an experience** to your config JSON that references the model.

3. **Run** with a custom bucket or a user-visible model name:
   ```bash
   VISION_BUCKET=my-bucket VISION_MODEL_NAME=my-model /path/to/images
# See the numbered steps below in this README
   ```

If the bucket/experience itself needs to be created or rewired, that is a site
operator task rather than part of the normal user demo.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_BUCKET` | `user1-sam3` | Bucket alias for image uploads |
| `MAX_IMAGES` | `3` | Limit number of images to upload |
| `FORCE_RECOMPUTE` | *(unset)* | Operator-only maintenance; ignored in normal user mode unless `DEMO_ADMIN_ACTIONS=1` |
| `IMAGE_DIR` | `../sample_data/images` | Override via first argument |

## How It Works

1. Images are uploaded to MinIO (S3) via `umrs data dir-upload`
2. The server detects new objects and dispatches them to the BentoML vision
   service running on HPC (Slurm)
3. The vision service loads the model, runs inference, and writes results
   back to S3
4. `download_results` retrieves the output files to your local machine

For SAM3-backed experiences, the result pickle is compact by default so demo
downloads stay practical. Full-resolution mask arrays can be re-enabled on the
server with `SAM3_INCLUDE_MASKS_IN_RESULTS=1`.

This demo intentionally assumes a general-user deployment model: the server-side
bucket alias and model wiring already exist before the user runs the demo.
