# Demo 11: SAM3 Segmentation — Jens Direct Mode

Runs SAM3 segmentation by calling Jens's **VisionFrontend** BentoML service
directly, bypassing the UMRS generic vision pipeline.

## Architecture

```
run_demo.sh → jens_client.py → VisionFrontend /vision_task
                                   └── SAM3 model (GPU)
                                   └── Image fetch (S3/HTTP/local)
                                   └── Result → pickle / S3 upload
```

The key difference from `demos/03_sam3_segmentation/` is that this demo
**does not go through UMRS's orchestrator or upload/download pipeline**.
It communicates with the BentoML service endpoint directly.

## Prerequisites

1. Jens VisionFrontend running on a compute node (BentoML port 3000)
2. Python 3 with `bentoml>=1.4` installed
3. Network access to the VisionFrontend (SSH tunnel if remote)

## Current Validation Note

The current `rccs-cloud` validation environment does not expose a ready
VisionFrontend at `http://localhost:3000/readyz`, so this demo is blocked until
that service is started or tunneled locally.

Also note that the current `umrs-client` does not expose a dedicated
`presign` subcommand. The checked-in `run_demo.sh` already degrades to `s3://`
references automatically, so prefer `IMAGE_MODE=s3` or `IMAGE_MODE=local` when
running manually.

## Quick Start

```bash
# Default: images from SAM3 demo's test_data_full
bash run_demo.sh

# Custom image directory
bash run_demo.sh /path/to/images/

# Direct S3 mode (recommended with the current CLI surface)
IMAGE_MODE=s3 BUCKET_ALIAS=user1-sam3 bash run_demo.sh

# Local mode (service reads from shared filesystem)
IMAGE_MODE=local bash run_demo.sh /shared/nfs/images/

# Custom server endpoint
JENS_SERVER=http://compute01 JENS_PORT=3000 bash run_demo.sh
```

## Running Manually

This demo bypasses `umrs-client` for inference and calls the BentoML
service directly. `umrs-client` is still used for staging images, but in the
current CLI the practical manual paths are `s3://` or local-file input.

```bash
# 1. Check VisionFrontend service readiness
curl -sf http://localhost:3000/readyz && echo "Service ready"

# 2. Upload images to S3 via umrs-client
#    The images are uploaded to MinIO so the BentoML service can
#    access them via direct `s3://` URIs.
umrs-client upload_dir ./test_images user1-sam3

# 3. Submit SAM3 inference via the jens_client.py helper
#    This calls POST /vision_task directly on the VisionFrontend,
#    bypassing the UMRS orchestrator for lower latency.
python3 ../lib/jens_client.py \
    --server http://localhost \
    --port 3000 \
    --model sam3 \
    --task "all" \
    --image "s3://user1-sam3/image.png" \
    --output-dir ./results \
    --timeout 300

# 4. Results are saved as pickle files in results/
ls results/*.pkl
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JENS_SERVER` | `http://localhost` | VisionFrontend host |
| `JENS_PORT` | `3000` | VisionFrontend port |
| `JENS_TOKEN` | _(none)_ | Bearer token (skip if `NO_AUTH=1` on server) |
| `SAM3_PROMPT` | `all` | Object to segment (e.g. `cat`, `crack`, `all`) |
| `MAX_IMAGES` | `3` | Max images to process |
| `IMAGE_MODE` | `presigned` | `presigned`, `s3`, or `local` (`presigned` currently falls back to `s3://`) |
| `BUCKET_ALIAS` | `user1-sam3` | S3 bucket for presigned/s3 mode |

## Output

Results are saved as pickle files in `results/`:

```
results/
├── image1_sam3_all.pkl      # pickled dict
├── image2_sam3_all.pkl
└── ...
```

Each pickle contains:
- `status`: `"success"` or error
- `masks_logits`: raw mask logits (tensor)
- `masks`: binary masks (tensor)
- `boxes`: bounding boxes (tensor)
- `scores`: confidence scores (tensor)

## Comparison with Demo 03

| | Demo 03 (Generic) | Demo 11 (Jens Direct) |
|---|---|---|
| Path | UMRS orchestrator → BentoML | Direct BentoML call |
| Auth | UMRS config | Bearer token / NO_AUTH |
| Image input | S3 upload via UMRS | Presigned URL / S3 URI / local |
| Output | S3 download via UMRS | Pickle file / S3 URI |
| Latency | Higher (orchestrator overhead) | Lower (direct) |
| Reliability | UMRS retry + monitoring | Simple polling |
