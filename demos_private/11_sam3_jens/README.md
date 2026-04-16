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

## Quick Start

```bash
# Default: images from SAM3 demo's test_data_full, presigned S3 URLs
bash run_demo.sh

# Custom image directory
bash run_demo.sh /path/to/images/

# Direct S3 mode (images already in S3)
IMAGE_MODE=s3 BUCKET_ALIAS=user1-sam3 bash run_demo.sh

# Local mode (service reads from shared filesystem)
IMAGE_MODE=local bash run_demo.sh /shared/nfs/images/

# Custom server endpoint
JENS_SERVER=http://compute01 JENS_PORT=3000 bash run_demo.sh
```

## Running Manually

This demo bypasses `umrs-client` for inference and calls the BentoML
service directly. However, `umrs-client` is still used for uploading
images and generating presigned URLs.

```bash
# 1. Check VisionFrontend service readiness
curl -sf http://localhost:3000/readyz && echo "Service ready"

# 2. Upload images to S3 via umrs-client (for presigned URL mode)
#    The images are uploaded to MinIO so the BentoML service can
#    access them via presigned URLs.
umrs-client upload_dir ./test_images user1-sam3

# 3. Generate a presigned URL for each image
#    The BentoML service fetches images via these URLs.
umrs-client presign user1-sam3 image.png

# 4. Submit SAM3 inference via the jens_client.py helper
#    This calls POST /vision_task directly on the VisionFrontend,
#    bypassing the UMRS orchestrator for lower latency.
python3 ../lib/jens_client.py \
    --server http://localhost \
    --port 3000 \
    --model sam3 \
    --task "all" \
    --image "https://presigned-url-from-step-3" \
    --output-dir ./results \
    --timeout 300

# 5. Results are saved as pickle files in results/
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
| `IMAGE_MODE` | `presigned` | `presigned`, `s3`, or `local` |
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
