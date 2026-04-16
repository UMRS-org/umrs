# Demo 12: HMAE Inference — Jens Direct Mode

Runs HMAE inference by calling Jens's **VisionFrontend** BentoML service
directly, bypassing the UMRS generic vision pipeline.

## Architecture

```
run_demo.sh → jens_client.py → VisionFrontend /vision_task
                                   └── HMAE model (GPU)
                                   └── Image fetch (S3/HTTP/local)
                                   └── Result → pickle / S3 upload
```

## HMAE Modes

| Mode | Task param | Output |
|------|-----------|--------|
| `reconstruct` | `"reconstruct"` | `{prediction: tensor}` — reconstructed image |
| `mae` | `"mae"` | `{prediction, mask, loss}` — MAE with masking |
| `features` | `"features"` | `{features: tensor}` — feature extraction |

## Prerequisites

1. Jens VisionFrontend running on a compute node (BentoML port 3000)
2. HMAE model checkpoint accessible by the service
3. Python 3 with `bentoml>=1.4` installed
4. Volume slice images (PNG format recommended)

## Quick Start

```bash
# Default: reconstruct mode, presigned S3 URLs
bash run_demo.sh

# All 3 modes sequentially
HMAE_MODE=all bash run_demo.sh

# Feature extraction only
HMAE_MODE=features bash run_demo.sh

# MAE mode with custom images
HMAE_MODE=mae bash run_demo.sh /path/to/volume_slices/

# Custom server endpoint
JENS_SERVER=http://compute01 JENS_PORT=3000 bash run_demo.sh

# Direct S3 mode
IMAGE_MODE=s3 BUCKET_ALIAS=user1-hmae bash run_demo.sh
```

## Running Manually

This demo bypasses `umrs-client` for inference and calls the BentoML
service directly. However, `umrs-client` is still used for uploading
images and generating presigned URLs.

```bash
# 1. Check VisionFrontend service readiness
curl -sf http://localhost:3000/readyz && echo "Service ready"

# 2. Upload images to S3 via umrs-client (for presigned URL mode)
umrs-client upload_dir ./volume_slices user1-hmae

# 3. Generate a presigned URL for each image
umrs-client presign user1-hmae volume_170.png

# 4. Submit HMAE inference via the jens_client.py helper
#    --task selects the HMAE mode: reconstruct, mae, or features.
#    Each mode produces different output:
#      reconstruct: reconstructed image (prediction tensor)
#      mae:         masked reconstruction + loss score (anomaly detection)
#      features:    CLS + patch embeddings (feature vectors for downstream use)
python3 ../lib/jens_client.py \
    --server http://localhost \
    --port 3000 \
    --model hmae \
    --task "reconstruct" \
    --image "https://presigned-url-from-step-3" \
    --output-dir ./results/reconstruct \
    --timeout 300

# 5. Run all 3 modes on the same images
for mode in reconstruct mae features; do
    python3 ../lib/jens_client.py \
        --server http://localhost --port 3000 \
        --model hmae --task "$mode" \
        --image "https://presigned-url" \
        --output-dir "./results/$mode" --timeout 300
done

# 6. Results are saved as pickle files organized by mode
ls results/reconstruct/*.pkl
ls results/mae/*.pkl
ls results/features/*.pkl
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JENS_SERVER` | `http://localhost` | VisionFrontend host |
| `JENS_PORT` | `3000` | VisionFrontend port |
| `JENS_TOKEN` | _(none)_ | Bearer token |
| `HMAE_MODE` | `reconstruct` | `reconstruct`, `mae`, `features`, or `all` |
| `MAX_IMAGES` | `3` | Max images to process per mode |
| `IMAGE_MODE` | `presigned` | `presigned`, `s3`, or `local` |
| `BUCKET_ALIAS` | `user1-hmae` | S3 bucket for presigned/s3 mode |

## Output

Results are organized by mode in `results/`:

```
results/
├── reconstruct/
│   ├── volume_170_hmae_reconstruct.pkl
│   └── volume_171_hmae_reconstruct.pkl
├── mae/
│   ├── volume_170_hmae_mae.pkl
│   └── volume_171_hmae_mae.pkl
└── features/
    ├── volume_170_hmae_features.pkl
    └── volume_171_hmae_features.pkl
```

## Comparison with Demo 04

| | Demo 04 (Generic) | Demo 12 (Jens Direct) |
|---|---|---|
| Path | UMRS orchestrator → BentoML | Direct BentoML call |
| Modes | Single mode per run | All modes / any mode |
| Auth | UMRS config | Bearer token / NO_AUTH |
| Image input | S3 upload via UMRS | Presigned URL / S3 URI / local |
| Output | S3 download via UMRS | Pickle file / S3 URI |
