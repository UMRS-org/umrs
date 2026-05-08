# Generic Vision Model Serving

## Overview

The UMRS vision service provides a **generic, pluggable framework** for serving
any vision model through a unified BentoML-based API. It supports:

- **Standard HuggingFace models** via the generic `huggingface` adapter (auto-loads any HF vision model)
- **Custom research models** via dedicated adapters (SAM3, HMAE, and user-defined)

## Architecture

```
BentoML Vision Service (Podman container on Slurm node)

Endpoints:
  POST /infer/submit  -> async inference task
  GET  /models        -> list available adapters and capabilities
  GET  /health        -> readiness check

Adapter Registry (auto-discovered from /adapters/)
  - sam3: custom SAM3 adapter
  - hmae: custom HMAE adapter
  - huggingface: generic HuggingFace adapter
  - user adapters: mounted from /adapters/user

LRU Model Cache:
  MAX_VISION_MODELS=3
```

## Quick Start

### Serving a HuggingFace model (generic)

```bash
# Build the container
podman build -t umrs-vision-service:latest -f Containerfile.vision .

# Run with a pre-staged HuggingFace model
podman run --rm --device nvidia.com/gpu=all \
  -v /models:/models:ro \
  -p 3000:3000 \
  -e TRANSFORMERS_OFFLINE=1 \
  umrs-vision-service:latest

# Submit inference
curl -X POST http://localhost:3000/infer/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "huggingface",
    "inference_config": {
      "model_id": "google/vit-base-patch16-224",
      "task": "image-classification"
    },
    "image_url": "s3://bucket/image.png",
    "presigned_input_url": "https://s3.example.com/presigned..."
  }'
```

### Serving SAM3 (custom adapter)

```bash
curl -X POST http://localhost:3000/infer/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "sam3",
    "inference_config": {
      "prompt": "anomaly",
      "threshold": 0.5
    },
    "image_url": "s3://spring8-data/ct_slice_001.tif",
    "presigned_input_url": "https://...",
    "presigned_output_urls": {
      "ct_slice_001_sam3_seg.png": "https://...",
      "ct_slice_001_sam3_results.pkl": "https://..."
    }
  }'
```

### Serving HMAE (custom research model)

```bash
curl -X POST http://localhost:3000/infer/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "hmae",
    "inference_config": {
      "task": "reconstruction",
      "config_path": "/models/hmae/configs/infer_hmae_vit_b.yaml",
      "checkpoint": "/models/hmae/checkpoints/latest_checkpoint.pth"
    },
    "image_url": "s3://spring8-data/ct_slice_001.tif",
    "presigned_input_url": "https://...",
    "presigned_output_urls": {...}
  }'
```

## Supported Task Types

### Via Generic HuggingFace Adapter

| Task | Example Models |
|------|---------------|
| `image-classification` | ViT, DINOv2, ResNet |
| `object-detection` | DETR, YOLO (HF) |
| `image-segmentation` | Mask2Former, SegFormer |
| `image-feature-extraction` | DINOv2, CLIP |
| `zero-shot-image-classification` | CLIP |
| `depth-estimation` | DPT, Depth Anything |
| `image-to-image` | Pix2Pix, Super-resolution |

### Via Custom Adapters

| Adapter | Tasks |
|---------|-------|
| `sam3` | Text-prompted segmentation |
| `hmae` | Reconstruction, MAE reconstruction, Feature extraction |

## Writing a Custom Adapter

Create a `.py` file in `/adapters/user/` (or a directory pointed to by
`VISION_ADAPTERS_DIR`):

```python
from umrs.vision_service.adapters.base import VisionModelAdapter

class MyModelAdapter(VisionModelAdapter):
    def capabilities(self):
        return {
            "name": "my_model",
            "supported_tasks": ["custom:my-task"],
            "gpu_memory_mb": 4096,
        }

    def describe_outputs(self, key, config):
        base = key.rsplit(".", 1)[0]
        return {f"{base}_output.png": "image/png"}

    def cache_key(self, config):
        return f"my_model:{config.get('checkpoint', 'default')}"

    def load(self, config):
        # Load your model here
        ...

    def infer(self, model, image, bucket, key, config, s3,
              output_bucket, presigned_output_urls=None,
              raw_image_bytes=None):
        # Run inference here
        ...
```

## Deployment on Slurm (Podman)

The UMRS orchestrator generates Slurm job scripts that:
1. Allocate a GPU node
2. Launch the vision service in a Podman container
3. Wait for the `/health` endpoint to respond
4. Write a job info file for discovery

```bash
# Generated script runs:
podman run --rm --device nvidia.com/gpu=all \
  -v /shared/models:/models:ro \
  -v /home/user/adapters:/adapters/user:ro \
  -e VISION_ADAPTERS_DIR=/adapters/system:/adapters/user \
  -e MAX_VISION_MODELS=3 \
  -e TRANSFORMERS_OFFLINE=1 \
  -p 3000:3000 \
  umrs-vision-service:latest
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_ADAPTERS_DIR` | `/adapters/system:/adapters/user` | Adapter discovery paths |
| `VISION_ADAPTERS` | (empty) | Explicit `module:Class` entries |
| `MAX_VISION_MODELS` | `3` | LRU cache capacity |
| `VISION_REQUIRE_PRESIGNED_URLS` | `false` | Enforce presigned I/O |
| `VISION_S3_PRESIGN_ENDPOINT` | (empty) | Override S3 endpoint for presigned URLs |
| `HF_HOME` | `/models/.hf_cache` | HuggingFace cache dir |
| `TRANSFORMERS_OFFLINE` | `1` | Disable network access (use pre-staged models) |
| `HF_DEFAULT_DEVICE` | `auto` | Default device for HF models |
| `BENTOML_PORT` | `3000` | Service listen port |
| `BENTOML_TRUSTED` | `true` | Allow pickle deserialization |
