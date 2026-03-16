# Demo 03: HMAE Feature Extraction

User-facing demo — upload images, run HMAE (Hierarchical Masked Auto-Encoder)
inference via Mini-Rivault's BentoML vision service, and retrieve results.

## Prerequisites

- Services running (see [Demo 00](../00_start_services/README.md))
- **HMAE model checkpoint and config pre-staged** by admin at `$HMAE_MODEL_PATH`
  (default: `./rivault/models/hmae/`)
- UMRS client installed (`pip install .` from `src/`)

## Architecture

Same as SAM3 — BentoML vision service runs inside Mini-Rivault sandbox:

```
Client → UMRS Server → Slurm sbatch → Mini-Rivault
              │                         ├── NGINX (:4000, published)
              │                         └── BentoML-vision (:3000, internal)
              │                              └── HMAE model (GPU)
              └──────── SSH tunnel ─────┘
                        (NGINX /vision/ prefix)
```

## Quick Start

```bash
cd demos/03_hmae_inference
bash run_demo.sh
```

## What This Demo Does

1. Verifies services and model weights are available
2. Activates Mini-Rivault `vision-bentoml` profile
3. Uploads test images (X-ray CT slices) to S3
4. Submits HMAE inference tasks via BentoML async task queue
5. Supports three modes:
   - **MAE Reconstruction** — mask 50% of patches, reconstruct
   - **Full Reconstruction** — no masking, full decode
   - **Feature Extraction** — encoder-only, outputs CLS + patch features

## HMAE Inference Modes

| Mode | Output | Use Case |
|------|--------|----------|
| `mae` | Side-by-side: original / quadtree grid + mask / reconstruction | Quality assessment, anomaly detection |
| `full` | Side-by-side: original / quadtree grid / full reconstruction | Visualize model understanding |
| `features` | `.pt` file with CLS token + patch features + coordinates | Downstream classification, similarity search |

## Running Manually

```bash
# Ensure vision profile is active
export RIVAULT_PROFILES="harness vllm vision-bentoml"

# Run all 3 modes
python3 ../../demos/slurm-dev/scripts/demo_hmae_inference.py \
    --bentoml-url http://localhost:$NGINX_PORT/vision \
    --s3-endpoint http://localhost:19000 \
    --image-dir ./test_data/ \
    --all

# Run a specific mode
python3 ../../demos/slurm-dev/scripts/demo_hmae_inference.py \
    --bentoml-url http://localhost:$NGINX_PORT/vision \
    --image-path ./test_data/volume_170.png \
    --mode features
```

## Configuration

### Mini-Rivault Profile

```bash
export RIVAULT_PROFILES="harness vllm vision-bentoml"
```

### Environment (`env.vision`)

```bash
HMAE_MODEL_PATH=./rivault/models/hmae
HMAE_CONFIG_PATH=./rivault/models/hmae/infer_hmae_vit_b.yaml
HMAE_CHECKPOINT=./rivault/models/hmae/latest_checkpoint.pth
RIVAULT_PROFILES="harness vllm vision-bentoml"
```

### Vision Model Config (`vision_models.json`)

```json
{
  "modality_name": "tomography",
  "model": {
    "model_name": "HMAE ViT-B",
    "vector_size": "768",
    "model_type": "bentoml://http://localhost:{LOCAL_PORT}/vision/hmae_reconstruct/submit",
    "model_cfg_file": "/models/hmae/infer_hmae_vit_b.yaml",
    "checkpoint_path": "/models/hmae/latest_checkpoint.pth",
    "return_type": "file"
  }
}
```

## BentoML Task Queue API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vision/hmae_reconstruct/submit` | POST | Submit HMAE task |
| `/vision/hmae_reconstruct/status` | GET | Check task status |
| `/vision/hmae_reconstruct/get` | GET | Retrieve result |
| `/vision/hmae_reconstruct/cancel` | POST | Cancel pending task |
| `/vision/hmae_reconstruct/retry` | PUT | Retry failed task |

## See Also

- [Admin Guide](../../doc/admin_guide.md) — deploying vision models
- [Demo 02: SAM3](../02_sam3_segmentation/README.md) — segmentation demo
- [Mini-Rivault README](../../src/umrs/rivault/README.md) — inference stack details
