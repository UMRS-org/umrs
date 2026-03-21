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

Each HMAE inference mode maps to its own modality entry:

| Modality | Mode | Output |
|----------|------|--------|
| `tomography_reconstruction` | Full encode+decode | `*_recon.png` |
| `tomography_mae_reconstruction` | Masked reconstruction | `*_mae.png` + loss |
| `tomography_features_extraction` | Encoder-only | `*_features.pt` |

Example entry for full reconstruction:

```json
{
  "modality_name": "tomography_reconstruction",
  "model": {
    "model_name": "HMAE ViT-B",
    "vector_size": "768",
    "model_type": "HMAE",
    "model_cfg_file": "/models/hmae/infer_hmae_vit_b.yaml",
    "checkpoint_path": "/models/hmae/latest_checkpoint.pth",
    "return_type": "file"
  }
}
```

## Adapter Architecture

The HMAE model is loaded as a **pluggable adapter** (`hmae_adapter.py`) — the
same mechanism used for all vision models including SAM3 and user-provided adapters.

At container startup, the BentoML vision service scans directories listed in
`VISION_ADAPTERS_DIR` and auto-registers every `VisionModelAdapter` subclass.
Built-in adapters ship in `rivault/default_adapters/` and are loaded via the
same auto-discovery mechanism as user adapters.

See [Demo 07: User-Defined Model](../07_user_model/README.md) for writing
your own VisionModelAdapter.

## BentoML Task Queue API

All vision models use the unified `/infer` endpoint:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vision/infer/submit` | POST | Submit inference task (any model_type) |
| `/vision/infer/status` | GET | Check task status |
| `/vision/infer/get` | GET | Retrieve result |
| `/vision/infer/cancel` | POST | Cancel pending task |
| `/vision/infer/retry` | PUT | Retry failed task |

## See Also

- [Admin Guide](../../doc/admin_guide.md) — deploying vision models
- [Demo 02: SAM3](../02_sam3_segmentation/README.md) — segmentation demo
- [Demo 07: User-Defined Model](../07_user_model/README.md) — custom adapters and pipelines
- [Mini-Rivault README](../../src/umrs/rivault/README.md) — inference stack details
