# Demo 02: SAM3 Image Segmentation

User-facing demo — upload images, run SAM3 text-prompted segmentation via
Mini-Rivault's BentoML vision service, and retrieve results.

## Prerequisites

- Services running (see [Demo 00](../00_start_services/README.md))
- **SAM3 model weights pre-staged** by admin at `$SAM3_MODEL_PATH`
  (default: `./rivault/models/sam3`)
- UMRS client installed (`pip install .` from `src/`)

## Architecture

```
Client → UMRS Server → Slurm sbatch → Mini-Rivault
              │                         ├── NGINX (:4000, published)
              │                         ├── vLLM (embeddings)
              │                         └── BentoML-vision (:3000, internal)
              │                              └── SAM3 model (GPU)
              └──────── SSH tunnel ─────┘
                        (single port, /vision/ prefix for BentoML)
```

All vision inference runs inside Mini-Rivault for security isolation:
- Per-job ephemeral containers
- Private network (no internet)
- Read-only model weights
- Tmpfs scratch (wiped on job end)

## Quick Start

```bash
cd demos/02_sam3_segmentation
bash run_demo.sh
```

## What This Demo Does

1. Verifies services are running
2. Activates Mini-Rivault `vision-bentoml` profile
3. Uploads test images to S3
4. Submits SAM3 segmentation tasks via BentoML async task queue
5. Polls task status until complete
6. Downloads segmentation results (masks, bounding boxes, scores)

## Running Manually

```bash
# 1. Ensure vision profile is active
export RIVAULT_PROFILES="harness vllm vision-bentoml"

# 2. Upload images
umrs-client -c conf_umrs.json upload_dir ./test_data/ alice-images

# 3. Use the demo Python script directly
python3 ../../demos/slurm-dev/scripts/demo_sam3_segmentation.py \
    --bentoml-url http://localhost:$LOCAL_PORT/vision \
    --s3-endpoint http://localhost:19000 \
    --image-path ./test_data/sample.jpg \
    --prompt "crack"
```

## Configuration

### Mini-Rivault Profile

The vision profile must be enabled for the Slurm job. Set the environment
variable before launching:

```bash
export RIVAULT_PROFILES="harness vllm vision-bentoml"
```

This tells `serve-model.slurm` to include the `bentoml-vision` service in the
Mini-Rivault stack. Traffic reaches BentoML through NGINX's `/vision/` prefix.

### Vision Model Config

The model is registered in `vision_models.json`:

```json
{
  "modality_name": "segmentation",
  "model": {
    "model_name": "facebook/sam3",
    "vector_size": "256",
    "model_type": "bentoml://http://localhost:{LOCAL_PORT}/vision/submit",
    "return_type": "file"
  }
}
```

### Environment

Environment overrides in `env.vision`:

```bash
SAM3_MODEL_PATH=./rivault/models/sam3
RIVAULT_PROFILES="harness vllm vision-bentoml"
```

## Adapter Architecture

The SAM3 model is loaded as a **pluggable adapter** — a Python file
(`sam3_adapter.py`) that implements the `VisionModelAdapter` interface.

At container startup, the BentoML vision service scans the directories listed
in `VISION_ADAPTERS_DIR` (colon-separated, e.g. `/adapters/system:/adapters/user`)
and auto-registers every `VisionModelAdapter` subclass it finds.

```
/adapters/system/sam3_adapter.py   → model_type="sam3_adapter"
/adapters/system/hmae_adapter.py   → model_type="hmae_adapter"
/adapters/user/my_model.py         → model_type="my_model"  (user-provided)
```

Built-in adapters (SAM3, HMAE) ship as `.py` files in `rivault/default_adapters/`
and are loaded via the same mechanism. Users can add their own adapters by placing
`.py` files in `USER_ADAPTERS_DIR` and optionally including a `requirements.txt`
for extra dependencies.

See [Demo 07: User-Defined Model](../07_user_model/README.md) for details on
writing your own adapter.

## BentoML Task Queue API

All vision models use the unified `/infer` endpoint:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vision/infer/submit` | POST | Submit inference task (any model_type) |
| `/vision/infer/status` | GET | Check task status |
| `/vision/infer/get` | GET | Retrieve result |
| `/vision/infer/cancel` | POST | Cancel pending task |
| `/vision/infer/retry` | PUT | Retry failed task |

All endpoints are accessed through NGINX's `/vision/` prefix via the SSH tunnel.

## Security

- Model runs inside Mini-Rivault sandbox (no direct GPU access from client)
- User-uploaded images are processed in ephemeral containers
- No cross-user data leakage (per-job isolation)
- Network-isolated: `TRANSFORMERS_OFFLINE=1`, no DNS, private subnet

## See Also

- [Admin Guide](../../doc/admin_guide.md) — deploying vision models
- [Demo 03: HMAE](../03_hmae_inference/README.md) — feature extraction demo
- [Demo 07: User-Defined Model](../07_user_model/README.md) — custom adapters and pipelines
- [Mini-Rivault README](../../src/umrs/rivault/README.md) — inference stack details
