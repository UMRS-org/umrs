# Demo 03: SAM3 Image Segmentation

User-facing demo — upload images, run SAM3 text-prompted segmentation via
Mini-Rivault's BentoML vision service, and retrieve results.

## Prerequisites

- UMRS server running and reachable
- **SAM3 model weights pre-staged** by the site/operator at `$SAM3_MODEL_PATH`
  (default: `./rivault/models/sam3`)
- UMRS client installed (see [demos/README.md](../README.md#prerequisites) — `uv pip install -e src/`)

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
# Navigate from any level (auto-detects parent repo vs umrs subdirectory)
cd "$(git rev-parse --show-toplevel)/$(test -d "$(git rev-parse --show-toplevel)/umrs/demos" && echo umrs)/demos/03_sam3_segmentation"
# See the numbered steps below in this README
```

## What This Demo Does

1. Verifies services are running
2. Activates Mini-Rivault `vision-bentoml` profile
3. Uploads test images to S3
4. Submits SAM3 segmentation tasks via BentoML async task queue
5. Polls task status until complete
6. Downloads segmentation results (masks, bounding boxes, scores)

The `*_sam3_results.pkl` artifact is intentionally compact by default on shared
deployments. It contains boxes, scores, object count, and mask shapes. Set the
server-side environment variable `SAM3_INCLUDE_MASKS_IN_RESULTS=1` only when you
explicitly need full-resolution mask arrays.

## Running Manually

```bash
# 1. Load the experience fragment into your user config
#    This registers the SAM3 segmentation experience (model + bucket + table)
umrs experience add experience.json

# 2. Reload to apply the new experience and provision buckets
umrs reload

# 3. Ensure the server deployment already includes the vision profile
#    On rccs-cloud this is configured in site.yaml, so no client-side export is needed.

# 4. Verify that the SAM3 model is already visible to the server
umrs model list --scope system
umrs model list --scope user

# Optional: only if your site exposes a writable, allowed staging root
umrs model download sam3.1 --name sam3.1 \
  --scope user --server-side --staging-dir /path/to/site-user-model-root

# 5. Upload images
umrs data dir-upload ./test_data_full user1-sam3

# 6. Download results once processing completes
umrs data download user1-sam3 ./results --force-overwrite
```

## Interactive Gradio UI (optional)

For exploratory, one-image-at-a-time segmentation you can drive the same
running BentoML vision endpoint from a local Gradio web UI instead of the
batch `dir-upload` path. This uses the `r4` client (Gateway-mediated), so it
requires an authenticated `r4` profile in addition to the `umrs` client.

```bash
# Gradio is an optional dependency of the r4 client
pip install gradio

# Launch the local UI (blocking). Point it at the SAM3 adapter.
# --endpoint-id pins a specific warm endpoint (see note below); omit it to
# let the Gateway pick the model by name.
r4 vision ui --adapter sam3 --endpoint-id <endpoint-job-id>
```

Then open http://127.0.0.1:7860, drop an image, and type a text prompt
(e.g. `defect`) for promptable segmentation. Results are written to project
S3 storage and the `s3://` object keys are shown in the UI.

> **Keep the endpoint warm.** Interactive use benefits from a persistent
> backend. Set `"keep_endpoint": true` on the modality in `experience.json`
> so the mini-rivault/BentoML endpoint is not auto-released between batches
> (the idle janitor still reaps it once genuinely idle). Otherwise the first
> UI request pays the endpoint cold-start cost.

## Configuration

### Mini-Rivault Profile

The vision profile must be enabled in the server-side RiVault launch context.
For a local single-machine deployment you can export it before starting the
server:

```bash
export RIVAULT_PROFILES="harness vllm vision-bentoml"
```

This tells `serve-model.slurm` to include the `bentoml-vision` service in the
Mini-Rivault stack. Traffic reaches BentoML through NGINX's `/vision/` prefix.
On `rccs-cloud`, the profile is already set in the site config used by the
native server.

### Vision Model Config

The model is registered in `vision_models.json`:

```json
{
  "modality_name": "segmentation",
  "model": {
    "model_name": "sam3.1",
    "vector_size": "256",
    "model_type": "rivault://bentoml://http://localhost:{LOCAL_PORT}/vision/infer/submit",
    "vision_model_type": "sam3",
    "return_type": "file"
  }
}
```

### Environment

Environment overrides in `env.vision`:

```bash
SAM3_MODEL_PATH=./rivault/models/sam3.1
RIVAULT_PROFILES="harness vllm vision-bentoml"
```

On shared deployments, `sam3.1` is usually pre-staged by the site/operator.
Normal users should not expect to create a new `system`-scoped SAM3 model from
the demo shell.

## Adapter Architecture

The SAM3 model is loaded as a **pluggable adapter** — a Python file
(`sam3_adapter.py`) that implements the `VisionModelAdapter` interface.

At container startup, the BentoML vision service scans the directories listed
in `VISION_ADAPTERS_DIR` (colon-separated, e.g. `/adapters/system:/adapters/user`)
and auto-registers every `VisionModelAdapter` subclass it finds.

```
/adapters/system/sam3_adapter.py   → model_type="sam3"
/adapters/system/hmae_adapter.py   → model_type="hmae"
/adapters/user/my_model.py         → model_type="my_model"  (user-provided)
```

Built-in adapters ship from `src/umrs/vision_service/default_adapters/` and are
mounted into the BentoML container under `/adapters/system`. Users can add their
own adapters by placing `.py` files in an adapter directory and optionally adding
a `requirements.txt` file for extra dependencies.

See [Demo 08: User-Defined Model](../08_user_model/README.md) for details on
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
- [Demo 04: Vision Inference](../04_vision_inference/README.md) — generic BentoML vision inference
- [Demo 08: User-Defined Model](../08_user_model/README.md) — custom adapters and pipelines
- [Mini-Rivault README](../../src/umrs/mini-rivault-llm/README.md) — inference stack details
