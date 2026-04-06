# rccs-cloud Demos

> **DEPRECATED** — These deployment-specific demo wrappers are superseded by the
> canonical demos at **[demos/](../../../demos/)**. The canonical demos work with
> any deployment (Docker Compose, rccs-cloud, hpbd-atom, etc.) as long as the
> server is reachable. See **[demos/README.md](../../../demos/README.md)** and
> **[doc/demo_guide.md](../../../doc/demo_guide.md)** for the full walkthrough.
>
> These scripts remain for historical reference but will not receive updates.

Lightweight demos for the rccs-cloud deployment.
These run **on your local Mac** through SSH tunnels — no `podman exec` needed.

## Prerequisites

1. SSH tunnels are running (`./ssh_tunnel.sh` from `deploy/rccs-cloud/`).
2. Client is set up (`./setup_client.sh` was run successfully).
3. The UMRS server is running on rccs-cloud.

## Demos

| # | Script | What it does |
|---|--------|--------------|
| 00 | [00_health_check.sh](00_health_check.sh) | Ping the server, check MinIO and RAG reachability |
| 01 | [01_text_embedding.sh](01_text_embedding.sh) | Upload text files → compute embeddings → check status |
| 02 | [02_rag_search.sh](02_rag_search.sh) | Semantic search: vector, hybrid, and Q&A queries |
| 03 | [03_sam3_segmentation.sh](03_sam3_segmentation.sh) | Upload images → SAM3 segmentation on Slurm → download results |
| 04 | [04_hmae_inference.sh](04_hmae_inference.sh) | (Optional) upload HMAE model → upload images → HMAE on Slurm → download results |
| 05 | [05_user_model.sh](05_user_model.sh) | Run a custom user Step (TextStats) against S3 data via pipeline_sdk |
| 06 | [06_pipeline_sdk.sh](06_pipeline_sdk.sh) | 6-phase Pipeline SDK lifecycle (LOCAL backend, no extra services needed) |
| 09 | [09_model_registry.sh](09_model_registry.sh) | Model registry: list, register, upload (with optional real HMAE model) |

## Quick start

```bash
cd deploy/rccs-cloud

# Terminal 1: start tunnels
./ssh_tunnel.sh

# Terminal 2: run demos
cd demos
./00_health_check.sh
./01_text_embedding.sh
./02_rag_search.sh

# Vision demos (need images in sample_data/images/)
./03_sam3_segmentation.sh
./04_hmae_inference.sh

# HMAE with model upload (optional — uploads checkpoint before inference)
HMAE_LOCAL_MODEL_DIR=~/Downloads/hmae_s8d-1k_pretrain_denorm-seq1k-pz32-bs16-lr2e-4-400ep-rope-cp \
  ./04_hmae_inference.sh

# User-defined model and Pipeline SDK
./05_user_model.sh        # requires embeddings from 01
./06_pipeline_sdk.sh

# Model registry (with optional real HMAE upload)
HMAE_LOCAL_MODEL_DIR=~/Downloads/hmae_s8d-1k_pretrain_denorm-seq1k-pz32-bs16-lr2e-4-400ep-rope-cp \
  ./09_model_registry.sh
```

## Configuration

All demos use the user config installed by `setup_client.sh`:

```
~/.config/umrs/users/user1.json   # user config (-c flag)
~/.config/umrs/s3cfg              # S3 credentials
```

Override with environment variables:

| Variable | Default | Used by |
|----------|---------|---------|
| `UMRS_USER_CONF` | `~/.config/umrs/users/user1.json` | all demos |
| `UMRS_TEXT_BUCKET` | `user1-texts` | 01, 02, 05 |
| `UMRS_IMAGE_BUCKET` | `user1-images` | 03, 04 |
| `UMRS_RAG_URL` | `http://localhost:8081` | 02 |
| `HMAE_LOCAL_MODEL_DIR` | *(unset)* | 04, 09 — path to HMAE checkpoint dir |
| `UMRS_REMOTE_HMAE_MODEL_PATH` | `/lvs0/.../umrs_models/hmae` | 04, 09 — server-side path for model registration |

Example:
```bash
UMRS_IMAGE_BUCKET=my-custom-bucket ./03_sam3_segmentation.sh /path/to/images
```

## Sample data

- `sample_data/*.txt` — text files for demos 01, 02, 05, 06
- `sample_data/images/` — place your `.jpg/.png/.tiff` files here for vision demos

