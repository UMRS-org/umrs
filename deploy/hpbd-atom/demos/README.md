# hpbd-atom Demos

Lightweight demos for the hpbd-atom deployment.
These run **on your local Mac** through SSH tunnels — no `podman exec` needed.

## Prerequisites

1. SSH tunnels are running (`./ssh_tunnel.sh` from `deploy/hpbd-atom/`).
2. Client is set up (`./setup_client.sh` was run successfully).
3. The UMRS server is running on hpbd-atom.

## Demos

| # | Script | What it does | Requirements |
|---|--------|--------------|-------------|
| 00 | [00_health_check.sh](00_health_check.sh) | Ping the server, check MinIO and RAG reachability | SSH tunnels |
| 01 | [01_text_embedding.sh](01_text_embedding.sh) | Upload text files → compute embeddings → check status | SSH tunnels |
| 02 | [02_rag_search.sh](02_rag_search.sh) | Semantic search: vector, hybrid, and Q&A queries | SSH tunnels, RAG service (`UMRS_RAG_ENABLE=1`) |
| 03 | [03_sam3_segmentation.sh](03_sam3_segmentation.sh) | Upload images → SAM3 segmentation on Slurm → download results | **Slurm + GPU + shared FS** |
| 04 | [04_hmae_inference.sh](04_hmae_inference.sh) | (Optional) upload HMAE model → upload images → HMAE on Slurm → download results | **Slurm + GPU + shared FS** |
| 05 | [05_user_model.sh](05_user_model.sh) | Run a custom user Step (TextStats) against S3 data via pipeline_sdk | SSH tunnels, demo 01 first |
| 06 | [06_pipeline_sdk.sh](06_pipeline_sdk.sh) | 6-phase Pipeline SDK lifecycle (LOCAL backend, no extra services needed) | None (runs locally) |
| 07 | [07_sam3_jens.sh](07_sam3_jens.sh) | SAM3 with Jens's standalone BentoML stack | **Slurm + GPU + Jens service** |
| 08 | [08_hmae_jens.sh](08_hmae_jens.sh) | HMAE with Jens's standalone BentoML stack | **Slurm + GPU + Jens service** |
| 09 | [09_model_registry.sh](09_model_registry.sh) | Model registry: list, register, upload (with optional real HMAE model) | SSH tunnels |

**Note:** Demos 03, 04, 07, 08 require a Slurm cluster with shared filesystem
between login and compute nodes, plus GPU-capable compute nodes. They are
skipped automatically by `run_all_core.sh` when Slurm is not available.

## Quick start

```bash
cd deploy/hpbd-atom

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
| `UMRS_REMOTE_HMAE_MODEL_PATH` | `/home/mnagaso/workspace/umrs_models/hmae` | 04, 09 — server-side path for model registration |

Example:
```bash
UMRS_IMAGE_BUCKET=my-custom-bucket ./03_sam3_segmentation.sh /path/to/images
```

## Sample data

- `sample_data/*.txt` — text files for demos 01, 02, 05, 06 (3 HPC/ML-themed articles included)
- `sample_data/images/` — `.jpg/.png/.tiff` images for vision demos (includes `test_pattern.png`)
