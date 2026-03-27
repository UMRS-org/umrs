# rccs-cloud Demos

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
| 04 | [04_hmae_inference.sh](04_hmae_inference.sh) | Upload images → HMAE feature extraction on Slurm → download results |
| 05 | [05_user_model.sh](05_user_model.sh) | Run a custom user Step (TextStats) against S3 data via pipeline_sdk |
| 06 | [06_pipeline_sdk.sh](06_pipeline_sdk.sh) | 6-phase Pipeline SDK lifecycle (LOCAL backend, no extra services needed) |
| 07 | [07_sam3_jens.sh](07_sam3_jens.sh) | SAM3 segmentation via Jens's BentoML stack (`VISION_STACK=jens`) |
| 08 | [08_hmae_jens.sh](08_hmae_jens.sh) | HMAE inference via Jens's BentoML stack (`VISION_STACK=jens`) |

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

# Or Jens's vision stack
./07_sam3_jens.sh
./08_hmae_jens.sh

# User-defined model and Pipeline SDK
./05_user_model.sh        # requires embeddings from 01
./06_pipeline_sdk.sh
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
| `UMRS_IMAGE_BUCKET` | `user1-images` | 03, 04, 07, 08 |
| `UMRS_RAG_URL` | `http://localhost:8081` | 02 |
| `VISION_STACK` | `jens` | 07, 08 |

Example:
```bash
UMRS_IMAGE_BUCKET=my-custom-bucket ./03_sam3_segmentation.sh /path/to/images
```

## Sample data

- `sample_data/*.txt` — text files for demos 01, 02, 05, 06
- `sample_data/images/` — place your `.jpg/.png/.tiff` files here for vision demos

