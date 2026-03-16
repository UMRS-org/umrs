# Demo 00: Start Services

Admin-facing demo — bring up the full UMRS control-plane stack and verify it.

## Prerequisites

- `podman` and `podman-compose` installed
- `slurmd`, `munged` available on the host
- NVIDIA or ROCm GPU drivers (for compute jobs)
- Complete the initial setup first (see below)

## Quick Start

```bash
# 1. One-time hardware setup (generates .env, slurm.conf, gres.conf)
cd demos/slurm-dev
bash scripts/init-setup.sh --auto

# 2. Start all services
cd ../00_start_services
bash start_all.sh

# 3. Verify
bash health_check.sh
```

## What `start_all.sh` Does

1. Starts the control-plane containers (`ai-login`, `postgres`, `minio`, `munge`)
2. Starts the host `slurmd` daemon (GPU-capable compute node)
3. Waits for Slurm `compute-01` to register
4. Resumes the node if it's in `DRAIN` state from a prior run
5. Runs a health check (ZMQ ping, Slurm status)

## Scripts

| Script | Description |
|--------|-------------|
| `start_all.sh` | Start control plane + host slurmd + verify |
| `stop_all.sh` | Clean shutdown of all services |
| `health_check.sh` | Verify all services are healthy |

## What Gets Started

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| UMRS Server | `ai-login` | 5555 (ZMQ) | Control plane, Slurm controller |
| PostgreSQL | `demo-postgres` | 15432 | pgvector embedding storage |
| MinIO | `demo-minio` | 19000 | S3-compatible object store |
| Munge | `demo-munge` | — | Slurm authentication |
| Host slurmd | (native) | — | GPU compute via Slurm |

## Stopping

```bash
bash stop_all.sh
```

## Next Steps

Once services are running:
- [Demo 01: Text Embedding](../01_text_embedding/README.md) — upload text, get embeddings
- [Demo 02: SAM3 Segmentation](../02_sam3_segmentation/README.md) — image segmentation
- [Demo 03: HMAE Inference](../03_hmae_inference/README.md) — feature extraction

## See Also

- [Admin Guide](../../doc/admin_guide.md) — full deployment documentation
- [demos/slurm-dev/](../slurm-dev/) — shared infrastructure configs
