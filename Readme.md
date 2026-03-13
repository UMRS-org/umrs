# UMRS — Unified Multimodal Retrieval System

Config-driven server and client for multimodal data ingestion into S3-compatible storage with automatic embedding computation and storage in PostgreSQL/pgvector.

## Overview

UMRS watches S3 buckets for new objects (text, images, text files), computes embeddings using configurable models, and stores results in PostgreSQL with the pgvector extension. It supports multiple users, projects ("experiences"), and modalities — all driven by JSON configuration.

### Supported Models

| Model Type | Use Case | Runs On |
|------------|----------|---------|
| `SentenceTransformer` | Text & image embeddings (e.g., `all-mpnet-base-v2`, `clip-ViT-B-32`) | Local |
| `TransformerEncoder` | Text embeddings via HuggingFace `AutoModel` | Local |
| `Resnet-50` | Image embeddings (with optional checkpoint) | Local |
| `MAE` | Raw image embeddings via Masked Autoencoder | Local |
| `openai://` | Text embeddings via any OpenAI-compatible API (e.g., vLLM) | Remote |
| `rivault://` | Two-phase: Slurm launches vLLM+LiteLLM, then uses `openai://` | Remote (HPC) |

### Architecture

```
                      ┌──────────────────────────────────┐
                      │          UMRS Client             │
                      │  upload, ping, status, search... │
                      └──────────┬───────────────────────┘
                                 │ ZeroMQ
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                          UMRS Server                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │ ZMQ Ctrl │  │ S3 Mgr   │  │ DB Mgr   │  │ Model Loader       │  │
│  │ (REP)    │  │ (boto3)  │  │ (psycopg)│  │ ST/OpenAI/Rivault  │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────────────┘  │
│        │             │             │                │              │
│        │             ▼             ▼                ▼              │
│        │      ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
│        │      │  MinIO   │  │ Postgres │  │ vLLM / LiteLLM     │   │
│        │      │  (S3)    │  │ pgvector │  │ (via Rivault/HPC)  │   │
│        │      └──────────┘  └──────────┘  └────────────────────┘   │
└────────┼───────────────────────────────────────────────────────────┘
         │
         │  (Optional: Slurm + Rivault for HPC inference)
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  HPC / Slurm Cluster                                                │
│  sbatch → vLLM + LiteLLM + NGINX → SSH tunnel back to UMRS          │
└─────────────────────────────────────────────────────────────────────┘
```

## Deployment Modes

### 1. Docker Standalone (Quick Start)

Local development with all services in Docker. No GPU or Slurm required.

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker logs -f umrs-server
```

Then try the client:

```bash
docker exec -it umrs-server python src/client_umrs.py ping
docker exec -it umrs-server python src/client_umrs.py --conf /root/conf_umrs.json list_buckets user1
```

See `docker/README.md` for details.

### 2. Slurm Development Environment

Single-workstation setup with a real Slurm cluster (Podman containers for control plane, native slurmd for GPU access). Rivault launches vLLM+LiteLLM as Slurm jobs.

```bash
cd demos/slurm-dev
podman-compose up -d
sudo bash scripts/start-host-slurmd.sh
bash scripts/run_e2e_demo.sh
```

See `demos/slurm-dev/README.md` for the full guide.

### 3. HPC Cluster (R-CCS / Production)

Deploy UMRS on a multi-node HPC cluster with Slurm scheduling and real GPU inference via Rivault.

See `doc/umrs_rivault_rag_plan.md` and `demos/slurm-dev/README.md`.

## Repository Structure

```
src/                     Core application code
├── server_umrs.py       Server entry point
├── client_umrs.py       Client CLI
├── embedding_worker.py  Standalone embedding worker
├── rag_service.py       RAG query service
└── umrs/                Core modules (S3, SQL, models, ZMQ, etc.)

docker/                  Docker standalone deployment
demos/
├── slurm-dev/           Slurm dev environment (Podman + native slurmd)
│   ├── rivault/         Rivault/vLLM inference stack
│   └── scripts/         E2E demo and setup scripts
└── backup/              Archived demo scripts

doc/                     Documentation
tests/                   Test suite
```

## Documentation

| Document | Description |
|----------|-------------|
| `doc/umrs_overview.md` | Full architecture, data flow, and implementation guide |
| `doc/zmq_api_contract.md` | ZeroMQ protocol specification |
| `doc/umrs_rivault_rag_plan.md` | Rivault + RAG integration plan |
| `doc/embedding_worker_deployment.md` | Embedding worker setup |
| `doc/set-up_pgvector.md` | PostgreSQL/pgvector configuration |
| `doc/multi_users.md` | Multi-user configuration guide |
| `doc/HOWTO.md` | Legacy how-to notes |
| `docker/README.md` | Docker deployment guide |
| `demos/slurm-dev/README.md` | Slurm dev environment guide |