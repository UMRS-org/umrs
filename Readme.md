# UMRS — Unified Multimodal Retrieval System

Config-driven server and client for multimodal data ingestion into S3-compatible storage with automatic embedding computation and storage in PostgreSQL/pgvector. Supports text, image, and vision model pipelines — including SAM3 segmentation and HMAE reconstruction via BentoML inside sandboxed Mini-Rivault containers.

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
| `bentoml://` | Vision tasks (SAM3, HMAE) via BentoML inside Mini-Rivault | Remote (HPC) |

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
         │  (Slurm + Mini-Rivault for HPC inference)
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  HPC / Slurm Cluster                                                │
│  sbatch → Mini-Rivault (podman-compose)                              │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │  NGINX (:4000)                                               │     │
│  │   /        → LiteLLM → vLLM   (text embeddings)             │     │
│  │   /vision/ → BentoML-vision    (SAM3, HMAE)                 │     │
│  └─────────────────────────────────────────────────────────────┘     │
│  SSH tunnel back to UMRS server                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start (Client)

Install the UMRS client as a pip package (no GPU or server dependencies needed):

```bash
cd src
pip install -e .
umrs-client --help
umrs-client ping
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

See [docker/README.md](docker/README.md) for details.

### 2. Slurm Development Environment

Single-workstation setup with a real Slurm cluster (Podman containers for control plane, native slurmd for GPU access). Mini-Rivault launches vLLM+LiteLLM and BentoML-vision as Slurm jobs.

```bash
cd demos/slurm-dev
podman-compose up -d
sudo bash scripts/start-host-slurmd.sh
bash scripts/run_e2e_demo.sh
```

See [demos/slurm-dev/README.md](demos/slurm-dev/README.md) for the full guide.

### 3. HPC Cluster (R-CCS / Production)

Deploy UMRS on a multi-node HPC cluster with Slurm scheduling and real GPU inference via Mini-Rivault.

See [doc/admin_guide.md](doc/admin_guide.md) and [doc/umrs_rivault_rag_plan.md](doc/umrs_rivault_rag_plan.md).

## Demos

Step-by-step demos that walk through UMRS workflows end to end:

| Demo | Description |
|------|-------------|
| [demos/00_start_services/](demos/00_start_services/) | Start/stop all UMRS services + health checks (admin) |
| [demos/01_text_embedding/](demos/01_text_embedding/) | Upload text, compute embeddings, export results |
| [demos/02_sam3_segmentation/](demos/02_sam3_segmentation/) | SAM3 image segmentation via BentoML inside Mini-Rivault |
| [demos/03_hmae_inference/](demos/03_hmae_inference/) | HMAE feature extraction / reconstruction via Mini-Rivault |
| [demos/04_rag_search/](demos/04_rag_search/) | RAG semantic search: vector, hybrid, reranking, Q&A |
| [demos/05_server_maintenance/](demos/05_server_maintenance/) | Data integrity: list_buckets, check, update, recompute |
| [demos/06_model_registry/](demos/06_model_registry/) | Model register, list, download (HuggingFace + local) |
| [demos/07_user_model/](demos/07_user_model/) | User-defined model workflows (Step) |
| [demos/08_defect_analysis/](demos/08_defect_analysis/) | Multi-step pipeline: SAM3 → crop → HMAE → cluster |
| [demos/09_hmae_full/](demos/09_hmae_full/) | Full HMAE pipeline (reconstruct, MAE, features) |
| [demos/10_job_submission/](demos/10_job_submission/) | Job submission, service lifecycle, --keep-alive |
| [demos/11_pipeline_sdk/](demos/11_pipeline_sdk/) | Pipeline SDK: 6-phase lifecycle |
| [demos/12_plugins/](demos/12_plugins/) | Plugin system: sandboxed execution (Slurm/K8s) |

See [demos/README.md](demos/README.md) for the full guide.

## Repository Structure

```
src/                        Core application code
├── pyproject.toml           Pip-installable client package
├── server_umrs.py           Server entry point
├── client_umrs.py           Client CLI (also installable as `umrs-client`)
├── embedding_worker.py      Standalone embedding worker
├── rag_service.py           RAG query service
└── umrs/                    Core modules
    ├── models/              Model implementations (HMAE, SAM3, MAE, etc.)
    ├── vision_service/      BentoML vision service (SAM3 + HMAE endpoints)
    ├── rivault/             Mini-Rivault compose stack + NGINX config
    ├── rivault_launcher.py  Rivault job launcher (Slurm + SSH tunnels)
    ├── pipeline_sdk/        Optional in-container pipeline library
    └── ...                  S3, SQL, ZMQ, auth, config, etc.

demos/                       Step-by-step demos (basic → advanced)
├── 00_start_services/       Admin: start/stop/health-check
├── 01_text_embedding/       Basic: text embedding workflow
├── 02_sam3_segmentation/    Basic: SAM3 vision pipeline
├── 03_hmae_inference/       Basic: HMAE vision pipeline
├── 04_rag_search/           Basic: RAG semantic search + Q&A
├── 05_server_maintenance/   Ops: data integrity (check, update, recompute)
├── 06_model_registry/       Ops: model management
├── 07_user_model/           Advanced: user-defined workflows
├── 08_defect_analysis/      Advanced: multi-step pipeline
├── 09_hmae_full/            Advanced: full HMAE pipeline
├── 10_job_submission/       Infra: job submission + lifecycle
├── 11_pipeline_sdk/         Infra: pipeline SDK
├── 12_plugins/              Infra: plugin system
└── slurm-dev/               Slurm dev environment (Podman + native slurmd)

docker/                      Docker standalone deployment
doc/                         Documentation
tests/                       Test suite
```

## Documentation

| Document | Description |
|----------|-------------|
| [doc/admin_guide.md](doc/admin_guide.md) | Admin deployment & operations guide |
| [doc/user_guide.md](doc/user_guide.md) | User getting started guide |
| [doc/umrs_overview.md](doc/umrs_overview.md) | Full architecture, data flow, and implementation guide |
| [doc/zmq_api_contract.md](doc/zmq_api_contract.md) | ZeroMQ protocol specification |
| [doc/umrs_rivault_rag_plan.md](doc/umrs_rivault_rag_plan.md) | Rivault + RAG integration plan |
| [doc/rag_implementation.md](doc/rag_implementation.md) | RAG implementation details |
| [doc/embedding_worker_deployment.md](doc/embedding_worker_deployment.md) | Embedding worker setup |
| [doc/set-up_pgvector.md](doc/set-up_pgvector.md) | PostgreSQL/pgvector configuration |
| [doc/multi_users.md](doc/multi_users.md) | Multi-user configuration guide |
| [docker/README.md](docker/README.md) | Docker deployment guide |
| [demos/slurm-dev/README.md](demos/slurm-dev/README.md) | Slurm dev environment guide |