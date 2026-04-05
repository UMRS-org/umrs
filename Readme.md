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

### 2. HPC Cluster (R-CCS / Production)

Deploy UMRS on a multi-node HPC cluster with Slurm scheduling and real GPU inference via Mini-Rivault.

See [doc/admin_guide.md](doc/admin_guide.md) and [doc/umrs_rivault_rag_plan.md](doc/umrs_rivault_rag_plan.md).

## Demos

Step-by-step demos that walk through UMRS workflows end to end:

### Core Demos

| Demo | Description |
|------|-------------|
| [demos/00_start_services/](demos/00_start_services/) | Start all UMRS services (control plane + host slurmd) |
| [demos/01_health_check/](demos/01_health_check/) | Verify UMRS server connectivity and user configuration |
| [demos/02_text_embedding/](demos/02_text_embedding/) | Upload text → server-side embedding → export vectors |
| [demos/03_sam3_segmentation/](demos/03_sam3_segmentation/) | SAM3 image segmentation via BentoML on Slurm |
| [demos/04_hmae_inference/](demos/04_hmae_inference/) | HMAE reconstruction / feature extraction via Slurm |
| [demos/05_rag_search/](demos/05_rag_search/) | RAG semantic search: vector, hybrid, reranking, Q&A |
| [demos/06_server_maintenance/](demos/06_server_maintenance/) | Maintenance: list_buckets, check, update, recompute |
| [demos/07_model_registry/](demos/07_model_registry/) | Model registry: register, upload, verify, filter |
| [demos/08_user_model/](demos/08_user_model/) | User-defined model workflows (Step) |
| [demos/09_defect_analysis/](demos/09_defect_analysis/) | Defect-oriented SAM3 segmentation pipeline |
| [demos/10_hmae_full_pipeline/](demos/10_hmae_full_pipeline/) | Full HMAE pipeline (reconstruct + MAE + features) |
| [demos/11_job_submission/](demos/11_job_submission/) | Job submission & service lifecycle, --keep-alive |
| [demos/12_pipeline_sdk/](demos/12_pipeline_sdk/) | Pipeline SDK: 6-phase lifecycle (LOCAL backend) |
| [demos/13_plugins/](demos/13_plugins/) | Plugin system: spec, validate, execute, audit |
| [demos/14_opossum_analysis/](demos/14_opossum_analysis/) | Video motion analysis via Pipeline SDK |

### Advanced Demos

| Demo | Description |
|------|-------------|
| [demos/15_complex_dag/](demos/15_complex_dag/) | Multi-step DAG with parallel branches and fan-in |
| [demos/16_custom_workflow/](demos/16_custom_workflow/) | Upload & execute custom workflows on the server |
| [demos/17_multi_user/](demos/17_multi_user/) | User isolation: separate S3 buckets and databases |
| [demos/18_model_lifecycle/](demos/18_model_lifecycle/) | Model lifecycle: register → use → deprecate |
| [demos/19_rag_advanced/](demos/19_rag_advanced/) | Compare RAG modes: vector, BM25, hybrid, reranked |
| [demos/20_durable_jobs/](demos/20_durable_jobs/) | Durable jobs: submit, poll, dedup, cancel |
| [demos/21_remote_plugins/](demos/21_remote_plugins/) | Plugin system with multiple source types & backends |

See [demos/README.md](demos/README.md) for the full guide.

## Client Library Features

The `umrs` package (`src/umrs/`) includes reusable library modules for building
workflows on top of the UMRS platform. All modules are stdlib-only and thread-safe.

| Module | Description |
|--------|-------------|
| `progress.py` | Streaming progress tracker with callbacks and console bar |
| `batch_upload.py` | Resume-capable batch uploads with manifest tracking |
| `result_diff.py` | Compare results across runs (numeric, text, image diffs) |
| `webhooks.py` | Event subscription & dispatch with HMAC-signed payloads |
| `resource_tracker.py` | CPU/memory snapshot tracking, quota policies, usage reports |
| `data_versioning.py` | Immutable dataset versions with tagging and lineage |
| `pipeline_templates.py` | Parameterized workflow skeletons with built-in templates |
| `pipeline_builder.py` | Fluent API for composing multi-step pipelines |
| `experiment_io.py` | Export/import experiments as portable ZIP archives |
| `scheduler.py` | Cron/interval/one-shot job scheduler with JSON persistence |

## Tests

Unit tests for all library modules (170 tests):

```bash
cd src
pip install -e .
cd ..
python -m pytest tests/ -v
```

## Repository Structure

```
src/                        Core application code
├── pyproject.toml           Pip-installable client package
├── server_umrs.py           Server entry point
├── client_umrs.py           Client CLI (also installable as `umrs-client`)
└── umrs/                    Core modules
    ├── api/                 FastAPI services including the RAG query service
    ├── models/              Model implementations (HMAE, SAM3, MAE, etc.)
    ├── vision_service/      BentoML vision service (SAM3 + HMAE endpoints)
    ├── rivault/             Mini-Rivault compose stack + NGINX config
    ├── rivault_launcher.py  Rivault job launcher (Slurm + SSH tunnels)
    ├── pipeline_sdk/        Optional in-container pipeline library
    ├── progress.py          Streaming progress tracker
    ├── batch_upload.py      Resume-capable batch uploads
    ├── result_diff.py       Result comparison & diff
    ├── webhooks.py          Event subscription & dispatch
    ├── resource_tracker.py  CPU/memory tracking & quotas
    ├── data_versioning.py   Immutable dataset versions
    ├── pipeline_templates.py Parameterized workflow templates
    ├── pipeline_builder.py  Fluent pipeline composition API
    ├── experiment_io.py     Experiment export/import (ZIP)
    ├── scheduler.py         Cron/interval job scheduler
    └── ...                  S3, SQL, ZMQ, auth, config, etc.

demos/                       Step-by-step demos (basic → advanced)
├── 00_start_services/       Admin: start/stop all services
├── 01_health_check/         Health: connectivity & config check
├── 02_text_embedding/       Basic: text embedding workflow
├── 03_sam3_segmentation/    Basic: SAM3 vision pipeline
├── 04_hmae_inference/       Basic: HMAE vision pipeline
├── 05_rag_search/           Basic: RAG semantic search + Q&A
├── 06_server_maintenance/   Ops: data integrity (check, update, recompute)
├── 07_model_registry/       Ops: model management
├── 08_user_model/           Advanced: user-defined workflows
├── 09_defect_analysis/      Advanced: defect-oriented segmentation
├── 10_hmae_full_pipeline/   Advanced: full HMAE pipeline
├── 11_job_submission/       Infra: job submission + lifecycle
├── 12_pipeline_sdk/         Infra: pipeline SDK
├── 13_plugins/              Infra: plugin system
├── 14_opossum_analysis/     Advanced: video motion analysis
├── 15_complex_dag/          Advanced: DAG with parallel branches
├── 16_custom_workflow/      Advanced: custom workflow execution
├── 17_multi_user/           Advanced: multi-user isolation
├── 18_model_lifecycle/      Advanced: model lifecycle management
├── 19_rag_advanced/         Advanced: RAG mode comparison
├── 20_durable_jobs/         Advanced: durable job patterns
└── 21_remote_plugins/       Advanced: plugin source types

docker/                      Docker standalone deployment
doc/                         Documentation
tests/                       Unit tests (170 tests)
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
| [doc/set-up_pgvector.md](doc/set-up_pgvector.md) | PostgreSQL/pgvector configuration |
| [doc/multi_users.md](doc/multi_users.md) | Multi-user configuration guide |
| [doc/design_user_defined_models.md](doc/design_user_defined_models.md) | User-defined model workflow design |
| [docker/README.md](docker/README.md) | Docker deployment guide |
| [demos/README.md](demos/README.md) | Demo guide (all 22 demos) |