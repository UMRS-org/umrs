# UMRS Overview

UMRS is the data and orchestration control plane used by R4/RiVault. It owns
durable jobs, endpoint lifecycle, model registry behavior, workflow execution,
RAG execution, and S3/MinIO file handling. Gateway and `r4` are integration
layers around this surface; they should not reimplement UMRS state machines.

## Runtime Surfaces

| Surface | Purpose | Current implementation |
|---|---|---|
| ZMQ control/data plane | Legacy and trusted-operator command channel | `src/server_umrs.py`, `src/umrs/core/MessageQueue.py` |
| FastAPI sidecar | Gateway-facing API for files, jobs, models, RAG, LLM proxy, MCP, workflows, endpoints | `src/umrs/api/` |
| Durable job service | Tracks logical jobs, attempts, retries, and recovery | `src/umrs/jobs/` |
| Orchestration backends | Launch and monitor Mini-RiVault endpoints | `src/umrs/orchestrator/` |
| RAG and pipelines | Retrieval, indexing, workflow execution, and pipeline SDK | `src/umrs/rag/`, `src/umrs/pipeline_sdk/` |

## Current Architecture

UMRS is organized around these current packages:

- `src/server_umrs.py`: native server entrypoint used by the RCCS deployment.
- `src/umrs/api/`: HTTP routers consumed by Gateway and test harnesses.
- `src/umrs/server/`: server-owned registries, workflow handlers, job tracking,
  and runtime helpers.
- `src/umrs/shared/`: auth, isolation, config, storage, modality, and common
  utilities.
- `src/umrs/jobs/`: durable job schema, adapters, model download backend,
  reconciler, scheduler, and monitor support.
- `src/umrs/model_registry/`: model metadata, scope/ACL policies, conversion,
  and storage references.
- `src/umrs/orchestrator/`: endpoint registry, readiness, Slurm/Podman,
  Kubernetes, Compose, site profiles, and launch-script generation.
- `src/umrs/inference/`: model loading and inference adapters.
- `src/umrs/vision_service/`: BentoML vision service and pluggable adapters.
- `src/umrs/pipeline_sdk/`: user-defined workflow and pipeline primitives.
- `src/umrs/rag/`: retrieval, reranking, and context assembly.

## Deployment Modes

| Mode | Use | Entry points |
|---|---|---|
| RCCS Slurm/Podman | Production-like HPC validation | `deploy/rccs-cloud/`, `src/umrs/orchestrator/site_profiles/rccs-cloud.yaml` |
| Docker/Podman Compose | Local development and service tests | `docker/`, parent `docker-compose.e2e.yaml` |
| Kubernetes | Cluster validation and cloud/server deployment | `docker/k8s/`, parent `helm/` and `tests/run_k8s_tests.sh` |

RCCS currently runs infrastructure services such as Postgres and MinIO in
rootless Podman while UMRS runs natively so it can invoke Slurm commands and
manage SSH tunnels.

## Model Downloads and File Handling

Model downloads are initiated through UMRS model APIs and run as durable jobs.
Mini-RiVault inference containers must use pre-staged model files and run in
offline mode.

Important files:

- `src/umrs/api/models.py`: model registry and staging endpoints.
- `src/umrs/jobs/model_download.py`: download backend and status checker.
- `src/umrs/jobs/model_download_state.py`: download task state and recovery.
- `src/umrs/jobs/model_download_sources.py`: source-specific download helpers.
- `src/umrs/api/files.py`: project-scoped file APIs and presigned URL handling.

## Gateway and R4 Boundary

Normal users should enter through `r4` or Gateway routes:

- `/gw/llm/*` for OpenAI-compatible LLM traffic.
- `/gw/rag/*` for RAG uploads, documents, and queries.
- `/gw/mcp/*` for MCP tool access.

UMRS remains the implementation owner behind those routes. Direct UMRS access is
for harnesses, local development, and trusted operators.

## Current Docs

- `Readme.md`: UMRS repository overview and quick start.
- `doc/admin_guide.md`: operator-oriented UMRS administration.
- `doc/user_guide.md`: UMRS CLI/user workflows.
- `doc/access_control.md`: current auth and ownership model.
- `doc/design_user_defined_models.md`: custom workflow/model extension model.
- `doc/rag_implementation.md`: RAG implementation notes.
- `doc/AI4S_PLATFORM_BOUNDARY.md`: boundary with the parent ai4s/R4 platform.
- `deploy/rccs-cloud/README.md`: RCCS deployment and validation workflow.
- `demos/README.md`: canonical public demo suite.
