# UMRS Overview

**Updated 2026-09-22.** UMRS is the infrastructure backbone of the R4 + UMRS
platform ([../../docs/R4-UMRS-DESIGN.md](../../docs/R4-UMRS-DESIGN.md)): the
one server on the cluster front-end that owns accounts, the client transport,
durable jobs and scheduling, model self-deployment, storage and the
registries the Scientific Catalog is built from. R4 is a client of this
surface; the NGINX + auth-sidecar edge is today's HTTP entry in front of it.
Gaps against the design are numbered in
[../../docs/R4-UMRS-GAP-ANALYSIS-2026-09-22.md](../../docs/R4-UMRS-GAP-ANALYSIS-2026-09-22.md).

## Runtime surfaces

| Surface | Purpose | Implementation |
|---|---|---|
| ZeroMQ control + data plane | authenticated (CURVE/ZAP) command channel and binary file transfer; direct TCP from clients | `src/server_umrs.py`, `src/umrs/core/MessageQueue.py`, `core/zmq_security.py`, `core/session_curve_provider.py` |
| HTTP sidecar (FastAPI) | the 26 lane-classified API families consumed through the edge: llm, rag, mcp, knowledge (+ graphs, proposals, query, upload, lineage, v2), catalog, companion (+ sessions, venue, approvals), jobs, workflows, models, endpoints, files, projects, grants, skills, vision, a2l, audit, ops, zmq sessions, … | `src/umrs/server/http_sidecar.py` (explicit router registry), `src/umrs/api/` |
| Durable job service | Job/Attempt semantics, scheduler, reconciler, per-kind backends | `src/umrs/jobs/` |
| Orchestration backends | launch and monitor mini-RiVault endpoints and jobs | `src/umrs/orchestrator/` (Slurm+Podman preferred, Slurm+Singularity legacy, Kubernetes, Compose, hybrid) |
| Venues + inference proxy | remote agent sessions in rootless Podman on the front-end; credential injection | `src/umrs/venues/`, `src/umrs/infer_proxy/` |
| Knowledge spine | ingest → understand → embed → verify; graph registry, ontology, proposals; loop engine; bundle export/import | `src/umrs/knowledge/`, `knowledge_graph/`, `knowledge_model/`, `scientific_memory/`, `embeddings_api/`, `search_api/`, `evidence/`, `experiment_registry/` |
| Data catalog + lineage | `catalog_objects`, asset index, A2L | `src/umrs/data_catalog/`, `asset_index/`, `a2l/`, `artifact_registry/` |
| Registries | models, skills, MCP servers | `src/umrs/model_registry/`, `skills/`, `mcp/` |
| Policy | users, projects, quotas, trust, grants, egress | `src/umrs/admin/`, `api/projects.py`, `api/agent_quota.py`, `companion_trust/`, `grants/`, `security/` |
| Operator surfaces | CLI, admin CLI, job monitors | `src/client_umrs.py`, `src/umrs/cli/`, `tui/`, `tui_v2/` |

## Boot sequence (today)

1. `server_umrs.py` reads `site.yaml` + `.env`, performs initial config
   (buckets, user config map), initialises the session-CURVE provider when
   `UMRS_ZMQ_AUTH_TYPE=session`, wires the reload context.
2. ZeroMQ control socket binds (CURVE applied before bind in `curve` /
   `session` mode; fail-closed without server keys); per-user data sockets
   are allocated on `CREATE_DATA_QUEUE`.
3. The HTTP sidecar starts (default port 8082; loopback in tunnel mode) and
   mounts the router registry; the durable job scheduler and reconciler
   start (in-flight attempts marked LOST and requeued).
4. Orchestrator backends are selected by `execution_authority` / site
   profile; endpoint registry, janitor and readiness probes run.

Details: [../../docs/architecture/runtime/RUNTIME_TOPOLOGY_REFERENCE.md](../../docs/architecture/runtime/RUNTIME_TOPOLOGY_REFERENCE.md)
(pre-design detail, verify before citing).

## Deployment modes

| Mode | Use | Entry points |
|---|---|---|
| umrs-front (data0) | production front-end; one stack per cluster (rccs, ai4s; bdrt, octopus) | parent `deploy/ansible/bootstrap.sh --profile umrs-front`, `deploy/umrs-front/` |
| rccs-cloud / login-direct | UMRS on a Slurm login node | `deploy/rccs/` (profile name `rccs-cloud`), `orchestrator/site_profiles/rccs-cloud.yaml` |
| Docker / Podman Compose | local development and service tests | `docker/`, parent `docker-compose.e2e.yaml` |
| Kubernetes | cluster validation and cloud/server deployment | `docker/k8s/`, parent `helm/`, `tests/run_k8s_tests.sh` |

Infrastructure services (Postgres, object store, edge) run in rootless
Podman on the front-end; the UMRS API containers hold the podman socket so
they can launch frontend-venue sandboxes; on login-direct profiles UMRS runs
natively so it can invoke Slurm.

## Model downloads and file handling

Model downloads are `MODEL_DOWNLOAD` durable jobs initiated through the
model API (`src/umrs/api/models.py`, `src/umrs/jobs/model_download*.py`);
inference containers use pre-staged files and run offline. Files are
project-scoped (`src/umrs/api/files.py`); presigned URLs are internal to
compute nodes; attempt-scoped STS credentials (ADR-043); object-store
backend selected by `S3_BACKEND` (ADR-055).

## Edge and R4 boundary

Users enter through `r4`, which reaches the edge routes (`/gw/<family>/*`)
and the ZeroMQ data plane. UMRS remains the implementation owner behind both.
Direct HTTP access to the sidecar is for harnesses, local development and
trusted operators. The design's target — UMRS as the single gateway over one
ZeroMQ channel — is decision D-01/D-02.

## Current docs

- `Readme.md` — repository overview and quick start (rewritten 2026-09-22)
- `doc/AI4S_PLATFORM_BOUNDARY.md` — branch topology and contracts preserved for the parent (rewritten 2026-09-22)
- `DEVELOPMENT_OBJECTIVES.md` — objectives and docstring design anchors (rewritten 2026-09-22)
- `doc/admin_guide.md`, `doc/user_guide.md`, `doc/onboarding.md`, `doc/access_control.md`, `doc/multi_users.md` — operations (pre-design-reset dates)
- `doc/zmq_api_contract.md` — ZeroMQ protocol
- `doc/rag_implementation.md`, `doc/set-up_pgvector.md`, `doc/design_user_defined_models.md`, `doc/s3_requirements_for_admin.md`
- `deploy/rccs/`, `deploy/umrs-front/`, `demos/README.md`
