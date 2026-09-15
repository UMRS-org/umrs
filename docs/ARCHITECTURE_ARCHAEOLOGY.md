# UMRS — Architecture Archaeology & Unification Strategy

**Date:** 2026-05-13
**Scope:** UMRS submodule (src/, tests/, deploy/, demos/)
**Branch:** `arch/archaeology-audit-2026-05-13`

---

## Table of Contents

- [Phase 1 — UMRS Internal Archaeology](#phase-1--umrs-internal-archaeology)
- [Phase 2 — Architectural Generations Within UMRS](#phase-2--architectural-generations-within-umrs)
- [Phase 3 — UMRS-Specific Invariants](#phase-3--umrs-specific-invariants)
- [Phase 4 — UMRS Internal Consistency Audit](#phase-4--umrs-internal-consistency-audit)
- [Phase 5 — UMRS Internal Target Architecture](#phase-5--umrs-internal-target-architecture)
- [Phase 6 — UMRS Safe Migration Strategy](#phase-6--umrs-safe-migration-strategy)
- [Phase 7 — UMRS Machine-Enforceable Architecture](#phase-7--umrs-machine-enforceable-architecture)

---

# Phase 1 — UMRS Internal Archaeology

## 1.1 The Dual-Era Server

UMRS is **not** a monolithic server — it is a dual-era hybrid where two architectural generations coexist in a single process:

```
┌────────────────────────────────────────────────────────────────┐
│                    server_umrs.py main()                       │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │ ZMQ Control  │  │ DurableJob     │  │ HTTP Sidecar       │  │
│  │ Loop (sync)  │  │ Runtime (async │  │ (FastAPI, port     │  │
│  │ Thread: main │  │ Thread: daemon)│  │ 8082, daemon)      │  │
│  └──────┬───────┘  └───────┬────────┘  └────────┬───────────┘  │
│         │                  │                    │              │
│    TaskCmd dispatch   DurableJobCommand     Gateway REST       │
│    S3 processing      Backend bridge       proxy routers       │
│    ↓                  ↓                    ↓                   │
│  core/UMRSThreads   jobs/store.py       api/*.py               │
│  core/MessageQueue   orchestrator/      (endpoints, jobs,      │
│  inference/           durable.py         models, RAG…)         │
│  server/processing                                             │
└────────────────────────────────────────────────────────────────┘
```

### Thread Architecture

| Thread | Role | Generation | Event Model |
|--------|------|-----------|-------------|
| **Main** | ZMQ control loop (`server_event_loop()`) | Gen1 | Synchronous poll loop |
| **ThreadPoolManager** | S3 object processing + embedding | Gen1 | Thread pool executor |
| **DurableJobRuntime** | Job scheduler + monitor + reconciler | Gen2 | `asyncio.run()` on daemon thread |
| **HTTP Sidecar** | FastAPI app for Gateway integration | Gen2 | `uvicorn.run()` on daemon thread |

### Startup Sequence

```
main()
  → argparse (admin? completion? → fast exit without ML imports)
  → perform_initial_config()
      → load JSON config
      → create ZMQ MessageQueue
      → provision S3 per user
      → load LocalEmbeddingBackend (torch)
  → start DurableJobRuntime (background async loop)
  → start HTTP Sidecar (background daemon thread)
  → write PID file
  → server_event_loop()  ← BLOCKS FOREVER
      → poll ZMQ control queue
      → dispatch commands via _REQUEST_REPLY_HANDLERS
```

## 1.2 Internal Dependency Graph

```
api/  ──────────▶ jobs/
      ──────────▶ model_registry/
      ──────────▶ orchestrator/
      ──────────▶ mcp/
      ──────────▶ rag/
      ──────────▶ plugins/
      ──────────▶ skills/
      ──────────▶ shared/
      ──────────▶ server/ (step_router, workflow_runner)

server/ ────────▶ jobs/
        ────────▶ model_registry/
        ────────▶ orchestrator/
        ────────▶ pipeline_sdk/
        ────────▶ core/
        ────────▶ shared/
        ────────▶ inference/

orchestrator/ ──▶ model_registry/ (durable.py only)
              ──▶ shared/
              ──▶ (SiteConfig is self-contained)

jobs/ ──────────▶ model_registry/ (model_download.py only)
      ──────────▶ orchestrator/ (adapters)

model_registry/ ─▶ (standalone, no internal UMRS deps)
mcp/ ────────────▶ (standalone)
rag/ ────────────▶ (standalone)
plugins/ ────────▶ pipeline_sdk/
pipeline_sdk/ ───▶ (standalone)
shared/ ─────────▶ core/ (some imports)
core/ ───────────▶ (standalone foundation)
inference/ ──────▶ (standalone, uses torch/transformers)
vision_service/ ─▶ (standalone, uses BentoML)
```

### Dependency Direction Summary

```
FOUNDATION (no internal deps):
  core/ → model_registry/ → mcp/ → rag/ → pipeline_sdk/ → inference/ → vision_service/

MID-LAYER (depends on foundation):
  shared/ → jobs/ → orchestrator/ → plugins/

TOP LAYER (depends on everything):
  api/ → server/

BRIDGE (connects generations):
  server/durable_job_runtime.py (async→sync)
  server/job_tracking_handler.py (ZMQ→durable jobs)
  server/model_registry_handler.py (ZMQ→model registry)
```

## 1.3 Data Flow Within UMRS

### Job Lifecycle Flow

```
Client Request (ZMQ or HTTP)
    │
    ├─[ZMQ]─▶ server_umrs.py → JOB_SUBMIT handler → DurableJobCommandBackend
    │                                                        │
    └─[HTTP]─▶ api/jobs.py → POST /umrs/jobs ────────────────┘
                                                             │
                                                    jobs/service.py
                                                    JobStore.create()
                                                             │
                                              ┌──────────────┘
                                              ▼
                                    jobs/scheduler.py
                                    (polls QUEUED jobs)
                                              │
                                    orchestrator/durable.py
                                    DurableOrchestrationService
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                         SlurmPodman     Kubernetes      DockerCompose
                          Backend         Backend          Backend
                              │               │               │
                              ▼               ▼               ▼
                          mini-RiVault    K8s Job       Compose stack
                              │               │               │
                         ┌────┘               │               │
                         ▼                    │               │
                    ReadinessPoller           │               │
                    (/v1/models)             │               │
                         │                    │               │
                    EndpointRegistry         │               │
                    (register endpoint)      │               │
                         │                    │               │
                    jobs/monitor.py          │               │
                    (poll completion)         │               │
                         │                    │               │
                    JobStore.complete()       │               │
```

### Endpoint Provisioning Flow

```
r4 endpoint up
    │
    ▼
Gateway → UMRS HTTP: POST /umrs/endpoints/provision
    │
    ▼
api/endpoints.py
    → orchestrator/factory.py → get_backend()
    → DurableOrchestrationService.provision()
        → JobStore.create(type=ENDPOINT_PROVISION)
        → Scheduler picks up → Backend.submit_job()
            │
            ├─ SlurmPodmanBackend: sbatch → serve-model.slurm
            │     → podman-compose up → vLLM + LiteLLM + NGINX
            │
            ├─ KubernetesBackend: kubectl apply → Job + Service
            │
            └─ DockerComposeBackend: docker-compose up
            
    → ReadinessPoller: poll /v1/models until responding
    → EndpointRegistry.register(project, model, node, port)
    → Job status → COMPLETED
```

## 1.4 Persistence Architecture

### PostgreSQL Tables

| Table | Module | Purpose | Schema Version |
|-------|--------|---------|----------------|
| `jobs` | jobs/schema.py | Job specs (immutable) | v2 |
| `job_attempts` | jobs/schema.py | Execution tries per job | v2 |
| `job_locks` | jobs/ha.py | Distributed advisory locks | v2 |
| `umrs_instances` | jobs/ha.py | UMRS instance heartbeats | v2 |
| `model_catalog_entries` | model_registry/schema.py | Model metadata + ACL | v5 |
| `model_policies` | model_registry/schema.py | Access policies + quotas | v5 |
| `model_releases` | model_registry/schema.py | Model version tracking | v5 |
| `mcp_servers` | mcp/ | MCP server registrations | v1 |
| `mcp_tools` | mcp/ | MCP tool registrations | v1 |
| `egress_audit_*` | migrations/ | EgressGuard audit logs | v1 |

### SQLite (Local)

| Store | Module | Purpose |
|-------|--------|---------|
| `diskcache.Cache` | core/UMRSThreads | Thread status sharing |
| `tasks.db` | core/UMRSTask | Task crash recovery |

### MinIO (S3)

| Bucket Pattern | Module | Purpose |
|----------------|--------|---------|
| `{user}-{bucket_type}` | shared/storage/ | Per-user data buckets |
| `s3://…/jobs/{job_id}/attempt_{n}/` | jobs/ | Attempt-scoped outputs |

### Dynamic Per-User Tables

| Table Pattern | Module | Purpose |
|---------------|--------|---------|
| `{user}_{object_type}` | server/sql.py | Per-user metadata tables |

## 1.5 Configuration Architecture

### SiteConfig (YAML-Based)

```yaml
# site.yaml structure
site:
  name: rccs-cloud
  arch: aarch64
scheduler:
  type: slurm
  partitions: [ng-dgx-m0, ng-dgx-m1, ...]
runtime:
  engine: podman-compose
  container_runtime: runc
  userns: host
  network_driver: netavark
storage:
  model_root: /lvs0/dne1/rccs-dmp/masaru.nagaso/umrs_models
  hf_cache_dir: /lvs0/dne1/rccs-dmp/masaru.nagaso/umrs_models/hf
  workspace_root: /tmp/umrs_workspace
  singularity_sif_dir: ...
network:
  litellm_master_key: ""  # from .env at runtime
  ports: {litellm: 8001, vllm: 8000, nginx: 8080, ...}
vllm:
  max_model_len: 4096
  max_num_seqs: 4
  ...
database:
  dsn: ...
partition_profiles:
  ng-dgx-m0:
    gpu_type: nvidia
    gpus_per_node: 4
    ...
```

### Environment Overlay

`SiteConfig.inject_env()` writes YAML values into `os.environ` **without overwriting**. This means:
1. Explicit env vars always win
2. YAML provides defaults
3. All modules can use `os.getenv()` regardless of config source

### Configuration Sources by Priority

| Priority | Source | Scope |
|----------|--------|-------|
| 1 (highest) | Environment variables | Runtime override |
| 2 | `.env` file (sourced in shell) | Deployment secrets |
| 3 | `site.yaml` via `inject_env()` | Site defaults |
| 4 | `OrchestratorConfig.from_env()` | Module defaults |

---

# Phase 2 — Architectural Generations Within UMRS

## 2.1 Generation Map

### Gen1: ZMQ Monolith (Active Foundation)

| Component | Status | Should Migrate? |
|-----------|--------|-----------------|
| `server_umrs.py` (main loop) | **Active core** | No — handles real-time S3 processing |
| `client_umrs.py` (CLI) | **Active core** | Eventually → structured `umrs.cli` |
| `core/MessageQueue` | **Active transport** | No — ZMQ is the data-plane transport |
| `core/TaskManager` | **Active** | No — thread↔task association still needed |
| `core/UMRSTask` | **Active** | Consider migrating to PostgreSQL |
| `core/UMRSThreads` | **Active** | No — thread pool management still needed |
| `core/TaskCmdStatus` | **Active** | Stable enum — no migration needed |
| `core/thread_pool` | **Active** | No — bounded executor with backpressure |
| `core/mq_posix_ipc` | **Dead** | Remove |
| `core/mq_sysv_ipc` | **Dead** | Remove |

### Gen2: Durable Platform (Target Architecture)

| Component | Status | Maturity |
|-----------|--------|----------|
| `jobs/` | **Stable production** | High — PostgreSQL-backed, crash-safe, HA |
| `orchestrator/` | **Stable production** | High — Slurm, K8s, Compose backends |
| `model_registry/` | **Stable production** | High — multi-scope ACL, Postgres |
| `api/` | **Stable production** | High — 14 REST routers |
| `rag/` | **Stable production** | Medium — pipeline definitions, search |
| `mcp/` | **Stable production** | Medium — server/tool registry |
| `server/http_sidecar` | **Stable production** | High — FastAPI app for Gateway |
| `pipeline_sdk/` | **Active development** | Medium — DAG scheduler, step templates |
| `plugins/` | **Active development** | Low — runners defined, sandbox incomplete |
| `connectors/` | **Active development** | Low — 8 connectors, likely untested |

### Bridge Components

| Component | Role |
|-----------|------|
| `server/durable_job_runtime.py` | Starts Gen2 async loop on daemon thread |
| `server/job_tracking_handler.py` | Dispatches Gen1 ZMQ commands to Gen2 job store |
| `server/model_registry_handler.py` | Dispatches Gen1 ZMQ commands to Gen2 model registry |
| `server/workflow_registry_handler.py` | Dispatches Gen1 ZMQ commands to Gen2 workflow catalog |

### Skeletal Placeholders (Future Generations)

| Component | Intent | Current State |
|-----------|--------|--------------|
| `knowledge_graph/` | Entity/relation data models | ~100 LOC, data classes only |
| `voice/` | STT/TTS adapters | ~120 LOC, Whisper stub |
| `recipes/` | Recipe templates | ~80 LOC, data classes only |
| `tenants/` | Multi-tenant isolation | ~80 LOC, data classes only |
| `scheduler/` | Cron scheduling | ~100 LOC, data classes + trigger matcher |
| `skills/` | Skill registry | ~80 LOC, data classes only |

## 2.2 The Bridge Pattern (Detailed)

The Gen1↔Gen2 bridge is the most architecturally interesting pattern in UMRS:

```python
# server_umrs.py — Gen1 ZMQ loop
_REQUEST_REPLY_HANDLERS = {
    "JOB_SUBMIT": _handle_job_submit,     # → DurableJobCommandBackend
    "JOB_STATUS": _handle_job_status,     # → DurableJobCommandBackend
    "JOB_CANCEL": _handle_job_cancel,     # → DurableJobCommandBackend
    "MODEL_REGISTER": _handle_model_reg,  # → ModelRegistryHandler
    "MODEL_LIST": _handle_model_list,     # → ModelRegistryHandler
    ...
}

# server/durable_job_runtime.py — Bridge
class DurableJobRuntime:
    def start(self):
        """Start the async event loop on a daemon thread."""
        self._thread = threading.Thread(
            target=self._run_async_loop,
            daemon=True,
        )
        self._thread.start()
    
    def _run_async_loop(self):
        asyncio.run(self._main())

# server/job_tracking_handler.py — Sync→Async adapter
class DurableJobCommandBackend:
    def handle(self, cmd, payload):
        """Synchronous entry point called from ZMQ loop."""
        future = asyncio.run_coroutine_threadsafe(
            self._async_handle(cmd, payload),
            self._runtime.loop,
        )
        return future.result(timeout=30)
```

This bridge is **not tech debt** — it is a deliberate architectural decision:
- Gen1's ZMQ loop cannot be made async without a full rewrite
- Gen2's durable job system requires async for concurrent I/O
- The bridge is minimal and well-tested

## 2.3 Why the ZMQ Loop Persists

The Gen1 ZMQ loop handles two workloads that are **not suitable** for the durable job system:

1. **Real-time S3 object processing**: When a user uploads files, they're processed immediately (embeddings, metadata extraction). These are short-lived, CPU-bound tasks that benefit from a thread pool, not a Postgres-backed job queue.

2. **Low-latency HPC-local data transfer**: ZMQ ROUTER/DEALER provides ~10μs latency for binary transfers between login and compute nodes. HTTP would add serialization overhead.

The correct migration path is **not** to replace the ZMQ loop, but to gradually move durable operations (jobs, models, workflows) to Gen2 while keeping real-time operations in Gen1.

---

# Phase 3 — UMRS-Specific Invariants

## 3.1 Database Invariants

### Advisory Lock for Job Attempts

```sql
-- Invariant C: No concurrent attempts per job
SELECT pg_try_advisory_lock(hashtext(job_id::text))
-- If false → another UMRS instance owns this job → skip
```

This lock is **not** a row lock — it's a session-level advisory lock. It survives connection pooling but NOT process crashes. The Reconciler handles post-crash cleanup by marking all in-flight attempts as `LOST`.

### Schema Version Independence

Each module manages its own schema version independently:
- `jobs`: v2
- `model_registry`: v5
- `mcp`: v1
- `egress_audit`: v1

There is **no unified migration framework** (no Alembic, no Flyway). Each module's `schema.py` contains `CREATE TABLE IF NOT EXISTS` + version checks. This means modules can be upgraded independently but there's no rollback mechanism.

## 3.2 Orchestrator Invariants

### Backend Immutability

Once a job's `execution_authority` is set, it never changes. This is enforced by the `JobStore.create()` method writing it at creation time and no `update()` method modifying it.

### Endpoint Registry Thread Safety

`EndpointRegistry` uses a `threading.Lock` for mutations. This is correct for the current single-process model but would need replacement for multi-process deployments.

### ReadinessPoller Contract

The poller hits `/v1/models` on the endpoint and checks for a successful JSON response. This is the **authoritative** readiness check — no other health check is trusted.

## 3.3 Mini-RiVault Invariants

### Container Composition (Podman Compose)

The `serve-model.yaml` compose file defines profiles:
- `harness`: vLLM + LiteLLM + NGINX (production)
- `vllm`: vLLM only
- `vision-bentoml`: BentoML vision serving
- `litellm-db`: LiteLLM with PostgreSQL (for ARM schema init)

### GPU Vendor Overrides

`x_nvidia-override.yaml` and `x_rocm-override.yaml` provide vendor-specific GPU configuration. The Slurm script detects the GPU vendor and applies the correct override.

### ARM Architecture Workaround

On ARM (aarch64), the LiteLLM Prisma schema-engine is x86-64 only. The Slurm script extracts LiteLLM's bundled SQL migrations and applies them directly with `psql`. This is a **critical workaround** — if LiteLLM updates its schema, this extraction must be re-verified.

## 3.4 S3 Invariants

### Per-User Bucket Isolation

Each user gets their own S3 buckets: `{user}-{bucket_type}`. The bucket names are derived from the username, ensuring data isolation.

### Attempt-Scoped Outputs

All job outputs go to `s3://…/jobs/{job_id}/attempt_{n}/…`. Only the `commit_attempt_id` is used for reads. Failed attempt outputs remain in S3 but are not visible through the commit pointer.

## 3.5 HA Invariants

### Heartbeat-Based Instance Tracking

Each UMRS instance writes a heartbeat to `umrs_instances` every 30 seconds. If a heartbeat is stale for >90 seconds, the instance is considered dead and its locks can be reclaimed.

### Reconciler Boot Sequence

On startup:
1. Mark all in-flight attempts (from dead instances) as `LOST`
2. Requeue parent jobs (if retries remain)
3. Create new attempt from scratch
4. **No backend cleanup** — rely on attempt-scoped isolation

---

# Phase 4 — UMRS Internal Consistency Audit

## 4.1 Architecture-Damaging Issues

### 4.1.1 In-Memory Pipeline Store

`api/pipelines.py` uses an in-memory dictionary with explicit comment: "Replace with DB in production."

**Impact:** Pipeline definitions are lost on UMRS restart. This is the only API resource without persistence.

### 4.1.2 Dual Entry Points

`server_umrs.py` (686 LOC, Gen1) and `umrs/server/main.py` (160 LOC, Gen2) both exist. The Gen2 entry point delegates to Gen1 for full server mode.

**Impact:** Confusing for developers — which is the "real" entry point? Answer: `umrs.server.main:main` is the registered entry point; it delegates to `server_umrs.main()`.

### 4.1.3 Dual Client Programs

`client_umrs.py` (2,403 LOC, Gen1 monolith) and `umrs/client/` (package, Gen2 structured) both exist. The Gen1 monolith is the actual CLI entry point.

**Impact:** The monolithic client is hard to maintain and test. However, it works and is stable.

## 4.2 Consistency Observations

### 4.2.1 Dataclass Consistency ✅

All domain models use stdlib `dataclasses`. Pydantic is only used in the `api/` layer (FastAPI requirement). This is clean and intentional.

### 4.2.2 Logging Consistency ✅

All modules use `logging.getLogger(__name__)`. No structlog, no print statements in production code.

### 4.2.3 Error Handling Pattern

Each module defines its own exceptions with no common base:

```
JobStoreError, HeartbeatError, LockError
ModelNotFoundError, DuplicateModelError, PermissionDeniedError, PathNotAllowedError
ResolutionError, SubmissionError, LaunchError
ServerOverloadedError, NetworkUnreachableError
SkillValidationError, UserStoreError
K8sOnSlurmNotEnabledError
```

**Assessment:** 16+ exceptions with no `UMRSError` root class. Catching "any UMRS error" requires `except Exception`. This makes error handling in the API layer coarser than necessary.

### 4.2.4 Configuration Access

Modules access configuration via `os.getenv()` after `SiteConfig.inject_env()` has run. This is consistent but creates a hidden dependency on initialization order.

### 4.2.5 Global Singletons

Several modules use module-level singletons with `set_*/get_*` functions:

```python
# Common pattern in model_registry, mcp, etc.
_store: Optional[ModelStore] = None

def set_store(store: ModelStore) -> None:
    global _store
    _store = store

def get_store() -> ModelStore:
    if _store is None:
        raise RuntimeError("Store not initialized")
    return _store
```

**Assessment:** Simple DI substitute. Works for single-process, but complicates testing (must call `set_*` before each test).

## 4.3 Harmless Inconsistencies

| Inconsistency | Cause | Assessment |
|---------------|-------|-----------|
| PascalCase filenames in `core/` | Gen1 naming convention | Harmless — legacy convention |
| Single-file packages (`__init__.py` only) | Some modules are small | Harmless — can grow |
| `server/runtime.py` deprecated shim | Migration residue | Remove (Priority 2 in platform doc) |
| Mixed file organization in `server/` | Some files are Gen1, some Gen2 | **Tolerable** — they're bridging layers |

---

# Phase 5 — UMRS Internal Target Architecture

## 5.1 Target Module Structure

```
umrs/src/umrs/
├── __init__.py
├── errors.py                    # NEW: UMRSError root class
│
├── core/                        # Gen1: ZMQ, threading, task management
│   ├── MessageQueue.py          # ZMQ ROUTER/DEALER transport
│   ├── TaskCmdStatus.py         # Command enums
│   ├── TaskManager.py           # Thread↔task tracking
│   ├── UMRSTask.py              # SQLite task persistence
│   ├── UMRSThreads.py           # Thread pool management
│   └── thread_pool.py           # Bounded executor
│
├── server/                      # Server runtime (Gen1+Gen2 bridge)
│   ├── main.py                  # Entry point (delegates to server_umrs)
│   ├── http_sidecar.py          # FastAPI app for Gateway
│   ├── durable_job_runtime.py   # Async→sync bridge
│   ├── job_tracking_handler.py  # ZMQ→durable job dispatch
│   ├── model_registry_handler.py # ZMQ→model registry dispatch
│   ├── workflow_registry_handler.py
│   ├── processing.py            # S3 object processing
│   ├── provisioning.py          # User provisioning
│   ├── step_router.py           # Workflow step routing
│   ├── step_executor.py         # Workflow step execution
│   ├── workflow_runner.py       # Workflow execution engine
│   └── ...
│
├── api/                         # REST API routers (Gen2)
│   ├── endpoints.py
│   ├── jobs.py
│   ├── models.py
│   ├── workflows.py
│   ├── pipelines.py
│   ├── rag.py
│   ├── mcp.py
│   ├── vision.py
│   ├── files.py
│   ├── skills.py
│   ├── egress_audit.py
│   └── llm.py
│
├── jobs/                        # Durable job system (Gen2, standalone)
│   ├── models.py
│   ├── store.py
│   ├── scheduler.py
│   ├── reconciler.py
│   ├── monitor.py
│   ├── service.py
│   ├── ha.py
│   └── ...
│
├── orchestrator/                # Multi-backend orchestration (Gen2, standalone)
│   ├── base.py
│   ├── factory.py
│   ├── site_config.py
│   ├── endpoint_registry.py
│   ├── readiness.py
│   ├── janitor.py
│   ├── durable.py
│   ├── slurm_backend.py
│   ├── slurm_podman_backend.py
│   ├── kubernetes_backend.py
│   ├── compose_backend.py
│   ├── hybrid/
│   └── ...
│
├── model_registry/              # Model catalog (Gen2, standalone)
├── mcp/                         # MCP registry (Gen2, standalone)
├── rag/                         # RAG definitions (Gen2, standalone)
├── pipeline_sdk/                # Workflow engine (Gen2, standalone)
├── plugins/                     # User code execution (Gen2)
├── connectors/                  # Data connectors (Gen2)
│
├── inference/                   # Embedding models (Gen1, standalone)
├── vision_service/              # BentoML vision (Gen2, standalone)
│
├── shared/                      # Cross-cutting utilities
├── admin/                       # User/daemon management
├── security/                    # Content scanning
├── client/                      # HTTP client package
├── cli/                         # CLI framework
├── tui/                         # Terminal UI
│
├── mini-rivault-llm/            # Container templates + Slurm scripts
└── migrations/                  # SQL schema migrations
```

## 5.2 Internal Dependency Rules

### Allowed Dependencies

```
api/ → jobs/, model_registry/, orchestrator/, mcp/, rag/, plugins/,
       skills/, shared/, server/ (step_router, workflow_runner)

server/ → jobs/, model_registry/, orchestrator/, pipeline_sdk/,
          core/, shared/, inference/

orchestrator/ → model_registry/ (durable.py only), shared/

jobs/ → model_registry/ (model_download only), orchestrator/ (adapters)

plugins/ → pipeline_sdk/

shared/ → core/ (minimal)
```

### Forbidden Dependencies

```
core/ → api/, jobs/, orchestrator/, model_registry/, mcp/, rag/
  (core is the foundation — no upward deps)

shared/ → api/, jobs/
  (shared is the foundation — no upward deps)

model_registry/ → jobs/, orchestrator/, api/
  (model_registry is standalone)

mcp/ → jobs/, orchestrator/, api/
  (mcp is standalone)

rag/ → jobs/, orchestrator/, api/
  (rag is standalone)

inference/ → api/, server/ (except shared imports)
  (inference is standalone)

pipeline_sdk/ → api/, server/
  (pipeline_sdk is standalone)
```

## 5.3 Lifecycle Authority

| Lifecycle | Owner | Pattern |
|-----------|-------|---------|
| UMRS server startup/shutdown | `server_umrs.py` main() | Process lifecycle |
| HTTP sidecar startup | `server/http_sidecar.py` | Daemon thread, auto-start |
| Durable job runtime | `server/durable_job_runtime.py` | Daemon thread, auto-start |
| Job creation/scheduling | `jobs/service.py` | API-driven or ZMQ-driven |
| Job execution | `jobs/scheduler.py` → orchestrator/ | Background polling loop |
| Endpoint lifecycle | `orchestrator/` backends | Backend-specific (Slurm/K8s/Compose) |
| Endpoint readiness | `orchestrator/readiness.py` | HTTP polling |
| Endpoint cleanup | `orchestrator/janitor.py` | Background sweep |
| Model registration | `model_registry/store/` | API-driven |
| Schema initialization | Each module's `schema.py` | `CREATE TABLE IF NOT EXISTS` at startup |

---

# Phase 6 — UMRS Safe Migration Strategy

## 6.1 Priority 1: Remove Dead Code (Risk: None)

Remove files that have no callers:
- `core/mq_posix_ipc.py`
- `core/mq_sysv_ipc.py`
- `server/runtime.py` (deprecated shim)

## 6.2 Priority 2: Add UMRSError Root Class (Risk: None)

Create `errors.py` with `class UMRSError(Exception)` and make all domain exceptions inherit from it. This is purely additive — existing `except` clauses still work.

## 6.3 Priority 3: Pipeline Store Persistence (Risk: Medium)

Replace the in-memory `PipelineStore` in `api/pipelines.py` with PostgreSQL-backed storage. Add a migration for the `pipelines` table.

## 6.4 Priority 4: client_umrs.py Modularization Prep (Risk: Low)

The 2,403-LOC monolithic CLI is maintainable today but growing. Preparation steps:
1. Document which sections of `client_umrs.py` delegate to `umrs.client/` package modules
2. Identify sections that can be extracted next
3. Do NOT rewrite — extract incrementally as changes are needed

## 6.5 Priority 5: Schema Migration Framework (Risk: Medium)

Currently each module does `CREATE TABLE IF NOT EXISTS` with inline version checks. This works but has no rollback mechanism.

Options:
- **Minimal**: Keep current pattern but add a `schema_versions` table that tracks all module schema versions centrally
- **Standard**: Introduce Alembic for PostgreSQL migrations
- **Recommended**: Minimal approach — it matches the current module independence

## 6.6 Priority 6: Connector Testing (Risk: Low)

The 8 connectors in `connectors/` appear untested. Before any production use:
1. Add unit tests for each connector
2. Add integration tests with mocked external services
3. Verify error handling and authentication

## 6.7 Priority 7: Skeletal Module Triage (Risk: None)

For each skeletal module, add a docstring documenting:
- Expected Phase 2 milestone
- Whether the data models are stable
- Integration points with other modules

---

# Phase 7 — UMRS Machine-Enforceable Architecture

## 7.1 UMRS Internal Import Boundary Validator

Extend the existing `tools/import_boundary_validator.py` with UMRS-internal rules:

```python
UMRS_INTERNAL_RULES = [
    {
        "source": "umrs/src/umrs/core",
        "forbidden": ["umrs.api", "umrs.jobs", "umrs.orchestrator",
                       "umrs.model_registry", "umrs.mcp", "umrs.rag"],
        "description": "core/ is the foundation — no upward dependencies",
    },
    {
        "source": "umrs/src/umrs/shared",
        "forbidden": ["umrs.api", "umrs.jobs"],
        "description": "shared/ is foundation — no upward dependencies",
    },
    {
        "source": "umrs/src/umrs/model_registry",
        "forbidden": ["umrs.jobs", "umrs.orchestrator", "umrs.api"],
        "description": "model_registry/ is standalone",
    },
    {
        "source": "umrs/src/umrs/mcp",
        "forbidden": ["umrs.jobs", "umrs.orchestrator", "umrs.api"],
        "description": "mcp/ is standalone",
    },
    {
        "source": "umrs/src/umrs/rag",
        "forbidden": ["umrs.jobs", "umrs.orchestrator", "umrs.api"],
        "description": "rag/ is standalone",
    },
    {
        "source": "umrs/src/umrs/pipeline_sdk",
        "forbidden": ["umrs.api", "umrs.server"],
        "description": "pipeline_sdk/ is standalone",
    },
    {
        "source": "umrs/src/umrs/inference",
        "forbidden": ["umrs.api", "umrs.server"],
        "description": "inference/ is standalone",
    },
]
```

## 7.2 Schema Version Tracking

```yaml
# umrs/schema_versions.yaml
jobs: 2
model_registry: 5
mcp: 1
egress_audit: 1
pipelines: 0  # not yet persistent
```

CI check: On every PR, verify that if a module's schema SQL changed, the version number was bumped.

## 7.3 Singleton Initialization Guard

Add a test that verifies all module singletons are properly initialized before the HTTP sidecar starts accepting requests:

```python
def test_all_stores_initialized():
    """Verify that HTTP sidecar setup initializes all stores."""
    # After http_sidecar.create_app() runs:
    assert get_job_store() is not None
    assert get_model_store() is not None
    assert get_mcp_store() is not None
    assert get_endpoint_registry() is not None
```

## 7.4 Thread Safety Audit

Add a CI check that scans for mutable global state that lacks synchronization:

```python
# Check that EndpointRegistry uses locks
# Check that diskcache.Cache is used for cross-thread state
# Check that no module-level mutable dicts are accessed without locks
```

## 7.5 Crash Recovery Test

Add an integration test that:
1. Starts UMRS
2. Creates a job
3. Kills UMRS mid-execution
4. Restarts UMRS
5. Verifies the Reconciler marks the attempt as LOST
6. Verifies the job is requeued

This invariant is documented but should be continuously tested.

---

# Appendix: UMRS Risk Matrix

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|-----------|
| Advisory lock loss on crash | Medium | Low | Reconciler marks LOST + requeue |
| Schema drift between modules | Low | Medium | Add schema_versions.yaml tracking |
| Pipeline state loss on restart | High | High | **Priority 3**: Add persistence |
| ARM LiteLLM schema breakage | High | Low | Pin LiteLLM version, add migration test |
| Connector auth credential leak | Medium | Low | Connectors not yet in production use |
| Thread pool exhaustion | Medium | Low | `ServerOverloadedError` with backpressure |
| Stale endpoint in registry | Low | Medium | Janitor background cleanup |
| ZMQ message ordering | Low | Low | ZMQ ROUTER guarantees per-connection ordering |
