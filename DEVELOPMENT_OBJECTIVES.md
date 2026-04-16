# UMRS Development Objectives

> Comprehensive roadmap for expanding UMRS functionality.
> Created: 2026-04-05

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| :white_check_mark: | Complete |
| :construction: | In Progress |
| :white_circle: | Planned |

---

## A. New Library Features (B1–B10)

### B1 — Streaming Progress Tracker

**Module**: `src/umrs/progress.py`
**Status**: :white_check_mark:

Real-time progress reporting for long-running operations (uploads, pipeline runs,
inference). Provides callback-based and polling-based APIs.

**Key Components**:
- `ProgressTracker` — thread-safe progress state manager
- `ProgressCallback` — protocol for progress consumers
- `ConsoleProgressBar` — terminal bar renderer
- `ProgressMixin` — mixin for Pipeline subclasses

**Integration Points**:
- Pipeline SDK `compute()` phase
- File upload/download in `client_umrs.py`
- BentoML task polling

---

### B2 — Batch Upload with Resume

**Module**: `src/umrs/batch_upload.py`
**Status**: :white_check_mark:

Resume-capable batch file uploads using a local manifest that tracks which files
have been successfully uploaded. Supports parallel uploads and automatic retry on
transient failures.

**Key Components**:
- `BatchUploadManager` — orchestrates multi-file uploads
- `UploadManifest` — JSON-based tracking of file states (pending/uploaded/failed)
- `UploadWorker` — per-file upload with retry logic

---

### B3 — Result Comparison & Diff

**Module**: `src/umrs/result_diff.py`
**Status**: :white_check_mark:

Compare results across pipeline runs, experiments, or model versions. Supports
numeric (embeddings, scores), image (SSIM/MSE), and text (edit distance) comparisons.

**Key Components**:
- `ResultComparator` — main comparison engine
- `DiffReport` — structured diff output
- `NumericDiff`, `ImageDiff`, `TextDiff` — type-specific comparators

---

### B4 — Webhook Notifications

**Module**: `src/umrs/webhooks.py`
**Status**: :white_check_mark:

Event-driven webhook system for notifying external services about UMRS events
(job completion, pipeline errors, model registration, etc.).

**Key Components**:
- `WebhookManager` — subscription registry and dispatcher
- `WebhookEvent` — typed event envelope (job_completed, pipeline_failed, etc.)
- `WebhookDelivery` — async HTTP delivery with retry and dead-letter queue

**Supported Events**:
- `job.completed`, `job.failed`
- `pipeline.started`, `pipeline.completed`, `pipeline.failed`
- `model.registered`, `model.deprecated`
- `upload.completed`

---

### B5 — Resource Tracking

**Module**: `src/umrs/resource_tracker.py`
**Status**: :white_check_mark:

Track CPU, memory, GPU, storage, and network resource usage per user, project,
and job. Enables capacity planning and quota enforcement.

**Key Components**:
- `ResourceTracker` — usage collection and aggregation
- `ResourceSnapshot` — point-in-time resource readings
- `UsageReport` — per-user/project resource summaries
- `QuotaPolicy` — configurable per-user resource limits

---

### B6 — Data Versioning

**Module**: `src/umrs/data_versioning.py`
**Status**: :white_check_mark:

Version-track datasets in S3 buckets. Each version captures a snapshot of bucket
contents (object keys + checksums). Supports tagging, rollback, and lineage queries.

**Key Components**:
- `DataVersionManager` — create/list/restore versions
- `DataVersion` — immutable snapshot (version_id, timestamp, manifest)
- `VersionManifest` — mapping of object keys to ETags/checksums

---

### B7 — Pipeline Templates

**Module**: `src/umrs/pipeline_templates.py`
**Status**: :white_check_mark:

Catalog of reusable workflow templates for common analysis patterns. Templates
are parameterized workflow.json skeletons that users fill in.

**Built-in Templates**:
- `image_embedding` — upload images → compute embeddings → export CSV
- `text_analysis` — upload text → tokenize → embed → search
- `defect_detection` — upload images → segment → classify defects
- `video_analysis` — upload video → extract frames → analyze
- `custom_pipeline` — empty scaffold for user-defined workflows

**Key Components**:
- `TemplateRegistry` — discover and instantiate templates
- `PipelineTemplate` — parameterized template definition
- `TemplateParameter` — typed parameter with defaults and validation

---

### B8 — Pipeline Builder (Programmatic API)

**Module**: `src/umrs/pipeline_builder.py`
**Status**: :white_check_mark:

Fluent API for constructing pipeline workflows programmatically, as an alternative
to writing workflow.json files manually.

**Usage**:
```python
from umrs.pipeline_builder import PipelineBuilder

workflow = (
    PipelineBuilder("my-analysis")
    .add_step("preprocess", "steps/preprocess.py:Preprocess")
    .add_step("embed", "steps/embed.py:Embed", depends_on=["preprocess"])
    .add_step("search", "steps/search.py:Search", depends_on=["embed"])
    .set_execution_mode("IN_PROCESS")
    .set_source("/path/to/steps/")
    .build()
)
```

**Key Components**:
- `PipelineBuilder` — fluent builder for WorkflowDefinition
- `StepBuilder` — per-step configuration
- Validates DAG integrity on `build()`

---

### B9 — Experiment Export/Import

**Module**: `src/umrs/experiment_io.py`
**Status**: :white_check_mark:

Export full experiment state (config, data manifest, results, embeddings) to a
portable archive. Import archives to reproduce experiments on different UMRS
instances.

**Key Components**:
- `ExperimentExporter` — serialize experiment to ZIP archive
- `ExperimentImporter` — reconstitute experiment from archive
- `ExperimentManifest` — metadata (user, timestamp, versions, config hash)

**Archive Structure**:
```
experiment_<name>_<timestamp>.zip
├── manifest.json
├── config/
│   └── user_config.json
├── data/
│   └── data_manifest.json    (S3 object list + checksums)
├── results/
│   └── embeddings.csv
└── metadata/
    └── pipeline_log.json
```

---

### B10 — Scheduled Recurring Jobs

**Module**: `src/umrs/scheduler.py`
**Status**: :white_check_mark:

Cron-like scheduler for recurring UMRS jobs. Supports fixed intervals, cron
expressions, and one-shot delayed execution. Persists schedules to survive
server restarts.

**Key Components**:
- `JobScheduler` — manages schedule lifecycle
- `Schedule` — schedule definition (cron expression, job spec, enabled flag)
- `ScheduleStore` — persistent schedule storage

**Supported Patterns**:
- Fixed interval: `every 30m`, `every 2h`
- Cron expression: `0 */6 * * *` (every 6 hours)
- One-shot: `at 2026-04-10T00:00:00`

---

## B. New Demos (15–21)

### Demo 15 — Complex DAG Workflow

**Directory**: `demos/15_complex_dag/`
**Status**: :white_check_mark:
**Server Required**: No (local Pipeline SDK)

Demonstrates a multi-step workflow with parallel branches and fan-in.
Uses DAGScheduler to execute independent steps concurrently.

**Workflow DAG**:
```
  [load_data]
      │
  ┌───┴───┐
  │       │
[analyze_text] [analyze_metadata]
  │       │
  └───┬───┘
      │
  [merge_results]
      │
  [generate_report]
```

---

### Demo 16 — Custom Server-Side Workflow

**Directory**: `demos/16_custom_workflow/`
**Status**: :white_check_mark:
**Server Required**: Yes

Demonstrates `upload-workflow` to push a custom workflow.json + step scripts to
the UMRS server, then triggers server-side execution.

---

### Demo 17 — Multi-User Isolation

**Directory**: `demos/17_multi_user/`
**Status**: :white_check_mark:
**Server Required**: Yes

Demonstrates that two users (user1, user2) have isolated S3 buckets, databases,
and cannot see each other's data.

---

### Demo 18 — Model Lifecycle

**Directory**: `demos/18_model_lifecycle/`
**Status**: :white_check_mark:
**Server Required**: Yes

Full model lifecycle: register → list → use for inference → deprecate.
Demonstrates model scoping (system vs user).

---

### Demo 19 — RAG Advanced

**Directory**: `demos/19_rag_advanced/`
**Status**: :white_check_mark:
**Server Required**: Yes (RAG service on port 8081)

Demonstrates all RAG search modes: vector-only, BM25-only, hybrid (RRF),
and reranking. Compares retrieval quality across modes.

---

### Demo 20 — Durable Job Orchestration

**Directory**: `demos/20_durable_jobs/`
**Status**: :white_check_mark:
**Server Required**: Yes

Demonstrates the durable job system: submit with idempotency key, poll status,
handle retries, and cancel a running job.

---

### Demo 21 — Remote Plugin Execution

**Directory**: `demos/21_remote_plugins/`
**Status**: :white_check_mark:
**Server Required**: No (local simulation)

Demonstrates the plugin system with different execution backends.
Shows PluginSpec definition, PluginRunner usage, and audit logging.

---

## C. Testing Strategy

Each new module has a corresponding test file under `tests/`:

| Module | Test File |
|--------|-----------|
| `progress.py` | `tests/test_progress.py` |
| `batch_upload.py` | `tests/test_batch_upload.py` |
| `result_diff.py` | `tests/test_result_diff.py` |
| `webhooks.py` | `tests/test_webhooks.py` |
| `resource_tracker.py` | `tests/test_resource_tracker.py` |
| `data_versioning.py` | `tests/test_data_versioning.py` |
| `pipeline_templates.py` | `tests/test_pipeline_templates.py` |
| `pipeline_builder.py` | `tests/test_pipeline_builder.py` |
| `experiment_io.py` | `tests/test_experiment_io.py` |
| `scheduler.py` | `tests/test_scheduler.py` |

All tests run without external dependencies (no server, no S3, no GPU).

---

## D. Priority & Dependencies

```
Phase 1 (Core Utilities):
  B1 (Progress) → B2 (Batch Upload) → B5 (Resource Tracking)

Phase 2 (Data Management):
  B6 (Versioning) → B9 (Export/Import) → B3 (Diff)

Phase 3 (Pipeline Enhancements):
  B7 (Templates) → B8 (Builder) → B10 (Scheduler)

Phase 4 (Integration):
  B4 (Webhooks) → Demos 15–21

All phases run in parallel where possible.
```

---

## E. Verification Summary

All 10 library modules (B1–B10) and 7 demo scripts (15–21) are implemented.
**170 unit tests** across 10 test files pass (pytest, no external deps).

---

## F. Rich Vision Results — Serialisation Strategy

**Current (Plan A — Pickle inline)**: The BentoML vision service
`infer()` endpoint returns `pickle.dumps(result_dict)`.  Adapters
include heavy objects (numpy mask arrays, torch tensors) directly in the
result dict.  The UMRS server-side client (`bentoml_adapter.py`,
`bentoml_client.py`) deserialises with `pickle.loads()`.

**Trade-offs**:
- Simple, zero-copy on the adapter side.
- Transfer is **mandatory** — callers that only need boxes/scores still
  receive the full mask payload.
- For large images (8192×8192) with many detected objects the pickled
  response can reach hundreds of MiB, which adds latency to the BentoML
  task-queue result retrieval.

**Future migration (Plan B — Sidecar artifacts)**:
Upload heavy data (masks as compressed `.npz`, features as `.pt`) to S3
inside the adapter, return only lightweight S3 URIs in the result dict.
The result dict stays JSON-serialisable and small (<1 KiB).  The
`output_specs.py` artifact declarations and presigned-URL infrastructure
already support multiple outputs per adapter.

Key changes for Plan B:
1. Extend `output_specs.py` so SAM3 declares three outputs:
   `{base}_sam3_seg.png`, `{base}_sam3_masks.npz`, `{base}_sam3_meta.json`.
2. In `sam3_adapter.py`, upload masks via `np.savez_compressed` → presigned PUT.
3. Return S3 keys in the response dict; remove raw arrays.
4. Revert `service.py` return type from `bytes` back to `dict`.
5. Remove `pickle.loads` in `bentoml_adapter.py` / `bentoml_client.py`.

**Plan C (Hybrid threshold)**: Return small results inline (pickle),
upload large ones as sidecars based on a configurable byte threshold.
Combines the simplicity of Plan A for small payloads with the lazy
transfer of Plan B for large ones.
