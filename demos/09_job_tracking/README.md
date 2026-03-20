# Demo 09 — Job Submission & Service Lifecycle

Explicit job submission for operations that are **not triggered by file uploads**.

The basic UMRS flow (upload file → compute embeddings) is fully automatic via
`PROCESS_OBJECTS` — no job submission needed. This demo covers 3 cases where
you do need an explicit `job submit`:

## Prerequisites

- UMRS server running (see `demos/00_start_services/`)
- `umrs-client` available

## Why 3 Job Types?

### 1. `endpoint_provision` — Pre-warm a service before uploads

Normally the server launches a Slurm/K8s job **lazily** on the first file
upload, which adds cold-start latency. If you know you'll process many
batches, pre-warm the endpoint:

```bash
# Pre-warm Mini-RiVault/BentoML endpoint (Slurm job starts immediately):
umrs-client job submit endpoint_provision \
    --spec '{"model": "sam3", "backend": "slurm"}' \
    --project my-project --keep-alive

# Now uploads use the warm endpoint — no cold start:
umrs-client upload_dir /data/images/batch-1/
umrs-client upload_dir /data/images/batch-2/

# When done, release the node hours:
umrs-client job cancel <job_id>
```

### 2. `index_build` — Rebuild RAG index after bulk inserts

After embeddings accumulate in pgvector, the IVFFlat index may need
re-clustering for optimal search performance. This is a database
operation, not a file upload:

```bash
# Rebuild the RAG index after a large ingestion run:
umrs-client job submit index_build \
    --spec '{"database": "xp_spring8", "table": "image_embeddings"}' \
    --project spring8
```

### 3. `cleanup` — Teardown resources explicitly

Force-clean endpoints, stale S3 objects, or database tables. The janitor
handles TTL-based cleanup automatically, but sometimes you want immediate
teardown:

```bash
# Teardown a specific endpoint:
umrs-client job submit cleanup \
    --spec '{"target": "endpoint", "job_id": "job-abc123"}' \
    --project my-project

# Clean stale S3 objects:
umrs-client job submit cleanup \
    --spec '{"target": "s3", "bucket": "old-experiment"}' \
    --project my-project
```

### Cancel a job

```bash
umrs-client job cancel job-abc123def456
```

### Query running jobs

Use the existing `status` command — no separate job-list needed:

```bash
umrs-client status all
```

## What About the Other Operations?

| Operation | Why no `job submit` needed |
|-----------|--------------------------|
| Embedding computation | Automatic via `upload_file`/`upload_dir` → `PROCESS_OBJECTS` |
| Model download | Dedicated `umrs-client model download` command |
| Re-embed with new model | `umrs-client recompute` / `umrs-client update` maintenance commands |

## Service Lifecycle

Controls what happens to Slurm/K8s jobs after upload-triggered processing completes.

| Mode | Behavior | Use case |
|------|----------|----------|
| `auto_release` (default) | `scancel` after last thread finishes | One-off processing, save node hours |
| `keep_alive` | Keep running until TTL or cancel | Successive/frequent batch processing |

```bash
# Server-wide (env var):
export UMRS_SERVICE_LIFECYCLE=auto_release   # default
export UMRS_SERVICE_LIFECYCLE=keep_alive     # for frequent processing

# Per-job (endpoint_provision only):
umrs-client job submit endpoint_provision --spec '...' --keep-alive
```

## Job Lifecycle

```
QUEUED → RUNNING → COMPLETED
                ↘ FAILED (retry → RUNNING)
                ↘ CANCELLED
```

Jobs are stored in `~/.config/umrs/jobs.json` (lightweight mode).
For production: `jobs/store.py` provides PostgreSQL-backed persistence
with advisory locks, crash reconciliation, and multi-instance support.

## Run Demo

```bash
./run_demo.sh
```
