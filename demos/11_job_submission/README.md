# Demo 11 — Job Submission & Endpoint Warm-Up

Explicit job submission for operations that are **not triggered by file uploads**.

The basic UMRS flow (upload file → compute embeddings) is fully automatic via
`PROCESS_OBJECTS` — no job submission needed. This demo covers 3 cases where
you do need an explicit `job submit`.

For durable site deployments, the validated endpoint lifecycle is explicit
submit, warm reuse while the endpoint remains registered, and explicit cancel
when you are done. The legacy `UMRS_SERVICE_LIFECYCLE` environment variable
still applies to upload-triggered processing in `server_umrs.py`, but it is
not the lifecycle control that has been validated for durable
`endpoint_provision`.

## Prerequisites

- UMRS server running and reachable
- `umrs` available

## Why 3 Job Types?

### 1. `endpoint_provision` — Pre-warm a service before uploads

Normally the server launches a Slurm/K8s job **lazily** on the first file
upload, which adds cold-start latency. If you know you'll process many
batches, pre-warm the endpoint and cancel it explicitly when you are done:

```bash
# Pre-warm Mini-RiVault/BentoML endpoint (the site-configured backend starts immediately):
umrs job submit endpoint_provision \
    --spec '{"model": "sam3"}' \
    --project my-project

# Now uploads use the warm endpoint — no cold start:
umrs data dir-upload /data/images/batch-1/
umrs data dir-upload /data/images/batch-2/

# When done, release the node hours:
umrs job cancel <job_id>
```

In current durable deployments, the UMRS job record completes once the warm
endpoint is ready. The backing endpoint then remains available for reuse until
you cancel it or janitor/TTL cleanup removes it. If you cancel while the
provisioning job is still active, that job transitions to `cancelled`. If you
cancel after the endpoint is already warm and the provision job is `completed`,
UMRS now submits the corresponding `cleanup` job for that endpoint and leaves
the original `endpoint_provision` record as `completed`.

### 2. `index_build` — Rebuild RAG index after bulk inserts

After embeddings accumulate in pgvector, the IVFFlat index may need
re-clustering for optimal search performance. This is a database
operation, not a file upload:

```bash
# Rebuild the RAG index after a large ingestion run.
# On the current demo sequence, Demo 02 populates this table:
umrs job submit index_build \
    --spec '{"database": "text_embedding_usr1_text", "table": "text_file"}' \
    --project my-project
```

### 3. `cleanup` — Teardown resources explicitly

Force-clean endpoints explicitly. The janitor handles TTL-based cleanup
automatically, but sometimes you want immediate teardown:

```bash
# Teardown a specific endpoint explicitly:
umrs job submit cleanup \
    --spec '{"target": "endpoint", "job_id": "<endpoint_provision_job_id>"}' \
    --project my-project

# Additional cleanup targets may be added later, but the durable backend
# currently supports only endpoint teardown.
```

### Cancel a job

```bash
# Running or queued jobs are cancelled directly.
# Completed endpoint_provision jobs are converted into cleanup jobs.
umrs job cancel job-abc123def456
```

### Query running jobs

Use the monitor dashboard for the durable job list, or `status all` for the
broader server summary:

```bash
umrs monitor --once --project my-project
umrs status all
```

## What About the Other Operations?

| Operation | Why no `job submit` needed |
|-----------|--------------------------|
| Embedding computation | Automatic via `upload_file`/`upload_dir` → `PROCESS_OBJECTS` |
| Model download | Dedicated `umrs model download` command |
| Re-embed with new model | `umrs recompute` / `umrs update` maintenance commands |

## Lifecycle Notes

Two lifecycle mechanisms exist today:

| Path | What controls cleanup | Current status |
|------|-----------------------|----------------|
| Upload-triggered processing | `UMRS_SERVICE_LIFECYCLE=auto_release|keep_alive` in `server_umrs.py` | Legacy server-side path |
| Durable `endpoint_provision` | Explicit `job cancel`, cleanup jobs, registry TTL/janitor cleanup, and warm reuse | Validated path on current site deployments |

For the validated durable path, prefer explicit `job cancel` when you are done
and use the site-local harness validator to prove reuse or cleanup behavior
after runtime changes:

```bash
cd deploy/<site>
./validate_harness.sh --skip-matrix --lifecycle-case full
```

## Job Lifecycle

```
QUEUED → RUNNING → COMPLETED
                ↘ FAILED (retry → RUNNING)
                ↘ CANCELLED
```

Legacy lightweight mode stored jobs in `~/.config/umrs/jobs.json`.
Current UMRS deployments use the durable PostgreSQL-backed store from
`jobs/store.py`, with advisory locks, crash reconciliation, and
multi-instance support.

## Running Manually

```bash
# 1. Verify server connectivity
umrs ping

# 2. Pre-warm a compute endpoint (avoid cold-start on first upload)
#    The endpoint stays warm until you cancel it or TTL expires.
#    Use this before batch upload sessions.
umrs job submit endpoint_provision \
    --spec '{"model": "sam3"}' \
    --project my-project

# 3. Inspect the durable job list for this project
umrs monitor --once --project my-project

# 4. Rebuild the pgvector index after bulk inserts
#    Demo 02 populates text_embedding_usr1_text.text_file.
umrs job submit index_build \
    --spec '{"database": "text_embedding_usr1_text", "table": "text_file"}' \
    --project my-project

# 5. When done processing, release the endpoint
#    Replace <job_id> with the ID returned in step 2.
#    If the provision job is still active it becomes cancelled.
#    If it already completed, UMRS submits the matching cleanup job.
umrs job cancel <job_id>

# 6. Explicit cleanup remains available when you want to target a specific
#    endpoint job id directly, or when you want a no-op cleanup against an
#    already-gone endpoint.
umrs job submit cleanup \
    --spec '{"target": "endpoint", "job_id": "<endpoint_provision_job_id>"}' \
    --project my-project

# 7. Monitor job status
umrs monitor --once --project my-project
umrs status all
```

## Run Demo

```bash
# See the step-by-step commands in the numbered sections below
umrs ping
umrs job submit --help
```
