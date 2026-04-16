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
- `umrs-client` available

## Why 3 Job Types?

### 1. `endpoint_provision` — Pre-warm a service before uploads

Normally the server launches a Slurm/K8s job **lazily** on the first file
upload, which adds cold-start latency. If you know you'll process many
batches, pre-warm the endpoint and cancel it explicitly when you are done:

```bash
# Pre-warm Mini-RiVault/BentoML endpoint (the site-configured backend starts immediately):
umrs-client job submit endpoint_provision \
    --spec '{"model": "sam3"}' \
    --project my-project

# Now uploads use the warm endpoint — no cold start:
umrs-client upload_dir /data/images/batch-1/
umrs-client upload_dir /data/images/batch-2/

# When done, release the node hours:
umrs-client job cancel <job_id>
```

In current durable deployments, the UMRS job record completes once the warm
endpoint is ready. The backing endpoint then remains available for reuse until
you cancel it or janitor/TTL cleanup removes it.

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

Force-clean endpoints explicitly. The janitor handles TTL-based cleanup
automatically, but sometimes you want immediate teardown:

```bash
# Teardown a specific endpoint:
umrs-client job submit cleanup \
    --spec '{"target": "endpoint", "job_id": "job-abc123"}' \
    --project my-project

# Additional cleanup targets may be added later, but the durable backend
# currently supports only endpoint teardown.
```

### Cancel a job

```bash
umrs-client job cancel job-abc123def456
```

### Query running jobs

Use the monitor dashboard for the durable job list, or `status all` for the
broader server summary:

```bash
umrs-client monitor --once
umrs-client status all
```

## What About the Other Operations?

| Operation | Why no `job submit` needed |
|-----------|--------------------------|
| Embedding computation | Automatic via `upload_file`/`upload_dir` → `PROCESS_OBJECTS` |
| Model download | Dedicated `umrs-client model download` command |
| Re-embed with new model | `umrs-client recompute` / `umrs-client update` maintenance commands |

## Lifecycle Notes

Two lifecycle mechanisms exist today:

| Path | What controls cleanup | Current status |
|------|-----------------------|----------------|
| Upload-triggered processing | `UMRS_SERVICE_LIFECYCLE=auto_release|keep_alive` in `server_umrs.py` | Legacy server-side path |
| Durable `endpoint_provision` | Explicit `job cancel`, registry TTL/janitor cleanup, and warm reuse | Validated path on current site deployments |

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
umrs-client ping

# 2. Pre-warm a compute endpoint (avoid cold-start on first upload)
#    The endpoint stays warm until you cancel it or TTL expires.
#    Use this before batch upload sessions.
umrs-client job submit endpoint_provision \
    --spec '{"model": "sam3"}' \
    --project my-project

# 3. Upload data (uses the warm endpoint — no cold start)
umrs-client upload_dir /data/images/batch-1/ user1-sam3
umrs-client upload_dir /data/images/batch-2/ user1-sam3

# 4. When done processing, release the endpoint
#    Replace <job_id> with the ID returned in step 2.
umrs-client job cancel <job_id>

# 5. Rebuild RAG index after bulk inserts
#    After large ingestions, the pgvector IVFFlat index benefits from
#    re-clustering for optimal search performance.
umrs-client job submit index_build \
    --spec '{"database": "xp_spring8", "table": "image_embeddings"}' \
    --project spring8

# 6. Clean up stale resources
#    Force-remove endpoints, S3 objects, or database tables immediately
#    (instead of waiting for TTL-based janitor cleanup).
umrs-client job submit cleanup \
    --spec '{"target": "endpoint", "job_id": "job-abc123"}' \
    --project my-project

# 7. Monitor job status
umrs-client monitor --once
umrs-client status all
```

## Run Demo

```bash
./run_demo.sh
```
