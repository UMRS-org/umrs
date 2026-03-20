# Demo 09 — Job Tracking via ZeroMQ

This demo shows how to submit, monitor, and manage durable jobs through
the UMRS ZeroMQ protocol.

## Prerequisites

- UMRS server running (see `demos/00_start_services/`)
- `umrs-client` available

## Commands

### Submit a job

```bash
umrs-client job submit embedding_batch \
    --spec '{"bucket": "my-bucket", "model": "e5-base-v2"}' \
    --project default --max-attempts 3
```

### Cancel a job

```bash
umrs-client job cancel job-abc123def456
```

### Query running jobs

Use the existing `status` command — it already covers active task state:

```bash
umrs-client status all
umrs-client status process_objects
```

## Job Types

| Type | Description |
|------|-------------|
| `index_build` | Build RAG index from dataset |
| `embedding_batch` | Batch embedding computation |
| `pipeline_run` | Execute ingestion pipeline |
| `endpoint_provision` | Provision Mini-RiVault endpoint |
| `model_download` | Download model weights |
| `cleanup` | Delete resources |

## Job Lifecycle

```
PLANNED → QUEUED → RUNNING → COMPLETED
                          ↘ FAILED (retry → RUNNING)
                          ↘ CANCELLED
```

Each job tracks multiple **attempts**, with automatic retry on failure.

## How It Works

- Client sends `JOB_SUBMIT` and `JOB_CANCEL` over ZeroMQ
- Server handles via `job_tracking_handler.py`
- Jobs are stored in `~/.config/umrs/jobs.json` (lightweight mode)
- For production: `jobs/store.py` provides PostgreSQL-backed persistence
  with advisory locks, crash reconciliation, and multi-instance support
- To **query** running jobs, use `umrs-client status all` (reads from the
  in-memory UMRSThreads store — adding a separate JOB_STATUS would duplicate it)

## Run Demo

```bash
./run_demo.sh
```
