# Demo 20: Durable Job Orchestration

Demonstrates the **durable job system**: submit jobs with idempotency,
poll status, test duplicate detection, and cancel a running durable endpoint job.

## Prerequisites

- UMRS server running and reachable (SSH tunnels if remote: ports 5555, 5556, 9000)
- UMRS server with the job subsystem enabled
- `umrs-client` installed and in PATH

## Quick Start

```bash
cd demos/20_durable_jobs
bash run_demo.sh
```

## What This Demo Does

1. **Pre-flight** — verifies server connectivity
2. **Submit a job** — sends an `index_build` job with a JSON spec
3. **Poll job status** — checks the job status 3 times with intervals
4. **Idempotent resubmit** — submits the same job spec again; the server
   should return the same `job_id` (deduplication)
5. **Submit + cancel** — submits an `endpoint_provision` job and immediately
    cancels it, demonstrating the cancellation flow

## Running Manually

```bash
# 1. Verify server connectivity
umrs-client ping

# 2. Submit an index_build job
#    The --spec flag passes a JSON payload with job parameters.
#    --project scopes the job to a project namespace.
#    On the current demo sequence, Demo 02 populated this table.
umrs-client job submit index_build \
    --spec '{"database": "text_embedding_usr1_text", "table": "text_file"}' \
    --project demo20
# Output includes a job_id, e.g.: {"job": {"job_id": "abc123..."}}

# 3. Check the job status
#    Poll until the job reaches "completed" or "failed".
#    Status values: queued → running → completed | failed | cancelled
umrs-client job status <job_id>

# 4. Test idempotency — resubmit the same job spec
#    If the server enforces idempotency for this job type, it returns
#    the same job_id instead of creating a duplicate job.
umrs-client job submit index_build \
    --spec '{"database": "text_embedding_usr1_text", "table": "text_file"}' \
    --project demo20
# Expected: same job_id as step 2

# 5. Submit a new endpoint_provision job and cancel it.
#    This is a better cancellation demo than index_build because it stays
#    active long enough to observe a real cancelled durable job state.
umrs-client job submit endpoint_provision \
    --spec '{"model": "sam3"}' \
    --project demo20-cancel
# Note the new job_id from the response

# 6. Cancel the job
#    Sends a cancellation request. The server stops the job and
#    transitions it to "cancelled" state.
umrs-client job cancel <job_id>

# 7. Verify the cancelled status
umrs-client job status <job_id>
# Expected: the job record reports status="cancelled"
```

## Job Lifecycle

```
QUEUED → RUNNING → COMPLETED
                ↘ FAILED (retry → RUNNING)
                ↘ CANCELLED
```

## Job Types

| Type | Purpose | When to use |
|------|---------|-------------|
| `index_build` | Rebuild or verify the pgvector HNSW index | After large bulk imports |
| `endpoint_provision` | Pre-warm a compute endpoint | Before batch uploads to avoid cold start |
| `cleanup` | Teardown resources | Clean stale endpoints, S3 objects, or tables |

## Environment Variables

No additional environment variables — uses the default `umrs-client` config.

## Expected Output

```
[durable_job]   ✓ Job submitted: job-1838df7fa827
[durable_job]     Poll 1: status=queued
[durable_job]     Poll 2: status=completed
[durable_job]   ✓ Status polling complete (final: completed)
[durable_job]   ✓ Idempotency confirmed — same job_id returned
[durable_job]   ✓ Job cancelled: job-a83a0d9a841b
[durable_job]     Cancelled job status: cancelled

Summary of durable job operations:
  ✓ Submitted index_build job
  ✓ Polled status through lifecycle
  ✓ Idempotent resubmit test completed
    ✓ endpoint_provision submit + cancel flow demonstrated
```

## See Also

- [Demo 11: Job Submission](../11_job_submission/README.md) — job types and endpoint warm-up
- [Demo 22: Monitor](../22_monitor/README.md) — TUI dashboard for job monitoring
- [Demo 06: Server Maintenance](../06_server_maintenance/README.md) — check/update/recompute
