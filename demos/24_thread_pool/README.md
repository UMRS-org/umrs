# Demo 24: Thread Pool & Backpressure

Demonstrates the UMRS bounded thread pool and its backpressure mechanism.

## What You'll Learn

- How to query pool utilisation metrics via `status all`
- What the pool metric fields mean (workers, pending, saturation)
- How backpressure protects the server from overload

## Quick Start

```bash
# Basic pool inspection:
bash run_demo.sh

# With a test task submission:
DEMO_SUBMIT_TASK=1 bash run_demo.sh
```

## Prerequisites

1. `umrs-client` installed and in PATH
2. UMRS server running and reachable

## Steps

| Step | Description |
|------|-------------|
| 1 | Query pool metrics from `status all` |
| 2 | Validate all metric fields are present |
| 3 | Check saturation (backpressure) indicator |
| 4 | (Optional) Submit a task and re-query |

## Pool Metrics

The `status all` response includes a `pool` object:

```json
{
  "pool": {
    "max_workers": 4,
    "max_pending": 64,
    "active_count": 1,
    "pending_count": 0,
    "total_queued": 1,
    "is_saturated": false
  }
}
```

| Field | Description |
|-------|-------------|
| `max_workers` | Maximum concurrent threads |
| `max_pending` | Maximum tasks waiting in queue |
| `active_count` | Currently executing threads |
| `pending_count` | Tasks waiting for a free thread |
| `total_queued` | `active_count + pending_count` |
| `is_saturated` | `true` when `pending_count >= max_pending` |

### Backpressure

When `is_saturated` is `true`, the server rejects new tasks with a
`ServerOverloadedError`. This prevents unbounded memory growth and
guarantees existing tasks complete in bounded time.

Clients should check `is_saturated` before submitting bulk work and
implement retry-with-backoff when the pool is full.
