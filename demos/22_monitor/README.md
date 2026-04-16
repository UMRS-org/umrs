# Demo 22: Jobs Monitor (TUI Dashboard)

Demonstrates the **live-refreshing terminal dashboard** for monitoring
UMRS jobs. Uses the `rich` library to render a table of job statuses
with optional filtering and interactive controls.

## Prerequisites

- UMRS server running and reachable (SSH tunnels if remote: ports 5555, 5556)
- `umrs-client` installed and in PATH
- `rich` library installed: `pip install 'rich>=13.0'`

## Quick Start

```bash
cd demos/22_monitor
bash run_demo.sh
```

## What This Demo Does

1. **Pre-flight** — verifies server connectivity and `rich` library
2. **Seed sample jobs** — submits `index_build` and `cleanup` jobs so the
   dashboard has data to display
3. **One-shot dashboard** — prints a single snapshot of all jobs
4. **Filtered view** — prints only jobs with `queued` status
5. **Interactive usage guide** — explains how to use the live dashboard

## Running Manually

```bash
# 1. Verify server connectivity
umrs-client ping

# 2. Verify the rich library is installed
#    The monitor requires rich>=13.0 for the TUI rendering.
python3 -c "import rich; print(rich.__version__)"

# 3. Submit some sample jobs (so the dashboard has data)
#    These are optional — the dashboard works with any existing jobs.
umrs-client job submit index_build \
    --spec '{"database": "demo_db", "table": "embeddings"}' \
    --project demo

umrs-client job submit cleanup \
    --spec '{"target": "stale_objects"}' \
    --project demo

# 4. Print a one-shot dashboard snapshot
#    The --once flag renders the table once and exits immediately
#    (no live refresh loop). Shows all jobs with their current status.
umrs-client monitor --once

# 5. Print filtered view — only queued jobs
#    The --status flag filters the display to show only jobs
#    matching the specified status.
umrs-client monitor --once --status queued

# 6. Launch the live-refreshing dashboard (interactive)
#    Without --once, the monitor enters a live loop that refreshes
#    the table every 3 seconds (configurable with --interval).
umrs-client monitor

# 7. Other useful monitor options:
#    Refresh every 5 seconds:
umrs-client monitor --interval 5

#    Filter by project:
umrs-client monitor --project myproject

#    Filter by status:
umrs-client monitor --status running
```

## Dashboard Keyboard Shortcuts (Live Mode)

| Key | Action |
|-----|--------|
| `q` | Quit the dashboard |
| `r` | Force refresh |
| `j` / `↓` | Move selection down |
| `k` / `↑` | Move selection up |
| `Enter` | Toggle job detail view (shows attempts) |
| `f` | Cycle status filter |
| `/` | Toggle compact / wide table |

## Monitor Command Reference

```
umrs-client monitor [OPTIONS]

Options:
  --once              Print once and exit (no live loop)
  --interval SECONDS  Refresh interval (default: 3)
  --status STATUS     Filter by job status (queued, running, completed, failed, cancelled)
  --project NAME      Filter by project name
```

## Expected Output (One-Shot)

```
┌─────────────────────────────────────────────────────────────┐
│                     UMRS Job Monitor                         │
├──────────┬──────────────┬──────────┬─────────┬──────────────┤
│ Job ID   │ Type         │ Status   │ Project │ Submitted    │
├──────────┼──────────────┼──────────┼─────────┼──────────────┤
│ abc123.. │ index_build  │ running  │ demo    │ 2 min ago    │
│ def456.. │ cleanup      │ queued   │ demo    │ 1 min ago    │
└──────────┴──────────────┴──────────┴─────────┴──────────────┘
```

## See Also

- [Demo 20: Durable Jobs](../20_durable_jobs/README.md) — job submit, cancel, idempotency
- [Demo 11: Job Submission](../11_job_submission/README.md) — job types and warm-up
- [Demo 06: Server Maintenance](../06_server_maintenance/README.md) — status monitoring
