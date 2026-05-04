# Demo 22: UMRS Monitor (Multi-Tab TUI Dashboard)

Multi-tab **live-refreshing terminal dashboard** for monitoring the
entire UMRS system. Uses the `rich` library to render tabbed views
for jobs, server status, models, workflows, and connection health.

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
2. **Seed sample jobs** — submits a sample `index_build` job so the
  Jobs tab has data to display
3. **Jobs tab** — one-shot snapshot of all jobs
4. **Filtered Jobs view** — only `queued` jobs
5. **Status tab** — server task summaries (recompute, update, check, etc.)
6. **Models tab** — registered model catalog
7. **Workflows tab** — registered workflow registry
8. **Connection tab** — connection profile details and server health ping
9. **Interactive usage guide** — explains all tabs, keys, and options

## Tabs

| Tab | Key | Content |
|-----|-----|---------|
| **Jobs** | `1` | Durable job queue with status, attempts, detail expansion |
| **Status** | `2` | Server task summaries with pending/completed counts |
| **Models** | `3` | Registered model catalog with detail view |
| **Workflows** | `4` | Registered workflows with step details |
| **Connection** | `5` | Active connection profile, server health, ping latency |

## Running Manually

```bash
# 1. Verify server connectivity
umrs-client ping

# 2. Verify the rich library is installed
python3 -c "import importlib.metadata; print(importlib.metadata.version('rich'))"

# 3. Submit a sample job (optional — dashboard works with any existing jobs)
umrs-client job submit index_build \
  --spec '{"database": "text_embedding_usr1_text", "table": "text_file"}' \
  --project demo22

# 4. One-shot snapshot — Jobs tab (default)
umrs-client monitor --once

# 5. One-shot snapshot — specific tabs
umrs-client monitor --once --tab status
umrs-client monitor --once --tab models
umrs-client monitor --once --tab workflows
umrs-client monitor --once --tab connection

# 6. Filtered jobs view
#    This may legitimately show zero rows if nothing is currently queued.
umrs-client monitor --once --status queued

# 7. Launch the live-refreshing interactive dashboard
umrs-client monitor

# 8. Start on a specific tab
umrs-client monitor --tab connection

# 9. Mount a config file (enables in-TUI reload via R key)
umrs-client monitor --conf ~/.config/umrs/users/user1.json

# 10. Other options
umrs-client monitor --interval 5         # refresh every 5 seconds
umrs-client monitor --project myproject  # filter jobs by project
```

## Keyboard Shortcuts (Live Mode)

### Global

| Key | Action |
|-----|--------|
| `1`-`5` | Switch tab |
| `q` / `Ctrl+C` | Quit |
| `r` | Force refresh |
| `?` | Toggle help overlay |

### Jobs Tab

| Key | Action |
|-----|--------|
| `j` / `↓` | Move selection down |
| `k` / `↑` | Move selection up |
| `Enter` | Toggle job detail view (shows attempts + spec) |
| `f` | Cycle status filter (all → running → failed → completed → ...) |
| `/` | Toggle compact / wide table |
| `c` | Cancel selected job |

### Status Tab

| Key | Action |
|-----|--------|
| `R` | Trigger server reload (operator/admin action; requires `--conf` / `-c`) |

### Models / Workflows Tabs

| Key | Action |
|-----|--------|
| `j` / `↓` | Move selection down |
| `k` / `↑` | Move selection up |
| `Enter` | Toggle detail panel |

## Monitor Command Reference

```
umrs-client monitor [OPTIONS]

Options:
  --once                   Print once and exit (no live loop)
  --interval SECONDS       Refresh interval (default: 3)
  --tab TAB                Start tab: jobs, status, models, workflows, connection
  --status STATUS          Filter jobs by status (Jobs tab)
  --project NAME           Filter jobs by project (Jobs tab)
  -c / --conf PATH         Mount config file (enables in-TUI reload)
```

## Config File Mounting

When you pass `-c` / `--conf` pointing to a user config JSON file:
- The **Status tab** shows the config filename and enables the `R` key
- Pressing `R` uploads the config to the server and triggers a RELOAD
- This is equivalent to running `umrs-client reload` but from
  within the dashboard

Treat the `R` key as an operator/admin workflow. Normal end-user monitoring does
not require reload access.

## Expected Output (One-Shot)

```
┌─────────────────────────────────────────────────────────────┐
│                     UMRS Job Monitor                         │
├──────────┬──────────────┬──────────┬─────────┬──────────────┤
│ Job ID   │ Type         │ Status   │ Project │ Submitted    │
├──────────┼──────────────┼──────────┼─────────┼──────────────┤
│ abc123.. │ index_build  │ running  │ demo    │ 2 min ago    │
│ def456.. │ endpoint_pr… │ queued   │ demo22  │ 1 min ago    │
└──────────┴──────────────┴──────────┴─────────┴──────────────┘
```

## See Also

- [Demo 20: Durable Jobs](../20_durable_jobs/README.md) — job submit, cancel, idempotency
- [Demo 11: Job Submission](../11_job_submission/README.md) — job types and warm-up
- [Demo 06: Server Maintenance](../06_server_maintenance/README.md) — status monitoring
