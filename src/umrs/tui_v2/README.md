# UMRS Monitor v2 — Operational TUI

Complete operational client for UMRS. Replaces all CLI commands used in the
HMAE meeting demo and everyday operations.

## Launch

```bash
umrs-client monitor-v2 [--profile PROFILE] [--conf CONFIG_PATH]
```

## Screens

| Tab | Key | Description |
|-----|-----|-------------|
| 1 Dashboard | `1` | Operations home — quick actions, server health, recent jobs |
| 2 Jobs | `2` | Job list with color-coded status |
| 3 Workflows | `3` | Workflow list with contextual actions |
| 4 Status | `4` | Server health and service status |
| 5 Models | `5` | Model registry |
| 6 Connection | `6` | ZMQ/HTTP connection details |
| 7 Logs | `7` | Server log output |
| 8 Errors | `8` | Error log output |
| 9 WF Logs | `9` | Workflow execution logs |
| 0 Data | `0` | Upload/download history |
| a Submit | `a` | Job submission form |
| b Config | `b` | Config editor |

## Operations

### Upload Data
Press `u` from any tab, or click **Upload Data** on the dashboard.

1. Select source directory or file
2. Choose target bucket from configured aliases
3. Optionally set max image limit for demo speed
4. Preview files
5. Click **Upload** — progress bar shows live progress

### Run Experience
Click **Run Experience** on the dashboard.

1. Select experience (Reconstruction, MAE, Feature Extraction, etc.)
2. Choose input bucket
3. Optionally add override parameters
4. Click **Run** — shows progress until terminal status

### Register Workflow
Click **Register Workflow** on the dashboard.

1. Select workflow directory (must contain `workflow.json` + `steps/`)
2. Set workflow name
3. Validation runs automatically
4. Click **Upload** — uploads and registers the bundle

### Run Workflow
Click **Run Workflow** on the dashboard, or select a workflow in tab 3 and press
`Enter` → **Run**.

1. Select workflow from list
2. Choose bucket
3. Optionally add parameters
4. Click **Run** — shows step-level progress

### Download Results
Click **Download Results** on the dashboard, or select an execution in tab 3 and
press `d`.

1. Choose source: bucket or execution
2. Select destination directory
3. Optionally allow overwrite
4. Click **Download** — per-file and overall progress

### Browse Results
Click **Browse Results** on the dashboard.

1. Navigate result directory tree
2. Type badges show file types (PNG, PKL, JSON, PT, NPY)
3. Context menu: preview, open directory, copy path

### Demo Mode
Click **Demo Mode** on the dashboard.

Runs the HMAE pipeline step-by-step with visual progress:
1. Connect → 2. Upload → 3. Reconstruction → 4. MAE → 5. Features → 6.
   Download → 7. Browse → 8. Complete

## Keybindings

| Key | Action |
|-----|--------|
| `1`–`b` | Switch tabs |
| `r` | Refresh active tab |
| `Ctrl+S` | Save preferences |
| `Ctrl+E` | Export tab data |
| `slash` | Focus filter |
| `?` | Show help |
| `Enter` | Show detail / run contextual action |
| `q` | Quit |

## Contextual Actions

Actions appear on row selection in table panes:

- **Jobs**: `Enter` = detail, `x` = cancel
- **Workflows**: `Enter` = detail, `r` = run, `c` = cancel execution
- **Endpoints**: `Enter` = detail, `s` = stop
