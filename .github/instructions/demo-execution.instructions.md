---
description: "CRITICAL: How to run UMRS demos and interact with the UMRS system. Load for ANY task involving demos, testing, validation, or client operations."
applyTo: "demos/**,demos_private/**,deploy/**,src/client_umrs.py,src/umrs/client/**,src/umrs/shared/connection_profiles.py"
---

# UMRS Demo Execution & Client Architecture

## CRITICAL: The Client Runs Locally

UMRS uses a **local-client / remote-server** split architecture.

```
┌──────────── LOCAL MACHINE (Mac) ─────────────┐    ┌────────── REMOTE (rccs-cloud) ──────────┐
│                                               │    │                                         │
│  umrs-client (Python CLI in local .venv)      │    │  server_umrs.py (native on login node)  │
│    ├─ ZMQ control → localhost:5555 ───────────┼──→ │    └─ ZMQ :5555 / :5556                │
│    └─ S3 (boto3)  → localhost:9000 ───────────┼──→ │  podman: postgres :5432, minio :9000   │
│                                               │    │  Slurm → compute nodes: BentoML, vLLM  │
│  SSH tunnel (ssh_tunnel.sh running in bg)     │    │                                         │
│    -L 5555:localhost:5555                      │    │                                         │
│    -L 5556:localhost:5556                      │    │                                         │
│    -L 9000:localhost:9000                      │    │                                         │
│    -L 8081:localhost:8081                      │    │                                         │
└───────────────────────────────────────────────┘    └─────────────────────────────────────────┘
```

**The client NEVER runs on rccs-cloud.** Demos are bash scripts executed
on the local machine. They call `umrs-client` (installed locally) which
talks to the remote server through SSH-tunnelled ports on `127.0.0.1`.

## NEVER Do This

- ❌ `ssh rccs-cloud 'bash demos/01_health_check/run_demo.sh'`
- ❌ `ssh rccs-cloud 'source .venv/bin/activate && bash demos/run_all.sh'`
- ❌ Running any demo script via SSH on the remote machine
- ❌ Using the remote venv (`deploy/rccs-cloud/.venv`) to run client code

## ALWAYS Do This

- ✅ Run demos **locally**: `bash demos/01_health_check/run_demo.sh`
- ✅ Run `demos/run_all.sh` **locally**: `bash demos/run_all.sh`
- ✅ Use the **local** venv (`.venv/` at repo root) with `umrs-client` in PATH
- ✅ Ensure SSH tunnel is running in a separate terminal first

## How to Run Demos (Step by Step)

### 1. Prerequisites (one-time setup)

```bash
# From repo root
source .venv/bin/activate            # local venv with umrs-client
which umrs-client                    # must print a path in .venv

# Verify connection profile
umrs-client --output text profile show
# Should show: mode=ssh_tunnel, server_host=127.0.0.1, tunnel_mode=true
```

### 2. Start SSH Tunnel (separate terminal)

```bash
cd deploy/rccs-cloud
./ssh_tunnel.sh                      # blocks; Ctrl-C to stop
```

This forwards four ports:

| Local Port | Remote Service | Purpose |
|------------|---------------|---------|
| 5555 | ZMQ control | Server commands (ping, upload, status, etc.) |
| 5556 | ZMQ data | File transfer channel |
| 9000 | MinIO S3 | Object storage (images, text files, results) |
| 8081 | RAG service | Vector search & Q&A |

### 3. Run Demos Locally

```bash
# Single demo
bash demos/01_health_check/run_demo.sh

# All public demos
bash demos/run_all.sh

# All public demos, skip vision (no GPU needed)
bash demos/run_all.sh --skip-vision

# Private demos (one at a time)
bash demos_private/04_hmae_inference/run_demo.sh
```

### 4. Environment Variables (optional overrides)

| Variable | Purpose | Default |
|----------|---------|---------|
| `UMRS_PROFILE` | Connection profile name | contents of `~/.config/umrs/current_connection_profile` |
| `UMRS_USER_CONF` | User config file | `~/.config/umrs/users/user1.json` |
| `UMRS_ZMQ_HOST` | ZMQ server address | From profile (usually `127.0.0.1`) |
| `UMRS_ZMQ_PORT` | ZMQ control port | From profile (usually `5555`) |
| `FORCE_RECOMPUTE` | Force recompute in vision demos | unset |
| `MAX_IMAGES` | Limit images in vision demos | 3 |

## Local Configuration Layout

All configuration lives on the **local machine** at `~/.config/umrs/`:

```
~/.config/umrs/
├── connections/
│   └── rccs-cloud.json              # connection profile (host, ports, tunnel mode)
├── current_connection_profile       # text file containing "rccs-cloud"
├── users/
│   └── user1.json                   # user config (username, bucket aliases, experiences)
├── s3cfg                            # MinIO S3 credentials
├── conf_umrs.json                   # admin/server config (NOT for client demos)
└── workflow_catalog.json            # optional workflow catalog
```

**Important:** `conf_umrs.json` is the ADMIN/SERVER config. It has NO
`username` field. Demos must use `users/user1.json` (which has `username`,
bucket aliases, and experience definitions). The config is resolved by
`demos/lib/common.sh` in this order:

1. `$UMRS_CONF` env var (if set)
2. `$UMRS_USER_CONF` env var (if set)
3. `~/.config/umrs/users/user1.json` (if exists)
4. `~/.config/umrs/conf_umrs.json` (fallback — will fail for demo upload operations!)

## Connection Profile System

The client discovers how to reach the server via a **connection profile**:

```json
{
  "name": "rccs-cloud",
  "mode": "ssh_tunnel",
  "zmq": {
    "host": "127.0.0.1",
    "control_port": 5555,
    "data_base_port": 5556,
    "tunnel_mode": true
  },
  "services": {
    "minio_url": "http://127.0.0.1:9000",
    "rag_url": "http://127.0.0.1:8081"
  },
  "ssh": {
    "host": "rccs-cloud",
    "command_hint": "cd deploy/rccs-cloud && ./ssh_tunnel.sh"
  }
}
```

When `tunnel_mode: true`, ALL services are addressed via `127.0.0.1`.
The SSH tunnel makes the remote services appear local.

Profile management:
```bash
umrs-client profile list             # show available profiles
umrs-client profile use rccs-cloud   # select active profile
umrs-client profile show             # show resolved connection params
umrs-client profile import <file>    # import a new profile
```

## How common.sh Works

Every demo sources `demos/lib/common.sh`, which:

1. Calls `umrs-client profile show` to auto-discover connection params
2. Exports `UMRS_ZMQ_HOST`, `UMRS_ZMQ_PORT`, `UMRS_MINIO_HOST`, etc.
3. Resolves the user config (`$CONF`)
4. Builds the `$CLIENT` command string (e.g., `umrs-client --profile rccs-cloud -c ~/.config/umrs/users/user1.json`)
5. Provides helpers: `preflight()`, `wait_tasks()`, `banner()`, `log()`, `ok()`, `warn()`, `die()`

`preflight()` verifies:
- `umrs-client` is in PATH
- `$CLIENT ping` succeeds (proves ZMQ connectivity through tunnel)

## Data Flow for a Typical Demo

Example: SAM3 Segmentation (Demo 03)

1. User runs `bash demos/03_sam3_segmentation/run_demo.sh` **locally**
2. Script calls `$CLIENT upload_dir ./test_data_full/ user1-sam3`
3. `umrs-client` uploads images via boto3 → S3 (MinIO) through tunnel at `localhost:9000`
4. `umrs-client` sends ZMQ control message to server at `localhost:5555`
5. Server (on rccs-cloud) detects new objects, dispatches SAM3 via `sbatch` to GPU node
6. Compute node runs BentoML model, writes results back to MinIO
7. Script calls `$CLIENT download_results user1-sam3 ./results/`
8. Results downloaded to **local** `./results/` directory

## What Runs Where

| Component | Where | How |
|-----------|-------|-----|
| `umrs-client` | Local Mac | `pip install -e src/` in local `.venv` |
| `demos/*.sh` | Local Mac | `bash demos/XX/run_demo.sh` |
| `demos/lib/common.sh` | Local Mac | Sourced by every demo |
| SSH tunnel | Local Mac | `deploy/rccs-cloud/ssh_tunnel.sh` in bg terminal |
| `server_umrs.py` | rccs-cloud login node | Native Python process (NOT in container) |
| PostgreSQL | rccs-cloud login node | Podman container |
| MinIO | rccs-cloud login node | Podman container |
| BentoML / vLLM | rccs-cloud compute node | Slurm job + podman-compose on GPU node |

## SSH-Only Operations (Server Admin)

These operations DO use SSH because they manage SERVER-SIDE infrastructure:

- `ssh rccs-cloud 'cd deploy/rccs-cloud && ./start.sh'` — start server
- `ssh rccs-cloud 'cd deploy/rccs-cloud && ./stop.sh'` — stop server
- `ssh rccs-cloud 'squeue -u $(whoami)'` — check Slurm jobs
- `ssh rccs-cloud 'cd deploy/rccs-cloud && bash validate_harness.sh ...'` — harness validation
- `sbatch deploy/rccs-cloud/serve-model.slurm` — launch model on compute node

## Private Demos

Private demos in `demos_private/` follow the same local execution model:

| Demo | Requires | Test data source |
|------|----------|-----------------|
| 04 HMAE Inference | Vision service on Slurm | `demos_private/03_sam3_segmentation/test_data_full` or custom |
| 09 Defect Analysis | SAM3 + HMAE services | Same as above |
| 10 HMAE Full Pipeline | HMAE service | Same as above |
| 11 SAM3 Jens Direct | VisionFrontend at `:3000` | `demos/03_sam3_segmentation/test_data_full` |
| 12 HMAE Jens Direct | VisionFrontend at `:3000` | Same as above |
| 14 Opossum Analysis | Opossum engine | `tmp/opossum/data/` |

Demos 11, 12 bypass UMRS and call BentoML directly — they still run **locally**
but need VisionFrontend accessible (typically via SSH tunnel or direct network).

## Two Venvs — Don't Confuse Them

| Venv | Location | Purpose | Contains |
|------|----------|---------|----------|
| **Local** | `.venv/` (repo root) | Client operations, demos | `umrs-client`, lightweight deps |
| **Remote** | `deploy/rccs-cloud/.venv/` | Server on rccs-cloud | `server_umrs.py`, torch, transformers, etc. |

When running demos, always use the **local** venv. The remote venv is only
for server-side operations (starting/stopping the server, harness validation).

## Setup Reference

One-time setup for a new machine:

```bash
# 1. Create local venv
cd /path/to/umrs
uv venv .venv
source .venv/bin/activate

# 2. Install client + setup configs
cd deploy/rccs-cloud
./setup_client.sh                # copies configs to ~/.config/umrs/, installs umrs package

# 3. Verify
umrs-client --output text profile show
# Should show: mode=ssh_tunnel, tunnel_mode=true

# 4. Start tunnel (separate terminal)
./ssh_tunnel.sh

# 5. Test connectivity
umrs-client ping
# Should print: server_ready=true

# 6. Run first demo
cd ../../demos/01_health_check
bash run_demo.sh
```
