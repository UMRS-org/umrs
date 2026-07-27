# Demo 01: Health Check

Quick connectivity check for the UMRS server. Works with any deployment
(Docker Compose, rccs-cloud, hpbd-atom, etc.).

## Prerequisites

- UMRS server running (any deployment)
- Selected connection profile (`umrs profile show`)
- Network connectivity (SSH tunnels if the selected profile uses tunnel mode)
- `umrs` installed and in PATH
- User config at `~/.config/umrs/users/user1.json`

## Quick Start

```bash
# Navigate from any level (auto-detects parent repo vs umrs subdirectory)
cd "$(git rev-parse --show-toplevel)/$(test -d "$(git rev-parse --show-toplevel)/umrs/demos" && echo umrs)/demos/01_health_check"
# See the numbered steps below in this README
```

## What This Demo Does

1. **Show connection info** — `umrs info` reveals your effective server, transport, and auth settings
2. **Ping** the UMRS server via `umrs` to verify real client connectivity
3. **Check experiences** — list loaded experiences for your user config
4. **List buckets** to confirm the user's S3 configuration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UMRS_ZMQ_HOST` | `localhost` | Host for ZMQ ports |
| `UMRS_ZMQ_PORT` | `5555` | ZMQ control port |
| `UMRS_ZMQ_DATA_PORT` | `5556` | ZMQ data port |
| `UMRS_MINIO_HOST` | `localhost` | MinIO host |
| `UMRS_MINIO_PORT` | `9000` | MinIO port |
| `RAG_URL` | profile-derived, default `http://127.0.0.1:8081` | RAG service URL |

## Running Manually

```bash
# Show effective connection, transport, auth, and S3 settings.
# This command reveals your actual server host, ports, and tunnel status —
# these values differ by deployment (local, rccs-cloud, etc.).
umrs info

# Ping the UMRS server
umrs ping

# List loaded experiences for your user config
umrs experience list

# List configured buckets
umrs bucket list
```

## Expected Output

`umrs info` shows your effective connection — host, ports, auth mode, and S3 status.
Values vary by deployment (local dev vs rccs-cloud). A healthy connection looks like:

```
  ● Server ready  |  profile=rccs-cloud  |  data0.ai.r-ccs.riken.jp:50090

  Transport:
    Control channel : ZMQ (tcp://data0.ai.r-ccs.riken.jp:50090)
    Data channel    : ZMQ (tcp://data0.ai.r-ccs.riken.jp:50091)
    HTTP gateway    : https://data0.ai.r-ccs.riken.jp/ai4s

  Authentication:
    Mode            : JWT

  S3 / MinIO:
    Endpoint        : http://data0.ai.r-ccs.riken.jp:9000
    Access key      : mnagaso
    Buckets         : 1 configured
```

If checks fail, verify that:
- The UMRS server is running
- `umrs ping` succeeds through the selected profile
- SSH tunnels are active (if remote)
- The selected connection profile matches the deployment you are targeting
