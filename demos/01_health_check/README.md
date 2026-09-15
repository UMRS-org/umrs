# Demo 01: Health Check

Quick connectivity check for the UMRS server. Works with any deployment
(Docker Compose, rccs-cloud, hpbd-atom, etc.).

## Prerequisites

- UMRS server running (any deployment)
- Selected connection profile (`umrs-client --output text profile show`)
- Network connectivity (SSH tunnels if the selected profile uses tunnel mode)
- `umrs-client` installed and in PATH
- User config at `~/.config/umrs/users/user1.json`

## Quick Start

```bash
cd demos/01_health_check
# Follow the step-by-step commands in README.md
```

## What This Demo Does

1. **Ping** the UMRS server via `umrs-client` to verify real client connectivity
2. **Supplemental endpoint checks** — inspects MinIO (9000) and reports raw ZMQ port visibility for debugging
3. **RAG service** — checks if the optional RAG endpoint (8081) is available
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
# Inspect the effective connection first
umrs-client --output text profile show

# Ping the UMRS server
umrs-client ping

# Optional supplemental checks
curl -sf http://localhost:9000/minio/health/live && echo "MinIO OK"
nc -z localhost 5555 && echo "ZMQ port visible"

# List configured buckets
umrs-client list_buckets
```

## Expected Output

```
[health_check] Server is ready.
[health_check]   ✓ MinIO S3     (localhost:9000)
[health_check]   ✓ RAG service  (http://127.0.0.1:8081)
```

If checks fail, verify that:
- The UMRS server is running
- `umrs-client ping` succeeds through the selected profile
- SSH tunnels are active (if remote): ports 5555, 5556, 9000, 8081
- The selected connection profile matches the deployment you are targeting
