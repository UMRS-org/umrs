# Demo 01: Health Check

Quick connectivity check for the UMRS server. Works with any deployment
(Docker Compose, rccs-cloud, hpbd-atom, etc.).

## Prerequisites

- UMRS server running (any deployment)
- Network connectivity (SSH tunnels if remote)
- `umrs-client` installed and in PATH
- User config at `~/.config/umrs/users/user1.json`

## Quick Start

```bash
cd demos/01_health_check
bash run_demo.sh
```

## What This Demo Does

1. **Port connectivity** — checks ZMQ control (5555), ZMQ data (5556), MinIO (9000)
2. **Ping** the UMRS server via `umrs-client`
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
| `RAG_URL` | `http://localhost:8081` | RAG service URL |

## Running Manually

```bash
# Check port connectivity
nc -z localhost 5555 && echo "ZMQ OK"
curl -sf http://localhost:9000/minio/health/live && echo "MinIO OK"

# Ping the UMRS server
umrs-client -c ~/.config/umrs/users/user1.json ping

# List configured buckets
umrs-client -c ~/.config/umrs/users/user1.json list_buckets
```

## Expected Output

```
[health_check]   ✓ ZMQ control  (localhost:5555)
[health_check]   ✓ MinIO S3     (localhost:9000)
[health_check] Server is ready.
[health_check]   ✓ RAG service  (http://localhost:8081)
```

If checks fail, verify that:
- The UMRS server is running
- SSH tunnels are active (if remote): ports 5555, 5556, 9000, 8081
- Your config file has the correct server settings
