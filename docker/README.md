# Docker Setup Guide

This directory contains the local standalone Docker deployment for UMRS.
It is intended for local development, API/server smoke tests, and lightweight
single-node experimentation. It does **not** replicate the full Slurm-backed
Mini-RiVault flow used on HPC sites.

All commands below assume you are running them from the repository root.

## What the Docker stack provides

- `umrs-server` container running `src/server_umrs.py`
- PostgreSQL + pgvector
- MinIO for S3-compatible object storage

The compose file does not provision Slurm, Mini-RiVault, or compute-node GPU
launches.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- Docker Compose / `docker compose`

## Quick Start

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker logs -f umrs-server
```

To stop the stack:

```bash
docker compose -f docker/docker-compose.yml down
```

## Smoke Checks

The image runs the server directly and includes the source tree under `/app`.
The simplest validation commands are:

```bash
# ZMQ control-plane ping from inside the container
docker exec -it umrs-server python src/client_umrs.py ping

# List buckets from the bundled config
docker exec -it umrs-server \
   python src/client_umrs.py --conf /root/conf_umrs.json list_buckets user1

# HTTP API docs from the host
curl http://localhost:8001/docs
```

The Docker image does not currently install the editable package entrypoint, so
the canonical in-container CLI invocation is still `python src/client_umrs.py`.

## Ports

| Host port | Service |
|---|---|
| `8001` | UMRS HTTP API (`/docs`) |
| `5555` | ZMQ control plane |
| `9000` | MinIO S3 API |
| `9001` | MinIO console |

## Files In This Directory

| File | Purpose |
|---|---|
| `docker-compose.yml` | local standalone stack |
| `Dockerfile` | UMRS server image |
| `Dockerfile.umrs-api-test` | API test image |
| `conf_umrs.json` | bundled server config used by the container |
| `user_files/` | bundled user configs |
| `init_pgvector.sql` | pgvector initialization |
| `s3cfg`, `s3cfg_user1`, `s3cfg_user2` | MinIO/S3 credentials |
| `env.default` | example environment values |
| `data/` | sample data baked into the image |

## Notes And Limitations

- This stack is for local development; it is not the same runtime path as the
   site-specific Slurm+Podman deployments under `deploy/`.
- The Docker image starts the server with `python src/server_umrs.py -c /root/conf_umrs.json`.
- For site deployments, use the matching guide under `../deploy/` (for example
   [../deploy/rccs-cloud/README.md](../deploy/rccs-cloud/README.md)) or the
   admin guide instead of copying Docker commands.

## See Also

- [../Readme.md](../Readme.md)
- [../doc/admin_guide.md](../doc/admin_guide.md)
- [../doc/umrs_overview.md](../doc/umrs_overview.md)
