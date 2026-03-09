# Docker Setup Guide

This directory contains Docker-related files for deploying UMRS (Unified Multimodal Retrieval System) as a standalone stack.

> **Note:** All Docker-related commands and file references below assume you are at the repository root. Docker files live under the `docker/` directory.

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed

## Quick Start

1. **Build and start all services:**
   ```bash
   docker compose -f docker/docker-compose.yml up -d --build
   ```
   This builds the images and starts the containers: UMRS server, PostgreSQL+pgvector, and MinIO.

2. **Watch the server logs:**
   ```bash
   docker logs -f umrs-server
   ```

3. **Stop all services:**
   ```bash
   docker compose -f docker/docker-compose.yml down
   ```

## Quick Demo

After starting the services, test with the client:

```bash
# Ping the server
docker exec -it umrs-server python src/client_umrs.py ping

# List buckets
docker exec -it umrs-server python src/client_umrs.py --conf /root/conf_umrs.json list_buckets user1

# Upload inline text
docker exec -it umrs-server python src/client_umrs.py \
   --conf /root/conf_umrs.json \
   upload_text "I eat apple at breakfast" bucket-2 user1 --meta keywords:fruit
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| `umrs-server` | 5555 (ZMQ) | UMRS server with ZeroMQ control plane |
| `postgres` | 5432 | PostgreSQL with pgvector extension |
| `minio` | 9000 (API), 9001 (Console) | S3-compatible object storage |
| `umrs-api-test` | - | Optional: scripted API tests |

## Mounting Local Files

To process a local file or directory, mount it into the container:

```bash
docker run --rm \
  -v /path/to/local/data:/data \
  -v $(pwd)/docker/s3cfg:/root/.s3cfg:ro \
  --network=host \
  -it umrs-server python src/client_umrs.py /data/apple.txt ej-bucket-2
```

## Files

| File | Description |
|------|-------------|
| `docker-compose.yml` | Main Docker Compose configuration |
| `Dockerfile` | Main application Dockerfile |
| `init_pgvector.sql` | SQL script to initialize pgvector extension |
| `conf_umrs.json` | Example UMRS server configuration |
| `s3cfg` | S3 admin credentials for MinIO |
| `s3cfg_user1`, `s3cfg_user2` | Per-user S3 credentials |
| `env.default` | Default environment variables |
| `user_files/` | Per-user JSON configuration files |

## Notes
- Environment variables are configured in `docker-compose.yml` and `env.default`
- The server mounts `s3cfg` as `~/.s3cfg` inside the container for admin S3 access
- User config files from `user_files/` are copied into the image at build time
- For detailed architecture and implementation, see `doc/umrs_overview.md`
