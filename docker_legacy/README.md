# UMRS Legacy Docker Test Setup

This setup runs MinIO and Postgres as separate containers, and runs the UMRS server and client together in a single container without using the HTTP API. The client communicates with the server via the ZeroMQ control plane (legacy mode).

## What it includes
- `docker-compose.yml` — Compose file defining `minio`, `postgres`, and a single `umrs-legacy` container that runs both the UMRS server and client tests.
- `Dockerfile.legacy` — Image for `umrs-legacy` containing both server and client code.
- `test_umrs_legacy.sh` — Entry script that starts the server in the background, waits for readiness, then runs client commands (text/image uploads, batch uploads, and embedding operations) via ZeroMQ only.
   Uploads are executed using the client's legacy_mode (positional arguments: "data" and "bucket"), not subcommands.

## How to run
1. Build and start services:
   - docker compose -f docker_legacy/docker-compose.yml up --build

2. Stop services:
   - docker compose -f docker_legacy/docker-compose.yml down

## Notes
- S3 credentials (`s3cfg`) and user files are mounted read-only from the main `docker/` directory to keep consistency with the primary test setup.
- No HTTP ports are published for the `umrs-legacy` service since tests do not use the HTTP API.
- The ZeroMQ control plane runs inside the container and is not required to be exposed; you can add a port mapping for 5555 if you need external inspection.