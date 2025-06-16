# Docker Setup Guide

This directory contains Docker-related files for setting up the environment for the Infinia Demo project.

> **Note:** The `docker` directory is located at the root of the project (not inside `Infinia_demo`). All Docker-related commands and file references below assume you are using the root-level `docker` directory.

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed

## Quick Start

1. **Build and start all services:**
   ```bash
   docker-compose up --build
   ```
   This will build the images (if needed) and start the containers as defined in `docker/docker-compose.yml`.

2. **Stop all services:**
   ```bash
   docker-compose down
   ```

## Quick Demo

After starting the services, you can run a simple demo command to add a text entry:

```bash
docker exec -it umrs-server python src/client_umrs.py "I eat apple at breakfast" ej-bucket-2
```

This will insert a sample text into the database using the running container.

## Mounting a Directory or File for Processing

To process a local file or directory (e.g., images or text files) with the demo command, you need to mount it into the container using the `-v` (volume) option with Docker. For example:

```bash
docker run --rm \
  -v /path/to/local/data:/data \
  -v $(pwd)/s3cfg:/root/.s3cfg:ro \
  --network=host \
  -it umrs-server python src/client_umrs.py /data/apple.txt ej-bucket-2
```

- Replace `/path/to/local/data` with the absolute path to your local directory or file.
- `/data` is the path inside the container where your files will be accessible.
- The `-v $(pwd)/s3cfg:/root/.s3cfg:ro` option mounts your local `s3cfg` file into the container at `/root/.s3cfg` (read-only), just like in docker-compose.
- You can then reference `/data/yourfile.txt` in your command inside the container.


## Files
- `docker/docker-compose.yml`: Main Docker Compose configuration.
- `docker/Dockerfile`: Main application Dockerfile.
- `docker/Dockerfile.pgvector`: Dockerfile for PostgreSQL with pgvector extension.
- `docker/Dockerfile.redsql`: Dockerfile for RedSQL (if used).
- `docker/init_pgvector.sql`: SQL script to initialize pgvector extension.
- `docker/conf_umrs.json`: Example configuration file.
- `docker/s3cfg`: S3 configuration file for MinIO or AWS S3.

## Notes
- You may need to adjust environment variables in the `docker/docker-compose.yml` or configuration files to match your setup.
- For more details, see the main project documentation or contact the project maintainer.
