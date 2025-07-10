#!/bin/bash
set -e

# test data directory
DATA_DIR="${PWD}/../data"
if [ ! -d "$DATA_DIR" ]; then
  echo "Data directory does not exist: $DATA_DIR"
  exit 1
fi

DOCKER_IMAGE="docker-umrs-server"

echo "=== Testing Multi-User UMRS Setup ==="

# Check if docker compose services are running
echo "Checking if required services are running..."
if ! docker compose ps | grep -q "minio.*Up"; then
  echo "Starting minio service..."
  docker compose up -d minio
  sleep 5
fi

if ! docker compose ps | grep -q "postgres.*Up"; then
  echo "Starting postgres service..."
  docker compose up -d postgres
  sleep 10
fi

echo "Services are ready."

# Initialize buckets and database by running server once for each user config
echo "Initializing buckets and database for user1..."
docker run --rm \
  -v "$PWD:/workdir" \
  -v "$DATA_DIR:/data" \
  --network="docker_default" \
  $DOCKER_IMAGE \
  python src/server_umrs.py -c /workdir/conf_umrs_user1.json --init-only

echo "Initializing buckets and database for user2..."
docker run --rm \
  -v "$PWD:/workdir" \
  -v "$DATA_DIR:/data" \
  --network="docker_default" \
  $DOCKER_IMAGE \
  python src/server_umrs.py -c /workdir/conf_umrs_user2.json --init-only

