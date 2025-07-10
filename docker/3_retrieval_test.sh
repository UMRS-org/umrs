#!/bin/bash
set -e

# test data directory
DATA_DIR="${PWD}/../data"
if [ ! -d "$DATA_DIR" ]; then
  echo "Data directory does not exist: $DATA_DIR"
  exit 1
fi

DOCKER_IMAGE="docker-umrs-server"


echo "=== Running retrieval tests ==="

# First, list all databases
echo "Listing all databases..."
docker run --rm -it \
  -v "$PWD:/workdir" \
  --network="docker_default" \
  $DOCKER_IMAGE \
  python src/retrieve_embeddings.py --host postgres --port 5432 --list-databases

# List tables for user1
echo "Listing tables for user1..."
docker run --rm -it \
  -v "$PWD:/workdir" \
  --network="docker_default" \
  $DOCKER_IMAGE \
  python src/retrieve_embeddings.py --host postgres --port 5432 --user user1 --experiment images_and_texts --list-tables

# List tables for user2
echo "Listing tables for user2..."
docker run --rm -it \
  -v "$PWD:/workdir" \
  --network="docker_default" \
  $DOCKER_IMAGE \
  python src/retrieve_embeddings.py --host postgres --port 5432 --user user2 --experiment images_and_texts --list-tables

# Run retrieval for user1
echo "Retrieving data for user1..."
docker run --rm -it \
  -v "$PWD:/workdir" \
  --network="docker_default" \
  $DOCKER_IMAGE \
  python src/retrieve_embeddings.py --host postgres --port 5432 --user user1 --experiment images_and_texts --table image --limit 5

# Run retrieval for user2
echo "Retrieving data for user2..."
docker run --rm -it \
  -v "$PWD:/workdir" \
  --network="docker_default" \
  $DOCKER_IMAGE \
  python src/retrieve_embeddings.py --host postgres --port 5432 --user user2 --experiment images_and_texts --table text --limit 5

echo "Retrieval tests completed successfully."
