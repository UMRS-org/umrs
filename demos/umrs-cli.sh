#!/bin/bash
# UMRS Client CLI Wrapper
# Convenience script to run UMRS client commands against the demo environment
#
# Usage:
#   ./umrs-cli.sh ping
#   ./umrs-cli.sh list_buckets
#   ./umrs-cli.sh status all
#   ./umrs-cli.sh upload_file /path/to/file bucket_alias

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="$SCRIPT_DIR/umrs-full"

# Check if Docker Compose environment is running
if ! docker compose -f "$COMPOSE_DIR/docker-compose.yaml" ps --quiet umrs-server &>/dev/null; then
    echo "Error: UMRS environment not running."
    echo "Start it with: cd $COMPOSE_DIR && docker compose up -d"
    exit 1
fi

# Run the client command
docker compose -f "$COMPOSE_DIR/docker-compose.yaml" exec -T umrs-client \
    python -m client_umrs "$@"
