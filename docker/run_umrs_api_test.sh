#!/bin/bash
# Build and run the UMRS API test container
set -e
cd "$(dirname "$0")"
DOCKER_BUILDKIT=1 docker build -f Dockerfile.umrs-api-test -t umrs-api-test .
docker run --rm --network host umrs-api-test
