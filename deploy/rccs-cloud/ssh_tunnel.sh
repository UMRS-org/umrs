#!/usr/bin/env bash
# SSH tunnel from local Mac to rccs-cloud for UMRS services.
#
# Usage:
#   ./ssh_tunnel_rccs.sh          # foreground (Ctrl-C to stop)
#   ./ssh_tunnel_rccs.sh &        # background
#
# Tunneled ports:
#   5555  ZMQ control plane (server ↔ client)
#   5556  ZMQ data plane    (1 user = 1 data port)
#   9000  MinIO S3 API      (client uploads/downloads)
#   8081  RAG service        (vector search & Q&A)
#
# Add more -L lines if you need additional data ports (5557, 5558, ...)
# for concurrent users.

set -euo pipefail

REMOTE_HOST="${1:-rccs-cloud}"

echo "Starting SSH tunnels to ${REMOTE_HOST} ..."
echo "  5555 → ZMQ control"
echo "  5556 → ZMQ data"
echo "  9000 → MinIO S3"
echo "  8081 → RAG service"
echo ""
echo "Press Ctrl-C to stop."

exec ssh -N \
    -L 5555:localhost:5555 \
    -L 5556:localhost:5556 \
    -L 9000:localhost:9000 \
    -L 8081:localhost:8081 \
    "${REMOTE_HOST}"
