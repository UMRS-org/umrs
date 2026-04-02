#!/usr/bin/env bash
# SSH tunnel from local Mac to rccs-cloud for UMRS services.
#
# Usage:
#   ./ssh_tunnel_rccs.sh          # foreground (Ctrl-C to stop)
#   ./ssh_tunnel_rccs.sh &        # background
#
# Tunneled ports:
#   5555   ZMQ control plane (server ↔ client)
#   15556  ZMQ data plane    (1 user = 1 data port)
#   9000   MinIO S3 API      (client uploads/downloads)
#   8081   RAG service        (vector search & Q&A)
#
# Add more -L lines if you need additional data ports (15557, 15558, ...)
# for concurrent users.
#
# NOTE: We use port 15556 (not 5556) for the data plane to avoid
# collisions with VS Code Remote SSH auto port forwarding.

set -euo pipefail

REMOTE_HOST="${1:-rccs-cloud}"

echo "Starting SSH tunnels to ${REMOTE_HOST} ..."
echo "  5555  → ZMQ control"
echo "  15556 → ZMQ data"
echo "  9000  → MinIO S3"
echo "  8081  → RAG service"
echo ""
echo "Press Ctrl-C to stop."

exec ssh -N \
    -L 5555:localhost:5555 \
    -L 15556:localhost:15556 \
    -L 9000:localhost:9000 \
    -L 8081:localhost:8081 \
    "${REMOTE_HOST}"
