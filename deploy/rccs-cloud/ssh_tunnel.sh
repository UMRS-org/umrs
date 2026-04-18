#!/usr/bin/env bash
# SSH tunnel from local Mac to rccs-cloud for UMRS services.
#
# Usage:
#   ./ssh_tunnel_rccs.sh          # foreground (Ctrl-C to stop)
#   ./ssh_tunnel_rccs.sh &        # background
#
# Tunneled ports:
#   5555        ZMQ control plane (server ↔ client)
#   5556-5563   ZMQ data plane    (1 user = 1 data port)
#   9000        MinIO S3 API      (client uploads/downloads)
#   8081        RAG service       (vector search & Q&A)
#
# The server allocates one data port per user starting at UMRS_DATA_BASE_PORT,
# so the tunnel needs to forward a range, not just the first data port.

set -euo pipefail

REMOTE_HOST="${1:-rccs-cloud}"
CONTROL_PORT="${UMRS_CONTROL_PORT:-5555}"
DATA_BASE_PORT="${UMRS_DATA_BASE_PORT:-5556}"
DATA_PORT_COUNT="${UMRS_DATA_PORT_COUNT:-8}"
S3_PORT="${UMRS_S3_PORT:-9000}"
RAG_PORT="${UMRS_RAG_PORT:-8081}"

data_last_port=$((DATA_BASE_PORT + DATA_PORT_COUNT - 1))
ssh_args=(-N "${REMOTE_HOST}" -L "${CONTROL_PORT}:localhost:${CONTROL_PORT}" -L "${S3_PORT}:localhost:${S3_PORT}" -L "${RAG_PORT}:localhost:${RAG_PORT}")

for ((port = DATA_BASE_PORT; port <= data_last_port; port++)); do
    ssh_args+=( -L "${port}:localhost:${port}" )
done

echo "Starting SSH tunnels to ${REMOTE_HOST} ..."
echo "  ${CONTROL_PORT}        → ZMQ control"
echo "  ${DATA_BASE_PORT}-${data_last_port}   → ZMQ data"
echo "  ${S3_PORT}        → MinIO S3"
echo "  ${RAG_PORT}        → RAG service"
echo ""
echo "Press Ctrl-C to stop."

exec ssh "${ssh_args[@]}"
