#!/usr/bin/env bash
# =============================================================================
# Stop UMRS services on umrs-front
# =============================================================================
# Usage:
#   ./stop.sh              # Stop UMRS server containers + tunnels (infra stays)
#   ./stop.sh --server     # Stop only UMRS server containers (infra + tunnels stay)
#   ./stop.sh --all        # Stop everything (UMRS + infra containers + tunnels)
#   ./stop.sh --clean      # Stop everything + remove containers and volumes
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ─── Ensure rootless Podman socket is available ──────────────────────────────
if [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
    export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi

COMPOSE_FILE="$(pwd)/docker-compose.yaml"

# ─── Stop UMRS server containers ────────────────────────────────────────────
echo "Stopping UMRS server containers..."
if [[ -f "$COMPOSE_FILE" ]]; then
    podman compose -f "$COMPOSE_FILE" stop umrs-server-rccs umrs-server-ai4s umrs-server-bdrt umrs-server-octopus 2>/dev/null || true
    podman compose -f "$COMPOSE_FILE" rm -f umrs-server-rccs umrs-server-ai4s umrs-server-bdrt umrs-server-octopus 2>/dev/null || true
else
    # Fallback: direct container stop (compose file missing)
    podman stop umrs-server-rccs umrs-server-ai4s umrs-server-bdrt umrs-server-octopus 2>/dev/null || true
    podman rm -f umrs-server-rccs umrs-server-ai4s umrs-server-bdrt umrs-server-octopus 2>/dev/null || true
fi

# Also clean up any legacy native PID-managed servers from before containerization
for pidfile in server-*.pid; do
    [[ -f "$pidfile" ]] || continue
    pid=$(cat "$pidfile")
    name="${pidfile%.pid}"
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Stopping legacy $name (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
done

# Stop RAG service container
echo "  Stopping RAG container..."
podman stop umrs-rag 2>/dev/null || true
podman rm -f umrs-rag 2>/dev/null || true

# Stop legacy native RAG (PID-managed, from before containerization)
if [[ -f "rag-service.pid" ]]; then
    pid=$(cat "rag-service.pid")
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Stopping legacy native rag-service (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "rag-service.pid"
fi

if [[ "${1:-}" == "--server" ]]; then
    echo "Server-only stop — infrastructure containers and tunnels left running."
    echo "Done."
    exit 0
fi

# ─── Stop MinIO reverse tunnels ─────────────────────────────────────────────
echo "Stopping MinIO reverse tunnels..."
for pidfile in *-minio-reverse-tunnel.pid; do
    [[ -f "$pidfile" ]] || continue
    pid=$(cat "$pidfile")
    name="${pidfile%.pid}"
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Stopping $name (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
done

# ─── Stop infrastructure ────────────────────────────────────────────────────
case "${1:-}" in
    --all)
        echo "Stopping infrastructure containers..."
        if [[ -f "$COMPOSE_FILE" ]]; then
            podman compose -f "$COMPOSE_FILE" down 2>/dev/null || true
        else
            podman stop umrs-postgres umrs-minio 2>/dev/null || true
        fi
        ;;
    --clean)
        echo "Stopping all containers and removing volumes..."
        # Refuse to destroy data that lives outside the podman graphroot.
        # With UMRS_VOLUME_ROOT the database and object store are bind-backed
        # on a separate filesystem (see start.sh). `--clean` exists to reset a
        # scratch stack, not to wipe relocated production data, and podman's
        # own `volume rm` gives no second chance.
        for _vol in umrs_pg_data umrs_minio_data; do
            _opts="$(podman volume inspect "$_vol" --format '{{.Options}}' 2>/dev/null || true)"
            if [[ "$_opts" == *device:* ]]; then
                echo "ERROR: volume '$_vol' is bind-backed outside the podman store:" >&2
                echo "         $_opts" >&2
                echo "       Refusing --clean. Use --all to stop the stack, or remove the" >&2
                echo "       volume and its backing directory explicitly if that is intended." >&2
                exit 1
            fi
        done
        if [[ -f "$COMPOSE_FILE" ]]; then
            podman compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true
        else
            podman stop umrs-postgres umrs-minio 2>/dev/null || true
            podman rm -f umrs-postgres umrs-minio 2>/dev/null || true
            podman volume rm umrs_pg_data umrs_minio_data 2>/dev/null || true
        fi
        ;;
    "")
        echo "Infrastructure containers left running (use --all to stop them too)"
        ;;
esac

echo "Done."
