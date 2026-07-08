#!/usr/bin/env bash
# =============================================================================
# start.sh — bring up the frontend → multi-HPC stack (TEMPLATE)
# =============================================================================
# 1. Start infra (postgres, minio) and wait for readiness.
# 2. Create the per-cluster Postgres DBs + pgvector extension.
# 3. Start the per-cluster UMRS servers.
#
#   ./start.sh            # everything
#   ./start.sh --server   # restart UMRS servers only (keep infra)
#
# Minimal reference — the production ../../umrs-front/start.sh adds CURVE key
# generation, S3 bucket bootstrap, and richer readiness checks.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"
export GOMAXPROCS="${GOMAXPROCS:-4}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

SCRIPT_DIR="$(pwd)"
COMPOSE="podman compose -f ${SCRIPT_DIR}/docker-compose.yaml"
START_MODE="${1:-}"

[[ "$(uname -s)" == "Linux" ]] || { echo "ERROR: run on the frontend host (Linux)."; exit 1; }
[[ -f .env ]] || { echo "ERROR: .env not found — copy env.example and fill it in."; exit 1; }
set -o allexport; source .env; set +o allexport

_pg() {
    podman exec umrs-postgres psql -U "${POSTGRES_USER:-umrs}" "$@" 2>/dev/null || \
    PGPASSWORD="${POSTGRES_PASSWORD}" psql -h 127.0.0.1 -p "${POSTGRES_PORT:-5432}" \
        -U "${POSTGRES_USER:-umrs}" "$@" 2>/dev/null
}

if [[ "$START_MODE" != "--server" ]]; then
    echo "Starting infra containers (postgres, minio)..."
    $COMPOSE up -d postgres minio

    echo -n "  Waiting for postgres..."
    for i in $(seq 1 30); do
        if bash -c "echo >/dev/tcp/127.0.0.1/${POSTGRES_PORT:-5432}" 2>/dev/null; then
            echo " ready."; break
        fi
        echo -n "."; sleep 1
        [[ "$i" -eq 30 ]] && echo " WARNING: postgres not ready after 30s"
    done

    # Per-cluster databases + pgvector. Add a line per cluster you deploy.
    for db in "${POSTGRES_DB_CLUSTER_A:-front_cluster_a}" "${POSTGRES_DB_CLUSTER_B:-front_cluster_b}"; do
        _pg -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='${db}'" | grep -q 1 || \
            _pg -d postgres -c "CREATE DATABASE ${db};" || true
        _pg -d "$db" -c "CREATE EXTENSION IF NOT EXISTS vector;" || true
    done

    echo -n "  Waiting for minio..."
    for i in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:${MINIO_PORT:-9000}/minio/health/live" >/dev/null 2>&1; then
            echo " ready."; break
        fi
        echo -n "."; sleep 1
        [[ "$i" -eq 30 ]] && echo " WARNING: minio not ready after 30s"
    done
fi

echo "Ensuring UMRS server image is built..."
podman image exists localhost/umrs-server:latest 2>/dev/null || \
    $COMPOSE build umrs-server-cluster-a

echo "Starting per-cluster UMRS servers..."
$COMPOSE up -d umrs-server-cluster-a umrs-server-cluster-b

echo "Done. Point the gateway (gateway.nginx.conf) at the per-cluster HTTP ports."
