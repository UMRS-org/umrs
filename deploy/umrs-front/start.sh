#!/usr/bin/env bash
# =============================================================================
# start.sh — Launch dual-cluster UMRS on umrs-front
# =============================================================================
# Starts:
#   - Postgres (podman container, port 5432)
#   - MinIO (podman container, port 9000)
#   - UMRS server for RCCS Cloud (ZMQ:5555, HTTP:8000)
#   - UMRS server for AI4S (ZMQ:5557, HTTP:8002)
#
# Prerequisites:
#   - Run ./setup.sh once (creates venv, installs deps)
#   - Edit .env
#
# Usage:
#   ./start.sh              # Start everything
#   ./start.sh --server     # Restart UMRS servers only (keep containers)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"
export GOMAXPROCS="${GOMAXPROCS:-4}"

SCRIPT_DIR="$(pwd)"
REPO_ROOT="$(cd ../.. && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LOG_DIR="${SCRIPT_DIR}/logs"

START_MODE="${1:-}"

# ─── Prerequisites ───────────────────────────────────────────────────────────
[[ "$(uname -s)" == "Linux" ]] || { echo "ERROR: Must run on umrs-front (Linux)"; exit 1; }
[[ -d "$VENV_DIR" ]] || { echo "ERROR: Venv not found. Run ./setup.sh first"; exit 1; }

# ─── Load .env ───────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Create from template in setup.sh"
    exit 1
fi
set -o allexport
source .env
set +o allexport

MINIO_BIND="${MINIO_BIND:-0.0.0.0}"
if [[ "$MINIO_BIND" == "0.0.0.0" || "$MINIO_BIND" == "::" ]]; then
    echo "WARNING: MinIO API will bind to ${MINIO_BIND}:${MINIO_PORT:-9000}; ensure host/firewall access is restricted to trusted cluster paths."
fi

mkdir -p "$LOG_DIR"

# ─── Start infrastructure containers ─────────────────────────────────────────
if [[ "$START_MODE" != "--server" ]]; then
    echo "Starting infrastructure containers..."

    # Postgres — skip if native postgres is already listening
    if ss -tlnp 2>/dev/null | grep -q ":${POSTGRES_PORT:-5432}" ; then
        echo "  Postgres already listening on port ${POSTGRES_PORT:-5432} (native/external)."
    elif ! podman inspect umrs-postgres >/dev/null 2>&1; then
        echo "  Creating postgres container..."
        podman run -d --name umrs-postgres \
            -e POSTGRES_USER="${POSTGRES_USER:-umrs}" \
            -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD}" \
            -e POSTGRES_DB="${POSTGRES_DB_RCCS:-front_rccs}" \
            -p 127.0.0.1:${POSTGRES_PORT:-5432}:5432 \
            -v umrs_pg_data:/var/lib/postgresql/data \
            docker.io/pgvector/pgvector:pg15 \
            postgres -c "unix_socket_directories="
    elif [[ "$(podman inspect -f '{{.State.Status}}' umrs-postgres)" != "running" ]]; then
        echo "  Starting existing postgres container..."
        podman start umrs-postgres
    else
        echo "  Postgres already running."
    fi

    # MinIO
    if ! podman inspect umrs-minio >/dev/null 2>&1; then
        echo "  Creating minio container..."
        podman run -d --name umrs-minio \
            -e MINIO_ROOT_USER="${MINIO_ROOT_USER:?Set MINIO_ROOT_USER}" \
            -e MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:?Set MINIO_ROOT_PASSWORD}" \
            -p "${MINIO_BIND}:${MINIO_PORT:-9000}:9000" \
            -p 127.0.0.1:${MINIO_CONSOLE_PORT:-9001}:9001 \
            -v umrs_minio_data:/data \
            docker.io/minio/minio:latest \
            server /data --console-address ":9001"
    elif [[ "$(podman inspect -f '{{.State.Status}}' umrs-minio)" != "running" ]]; then
        echo "  Starting existing minio container..."
        podman start umrs-minio
    else
        echo "  MinIO already running."
    fi

    # Wait for postgres
    echo -n "  Waiting for postgres..."
    for i in $(seq 1 30); do
        if pg_isready -h 127.0.0.1 -p "${POSTGRES_PORT:-5432}" -U "${POSTGRES_USER:-umrs}" >/dev/null 2>&1 || \
           podman exec umrs-postgres pg_isready -h localhost -U "${POSTGRES_USER:-umrs}" >/dev/null 2>&1; then
            echo " ready."
            break
        fi
        echo -n "."
        sleep 1
        [[ "$i" -eq 30 ]] && echo " WARNING: postgres not ready after 30s"
    done

    # Create AI4S database if it doesn't exist (works with native or container)
    _pg_cmd() { psql -h 127.0.0.1 -p "${POSTGRES_PORT:-5432}" -U "${POSTGRES_USER:-umrs}" "$@" 2>/dev/null || \
                podman exec umrs-postgres psql -U "${POSTGRES_USER:-umrs}" "$@" 2>/dev/null; }
    _pg_cmd -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='${POSTGRES_DB_AI4S:-umrs_ai4s}'" \
        | grep -q 1 || \
    _pg_cmd -d postgres -c "CREATE DATABASE ${POSTGRES_DB_AI4S:-umrs_ai4s};" || true

    # Ensure pgvector extension in both databases
    for db in "${POSTGRES_DB_RCCS:-front_rccs}" "${POSTGRES_DB_AI4S:-front_ai4s}"; do
        _pg_cmd -d "$db" -c "CREATE EXTENSION IF NOT EXISTS vector;" || true
    done

    # Wait for minio
    echo -n "  Waiting for minio..."
    for i in $(seq 1 30); do
        if curl -sf http://127.0.0.1:${MINIO_PORT:-9000}/minio/health/live >/dev/null 2>&1; then
            echo " ready."
            break
        fi
        echo -n "."
        sleep 1
        [[ "$i" -eq 30 ]] && echo " WARNING: minio not ready after 30s"
    done
fi

# ─── Activate venv ───────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export UMRS_AUTH_MODE="${UMRS_AUTH_MODE:-strict}"

# ─── Create s3cfg for admin ──────────────────────────────────────────────────
S3CFG="${SCRIPT_DIR}/s3cfg"
if [[ ! -f "$S3CFG" ]]; then
    cat > "$S3CFG" <<EOF
[default]
access_key = ${MINIO_ROOT_USER}
secret_key = ${MINIO_ROOT_PASSWORD}
host_base = 127.0.0.1:${MINIO_PORT:-9000}
host_bucket = 127.0.0.1:${MINIO_PORT:-9000}
bucket_location = us-east-1
use_https = False
signature_v2 = False
EOF
fi

# Ensure ~/.config/umrs/s3cfg exists (server reads from here)
mkdir -p "$HOME/.config/umrs"
cp -f "$S3CFG" "$HOME/.config/umrs/s3cfg"

# ─── Create conf_umrs.json ───────────────────────────────────────────────────
CONF_FILE="${SCRIPT_DIR}/conf_umrs.json"
if [[ ! -f "$CONF_FILE" ]]; then
    mkdir -p "${SCRIPT_DIR}/user_files"
    cat > "${SCRIPT_DIR}/user_files/user1.json" <<'EOF'
{
  "username": "user1",
  "fullname": "Default User",
  "s3_configuration_file": "~/.config/umrs/s3cfg_user1",
  "experiences": [
    {
      "xp_name": "Text Embedding",
      "sql_username": "usr1_text",
      "modalities": [
        {
          "modality_name": "text",
          "s3": {
            "bucket_alias": "user1-texts",
            "data": { "type": "inline", "mime_type": "text/plain" }
          },
          "model": {
            "model_name": "intfloat/e5-base-v2",
            "vector_size": "768",
            "model_type": "rivault://openai://http://localhost:{LOCAL_PORT}/v1",
            "return_type": null,
            "spacy_model": "en_core_web_sm"
          },
          "sql": {
            "store_embeddings": "yes",
            "table_name": "text",
            "meta_data": [
              { "name": "keywords", "type": "TEXT" }
            ]
          }
        }
      ]
    }
  ]
}
EOF
    cat > "$CONF_FILE" <<EOF
{
  "admin_s3_configuration_file": "$HOME/.config/umrs/s3cfg",
  "user_files_dir": "./user_files",
  "users": [
    "./user_files/user1.json"
  ]
}
EOF
fi

# ─── Stop existing servers ───────────────────────────────────────────────────
for pidfile in "${SCRIPT_DIR}"/server-*.pid; do
    [[ -f "$pidfile" ]] || continue
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping existing server (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
done

_stop_port_listener() {
    local port="$1"
    local label="$2"
    local listener_pids

    listener_pids=$(ss -H -ltnp 2>/dev/null \
        | awk -v port=":${port}" '$4 ~ port "$" { print $0 }' \
        | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
        | sort -u)

    for listener_pid in $listener_pids; do
        [[ "$listener_pid" == "$$" ]] && continue
        echo "Stopping stale ${label} listener on port ${port} (PID ${listener_pid})..."
        kill "$listener_pid" 2>/dev/null || true
    done
}

_stop_port_listener "${UMRS_RCCS_PORT:-8000}" "RCCS HTTP"
_stop_port_listener "5555" "RCCS ZMQ"
_stop_port_listener "${UMRS_AI4S_PORT:-8002}" "AI4S HTTP"
_stop_port_listener "5557" "AI4S ZMQ"
sleep 2
_stop_port_listener "${UMRS_RCCS_PORT:-8000}" "RCCS HTTP"
_stop_port_listener "5555" "RCCS ZMQ"
_stop_port_listener "${UMRS_AI4S_PORT:-8002}" "AI4S HTTP"
_stop_port_listener "5557" "AI4S ZMQ"

# ─── Start UMRS server for RCCS Cloud ────────────────────────────────────────
echo ""
echo "Starting UMRS server (RCCS Cloud)..."
echo "  Site config: ${SCRIPT_DIR}/../rccs-front/site.yaml"
echo "  ZMQ port: 5555"
echo "  HTTP port: ${UMRS_RCCS_PORT:-8000}"

export UMRS_SITE_CONFIG="${SCRIPT_DIR}/../rccs-front/site.yaml"
export UMRS_CONFIG="${SCRIPT_DIR}"
export UMRS_CONTROL_PORT=5555
export UMRS_HTTP_PORT="${UMRS_RCCS_PORT:-8000}"
export DB_HOST="127.0.0.1"
export DB_PORT="${POSTGRES_PORT:-5432}"
export DB_SUPERUSER="${POSTGRES_USER:-umrs}"
export DB_SUPERUSER_PASSWORD="${POSTGRES_PASSWORD}"
export UMRS_REGISTRY_POSTGRES_DSN="postgres://${POSTGRES_USER:-umrs}:${POSTGRES_PASSWORD}@127.0.0.1:${POSTGRES_PORT:-5432}/${POSTGRES_DB_RCCS:-front_rccs}"
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY}"
export UMRS_ADMIN_TOKEN="${UMRS_ADMIN_TOKEN:-}"
export UMRS_TUNNEL_MODE=true
export S3_PRESIGN_ENDPOINT="http://127.0.0.1:${MINIO_PORT:-9000}"

nohup python "${REPO_ROOT}/src/server_umrs.py" \
    -c "${CONF_FILE}" \
    >> "${LOG_DIR}/server-rccs.log" 2>&1 &
RCCS_PID=$!
echo "$RCCS_PID" > "${SCRIPT_DIR}/server-rccs.pid"
echo "  PID: $RCCS_PID"

echo -n "  Waiting for RCCS HTTP bootstrap..."
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${UMRS_RCCS_PORT:-8000}/health" >/dev/null 2>&1; then
        echo " ready."
        break
    fi
    if ! kill -0 "$RCCS_PID" 2>/dev/null; then
        echo " failed."
        echo "ERROR: RCCS server failed during bootstrap. Check ${LOG_DIR}/server-rccs.log"
        tail -20 "${LOG_DIR}/server-rccs.log" 2>/dev/null || true
        exit 1
    fi
    echo -n "."
    sleep 1
    [[ "$i" -eq 60 ]] && echo " WARNING: RCCS HTTP not ready after 60s"
done

# ─── Start UMRS server for AI4S ──────────────────────────────────────────────
echo ""
echo "Starting UMRS server (AI4S)..."
echo "  Site config: ${SCRIPT_DIR}/../ai4s/site.yaml"
echo "  ZMQ port: 5557"
echo "  HTTP port: ${UMRS_AI4S_PORT:-8002}"

export UMRS_SITE_CONFIG="${SCRIPT_DIR}/../ai4s/site.yaml"
export UMRS_CONTROL_PORT=5557
export UMRS_HTTP_PORT="${UMRS_AI4S_PORT:-8002}"
export UMRS_REGISTRY_POSTGRES_DSN="postgres://${POSTGRES_USER:-umrs}:${POSTGRES_PASSWORD}@127.0.0.1:${POSTGRES_PORT:-5432}/${POSTGRES_DB_AI4S:-front_ai4s}"

nohup python "${REPO_ROOT}/src/server_umrs.py" \
    -c "${CONF_FILE}" \
    >> "${LOG_DIR}/server-ai4s.log" 2>&1 &
AI4S_PID=$!
echo "$AI4S_PID" > "${SCRIPT_DIR}/server-ai4s.pid"
echo "  PID: $AI4S_PID"

# ─── Verify servers started ──────────────────────────────────────────────────
echo ""
echo -n "Verifying servers..."
sleep 3
FAIL=0
if ! kill -0 "$RCCS_PID" 2>/dev/null; then
    echo ""
    echo "ERROR: RCCS server failed to start. Check ${LOG_DIR}/server-rccs.log"
    tail -20 "${LOG_DIR}/server-rccs.log" 2>/dev/null || true
    FAIL=1
fi
if ! kill -0 "$AI4S_PID" 2>/dev/null; then
    echo ""
    echo "ERROR: AI4S server failed to start. Check ${LOG_DIR}/server-ai4s.log"
    tail -20 "${LOG_DIR}/server-ai4s.log" 2>/dev/null || true
    FAIL=1
fi

if [[ "$FAIL" -eq 0 ]]; then
    echo " both running."
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  UMRS Dual-Cluster Stack (umrs-front)                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Postgres:     127.0.0.1:${POSTGRES_PORT:-5432}                         ║"
echo "║  MinIO:        ${MINIO_BIND}:${MINIO_PORT:-9000}                            ║"
echo "║                                                              ║"
echo "║  RCCS Cloud:                                                 ║"
echo "║    UMRS HTTP:  0.0.0.0:${UMRS_RCCS_PORT:-8000}  (PID $RCCS_PID)           ║"
echo "║    Site:       rccs-front/site.yaml                          ║"
echo "║    Target:     rccs-cloud (SSH)                              ║"
echo "║                                                              ║"
echo "║  AI4S:                                                       ║"
echo "║    UMRS HTTP:  0.0.0.0:${UMRS_AI4S_PORT:-8002}  (PID $AI4S_PID)           ║"
echo "║    Site:       ai4s/site.yaml                                ║"
echo "║    Target:     134.160.189.24 (ea0022, SSH)                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Logs:"
echo "  RCCS:  tail -f ${LOG_DIR}/server-rccs.log"
echo "  AI4S:  tail -f ${LOG_DIR}/server-ai4s.log"
echo ""
echo "Stop:    ./stop.sh"
