#!/usr/bin/env bash
# =============================================================================
# start.sh — Launch dual-cluster UMRS on umrs-front
# =============================================================================
# Starts:
#   - Postgres (podman container, port 5432)
#   - MinIO (podman container, port 9000)
#   - UMRS server for RCCS Cloud (ZMQ:50090, HTTP:8000)
#   - UMRS server for AI4S (ZMQ:50100, HTTP:8002)
#
# Prerequisites:
#   - Podman (rootless) with compose support
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
LOG_DIR="${SCRIPT_DIR}/logs"

# Container image used for both UMRS servers and utility tasks (key gen, S3 bootstrap, etc.)
UMRS_IMAGE="localhost/umrs-server:latest"

START_MODE="${1:-}"

# ─── Prerequisites ───────────────────────────────────────────────────────────
[[ "$(uname -s)" == "Linux" ]] || { echo "ERROR: Must run on umrs-front (Linux)"; exit 1; }

# ─── Ensure rootless Podman socket is available ──────────────────────────────
# Non-interactive SSH sessions (Ansible) don't set XDG_RUNTIME_DIR and the
# Podman user socket may not be running.  podman compose delegates to the
# docker-compose plugin which requires the API socket.
if [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
    export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi
mkdir -p "$XDG_RUNTIME_DIR" 2>/dev/null || true
if ! podman info --format '{{.Host.RemoteSocket.Path}}' >/dev/null 2>&1; then
    echo "Starting rootless Podman socket..."
    systemctl --user start podman.socket 2>/dev/null || true
fi

# ─── Load .env ───────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Create from template in setup.sh"
    exit 1
fi
ENV_UMRS_SERVER_HOST="${UMRS_SERVER_HOST:-}"
ENV_S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-}"
ENV_S3_BUCKET="${S3_BUCKET:-}"
set -o allexport
source .env
set +o allexport
[[ -n "$ENV_UMRS_SERVER_HOST" ]] && UMRS_SERVER_HOST="$ENV_UMRS_SERVER_HOST"
[[ -n "$ENV_S3_ENDPOINT_URL" ]] && S3_ENDPOINT_URL="$ENV_S3_ENDPOINT_URL"
[[ -n "$ENV_S3_BUCKET" ]] && S3_BUCKET="$ENV_S3_BUCKET"

MINIO_BIND="${MINIO_BIND:-0.0.0.0}"
if [[ "$MINIO_BIND" == "0.0.0.0" || "$MINIO_BIND" == "::" ]]; then
    echo "WARNING: MinIO API will bind to ${MINIO_BIND}:${MINIO_PORT:-9000}; ensure host/firewall access is restricted to trusted cluster paths."
fi

mkdir -p "$LOG_DIR"

UMRS_RCCS_ZMQ_PORT="${UMRS_RCCS_ZMQ_PORT:-50090}"
UMRS_RCCS_DATA_PORT_RANGE="${UMRS_RCCS_DATA_PORT_RANGE:-50091-50098}"
UMRS_AI4S_ZMQ_PORT="${UMRS_AI4S_ZMQ_PORT:-50100}"
UMRS_AI4S_DATA_PORT_RANGE="${UMRS_AI4S_DATA_PORT_RANGE:-50101-50108}"
UMRS_ZMQ_AUTH_TYPE="${UMRS_ZMQ_AUTH_TYPE:-none}"

# ─── Start infrastructure containers ─────────────────────────────────────────
if [[ "$START_MODE" != "--server" ]]; then
    echo "Starting infrastructure containers via podman compose..."

    # Check for port conflicts before starting
    _pg_port="${POSTGRES_PORT:-5432}"
    if ss -tlnp 2>/dev/null | grep -q ":${_pg_port}\b"; then
        if ! podman inspect umrs-postgres >/dev/null 2>&1 || \
           [[ "$(podman inspect -f '{{.State.Status}}' umrs-postgres 2>/dev/null)" != "running" ]]; then
            echo "  WARNING: Port ${_pg_port} is occupied by another process (native postgres?)."
            echo "           Attempting to use it — if UMRS fails to connect, stop the native"
            echo "           postgres and re-run:  sudo systemctl stop postgresql"
        fi
    fi

    # Ensure named volumes exist (compose uses external: true)
    podman volume inspect umrs_pg_data >/dev/null 2>&1 || podman volume create umrs_pg_data
    podman volume inspect umrs_minio_data >/dev/null 2>&1 || podman volume create umrs_minio_data

    # Remove legacy containers that conflict with compose-managed names
    for _ctr in umrs-postgres umrs-minio; do
        if podman inspect "$_ctr" >/dev/null 2>&1; then
            echo "  Removing legacy container $_ctr..."
            podman stop "$_ctr" 2>/dev/null || true
            podman rm -f "$_ctr" 2>/dev/null || true
        fi
    done

    podman compose -f "${SCRIPT_DIR}/docker-compose.yaml" up -d postgres minio

    # Wait for postgres
    echo -n "  Waiting for postgres..."
    for i in $(seq 1 30); do
        if pg_isready -h 127.0.0.1 -p "${POSTGRES_PORT:-5432}" -U "${POSTGRES_USER:-umrs}" >/dev/null 2>&1 || \
           podman exec umrs-postgres pg_isready -h localhost -U "${POSTGRES_USER:-umrs}" >/dev/null 2>&1 || \
           bash -c "echo >/dev/tcp/127.0.0.1/${POSTGRES_PORT:-5432}" 2>/dev/null; then
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

# ─── Build UMRS server image (needed for utility tasks and servers) ──────────
echo "Ensuring UMRS server image is available..."
if [[ "${UMRS_IMAGE_REBUILD:-false}" == "true" ]] || \
   ! podman image exists "$UMRS_IMAGE" 2>/dev/null; then
    echo "  Building UMRS server image..."
    podman compose -f "${SCRIPT_DIR}/docker-compose.yaml" build umrs-server-rccs
fi

# Helper: run a one-shot Python snippet inside the UMRS server image.
# Mounts current dir at /work and uses host networking for DB/S3 access.
_run_util() {
    podman run --rm --network=host \
        -v "${SCRIPT_DIR}:/work:z" \
        -v "${HOME}/.config/umrs:/root/.config/umrs:z" \
        -w /work \
        -e S3_ENDPOINT_URL -e S3_ACCESS_KEY -e S3_SECRET_KEY \
        -e S3_REGION -e S3_BUCKET -e S3_USE_SSL \
        -e MINIO_ROOT_USER -e MINIO_ROOT_PASSWORD \
        -e MINIO_PORT -e POSTGRES_USER -e POSTGRES_PASSWORD \
        -e POSTGRES_PORT -e POSTGRES_DB_RCCS -e POSTGRES_DB_AI4S \
        "$UMRS_IMAGE" python -c "$1"
}

# ─── Set env vars for utility tasks ──────────────────────────────────────────
export PYTHONUNBUFFERED=1
export UMRS_AUTH_MODE="${UMRS_AUTH_MODE:-strict}"
export S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-http://127.0.0.1:${MINIO_PORT:-9000}}"
export S3_ACCESS_KEY="${S3_ACCESS_KEY:-${MINIO_ROOT_USER}}"
export S3_SECRET_KEY="${S3_SECRET_KEY:-${MINIO_ROOT_PASSWORD}}"
export S3_REGION="${S3_REGION:-us-east-1}"
export S3_BUCKET="${S3_BUCKET:-r4-data}"
export S3_USE_SSL="${S3_USE_SSL:-false}"

# Generate persisted server CURVE keys when ZMQ CURVE/session auth is enabled.
# Session mode uses these server keys plus an in-memory ZAP credential provider;
# it does not use a static client key directory.
if [[ "$UMRS_ZMQ_AUTH_TYPE" == "curve" || "$UMRS_ZMQ_AUTH_TYPE" == "session" ]]; then
    ZMQ_CURVE_SERVER_KEYS_DIR="${UMRS_ZMQ_CURVE_SERVER_KEYS_DIR:-${HOME}/.config/umrs/curve-keys/server}"
    mkdir -p "$ZMQ_CURVE_SERVER_KEYS_DIR"
    chmod 700 "$ZMQ_CURVE_SERVER_KEYS_DIR"

    if [[ -z "${UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY:-}" && -z "${UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY_FILE:-}" ]] || \
       [[ -z "${UMRS_ZMQ_CURVE_SERVER_SECRET_KEY:-}" && -z "${UMRS_ZMQ_CURVE_SERVER_SECRET_KEY_FILE:-}" ]]; then
        _run_util "
import zmq
from pathlib import Path

key_dir = Path('/root/.config/umrs/curve-keys/server')
key_dir.mkdir(parents=True, exist_ok=True)
public_path = key_dir / 'server.key'
secret_path = key_dir / 'server.key_secret'

if not public_path.exists() or not secret_path.exists():
    public_key, secret_key = zmq.curve_keypair()
    public_path.write_text(public_key.decode('ascii') + '\n', encoding='utf-8')
    secret_path.write_text(secret_key.decode('ascii') + '\n', encoding='utf-8')

public_path.chmod(0o600)
secret_path.chmod(0o600)
"
    fi

    if [[ -z "${UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY:-}" && -z "${UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY_FILE:-}" ]]; then
        export UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY_FILE="$ZMQ_CURVE_SERVER_KEYS_DIR/server.key"
    fi
    if [[ -z "${UMRS_ZMQ_CURVE_SERVER_SECRET_KEY:-}" && -z "${UMRS_ZMQ_CURVE_SERVER_SECRET_KEY_FILE:-}" ]]; then
        export UMRS_ZMQ_CURVE_SERVER_SECRET_KEY_FILE="$ZMQ_CURVE_SERVER_KEYS_DIR/server.key_secret"
    fi
fi

_run_util "
import os
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

bucket = os.environ['S3_BUCKET']
s3 = boto3.client(
    's3',
    endpoint_url=os.environ['S3_ENDPOINT_URL'],
    aws_access_key_id=os.environ['S3_ACCESS_KEY'],
    aws_secret_access_key=os.environ['S3_SECRET_KEY'],
    region_name=os.environ.get('S3_REGION', 'us-east-1'),
    config=BotoConfig(signature_version='s3v4', s3={'addressing_style': 'path'}),
    use_ssl=os.environ.get('S3_USE_SSL', 'false').lower() == 'true',
    verify=os.environ.get('S3_USE_SSL', 'false').lower() == 'true',
)
try:
    s3.head_bucket(Bucket=bucket)
except ClientError as exc:
    code = exc.response.get('Error', {}).get('Code')
    if code not in {'404', 'NoSuchBucket'}:
        raise
    s3.create_bucket(Bucket=bucket)
"

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

CONF_FILE="${CONF_FILE}" SCRIPT_DIR="${SCRIPT_DIR}" _run_util "
import json
import os
from pathlib import Path

conf_path = Path('/work/conf_umrs.json')
script_dir = Path('/work')
user_files_dir = script_dir / 'user_files'
if not conf_path.exists() or not user_files_dir.is_dir():
    raise SystemExit(0)

def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

def _canonical_config_ref(path):
    try:
        rel = path.relative_to(script_dir)
    except ValueError:
        return str(path)
    return f'./{rel.as_posix()}'

def _default_s3cfg_for(payload, fallback_name):
    username = (
        payload.get('s3_username')
        or payload.get('username')
        or payload.get('name')
        or fallback_name
    )
    return f'~/.config/umrs/s3cfg_{username}'

def _needs_remote_s3cfg_rewrite(value):
    if not isinstance(value, str) or not value.strip():
        return True
    raw = value.strip()
    if raw.startswith(('/Users/', '/private/var/')):
        return True
    return not (raw.startswith('~/') or Path(raw).is_absolute())

conf = json.loads(conf_path.read_text())
existing_users = list(conf.get('users') or [])
candidate_users = [str(user) for user in existing_users]

for user_file in sorted(user_files_dir.glob('*.json')):
    rel = f'./user_files/{user_file.name}'
    if rel not in candidate_users:
        candidate_users.append(rel)

users = []
seen = set()
required_keys = {'username', 'fullname', 's3_configuration_file', 'experiences'}
changed = False
conf_updates = {
    'admin_s3_configuration_file': '~/.config/umrs/s3cfg',
    'user_files_dir': './user_files',
}
for key, value in conf_updates.items():
    if conf.get(key) != value:
        conf[key] = value
        changed = True

for user in candidate_users:
    user_path = Path(user)
    if not user_path.is_absolute():
        user_path = script_dir / user_path
    try:
        payload = json.loads(user_path.read_text())
    except Exception as exc:
        print(f'WARNING: skipping unreadable user config {user}: {exc}')
        continue
    missing = sorted(required_keys - set(payload))
    if missing:
        print(f'WARNING: skipping invalid user config {user}: missing {chr(44).join(missing)}')
        continue
    if _needs_remote_s3cfg_rewrite(payload.get('s3_configuration_file')):
        payload['s3_configuration_file'] = _default_s3cfg_for(payload, user_path.stem)
        _write_json(user_path, payload)
    key = str(user_path.resolve())
    if key not in seen:
        canonical_user = _canonical_config_ref(user_path.resolve())
        users.append(canonical_user)
        seen.add(key)
        if canonical_user != user:
            changed = True

conf['users'] = users
if changed or conf.get('users') != existing_users:
    conf_path.write_text(json.dumps(conf, indent=2) + '\n')
"

# ─── Stop existing UMRS server containers (or legacy native processes) ────────
# Stop compose-managed UMRS containers (idempotent if not running)
echo "Stopping existing UMRS server containers..."
podman compose -f "${SCRIPT_DIR}/docker-compose.yaml" stop umrs-server-rccs umrs-server-ai4s 2>/dev/null || true

# Also clean up any legacy native PID-managed servers from before containerization
for pidfile in "${SCRIPT_DIR}"/server-*.pid; do
    [[ -f "$pidfile" ]] || continue
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping legacy native server (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
done

# Keep login-node MinIO reverse tunnels alive for Slurm jobs.
# Compute jobs forward minio-host:19000 -> login:9000, so login:9000 must
# reverse-forward to this front-end MinIO instance on each cluster.
_stop_existing_tunnel() {
    local pidfile="$1"
    local label="$2"
    if [[ -f "$pidfile" ]]; then
        local tunnel_pid
        tunnel_pid=$(cat "$pidfile")
        if kill -0 "$tunnel_pid" 2>/dev/null; then
            echo "Stopping existing ${label} MinIO reverse tunnel (PID $tunnel_pid)..."
            kill "$tunnel_pid" 2>/dev/null || true
            sleep 1
            kill -0 "$tunnel_pid" 2>/dev/null && kill -9 "$tunnel_pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
}

_launch_tunnel() {
    local label="$1"
    local ssh_target="$2"
    local reverse_spec="$3"
    local pidfile="$4"
    local logfile="$5"

    echo "Starting ${label} MinIO reverse tunnel: ${ssh_target} ${reverse_spec}"
    ssh -nNT \
        -R "$reverse_spec" \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        "$ssh_target" \
        >> "$logfile" 2>&1 &
    echo $! > "$pidfile"
}

_verify_tunnel() {
    local label="$1"
    local ssh_target="$2"
    local pidfile="$3"
    local logfile="$4"

    if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "WARNING: ${label} MinIO reverse tunnel process exited. Checking existing login-node endpoint..."
        tail -20 "$logfile" 2>/dev/null || true
        if ssh "$ssh_target" "curl -sf http://127.0.0.1:9000/minio/health/live >/dev/null"; then
            echo "  ${label} login node already has a healthy MinIO endpoint on 127.0.0.1:9000."
            rm -f "$pidfile"
            return 0
        fi
        echo "ERROR: ${label} MinIO reverse tunnel failed to start. Check ${logfile}"
        exit 1
    fi
    if ! ssh "$ssh_target" "curl -sf http://127.0.0.1:9000/minio/health/live >/dev/null"; then
        echo "ERROR: ${label} login node cannot reach MinIO through the reverse tunnel."
        exit 1
    fi
}

RCCS_SSH_TARGET="${RCCS_SSH_TARGET:-masaru.nagaso@rccs-cloud}"
AI4S_SSH_TARGET="${AI4S_SSH_TARGET:-ea0022@AI4S}"
RCCS_MINIO_REVERSE_SPEC="${RCCS_MINIO_REVERSE_SPEC:-127.0.0.1:9000:127.0.0.1:${MINIO_PORT:-9000}}"
AI4S_MINIO_REVERSE_SPEC="${AI4S_MINIO_REVERSE_SPEC:-127.0.0.1:9000:127.0.0.1:${MINIO_PORT:-9000}}"

# Stop existing tunnels
_stop_existing_tunnel "${SCRIPT_DIR}/rccs-minio-reverse-tunnel.pid" "RCCS"
_stop_existing_tunnel "${SCRIPT_DIR}/ai4s-minio-reverse-tunnel.pid" "AI4S"

# Launch both tunnels in parallel
_launch_tunnel "RCCS" "$RCCS_SSH_TARGET" "$RCCS_MINIO_REVERSE_SPEC" \
    "${SCRIPT_DIR}/rccs-minio-reverse-tunnel.pid" "${LOG_DIR}/rccs-minio-reverse-tunnel.log"
_launch_tunnel "AI4S" "$AI4S_SSH_TARGET" "$AI4S_MINIO_REVERSE_SPEC" \
    "${SCRIPT_DIR}/ai4s-minio-reverse-tunnel.pid" "${LOG_DIR}/ai4s-minio-reverse-tunnel.log"

# Wait once for both SSH processes to establish
sleep 3

# Verify both tunnels
_verify_tunnel "RCCS" "$RCCS_SSH_TARGET" \
    "${SCRIPT_DIR}/rccs-minio-reverse-tunnel.pid" "${LOG_DIR}/rccs-minio-reverse-tunnel.log"
_verify_tunnel "AI4S" "$AI4S_SSH_TARGET" \
    "${SCRIPT_DIR}/ai4s-minio-reverse-tunnel.pid" "${LOG_DIR}/ai4s-minio-reverse-tunnel.log"

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
_stop_port_listener "${UMRS_RCCS_ZMQ_PORT}" "RCCS ZMQ"
[[ "${UMRS_RCCS_ZMQ_PORT}" == "5555" ]] || _stop_port_listener "5555" "legacy RCCS ZMQ"
_stop_port_listener "${UMRS_AI4S_PORT:-8002}" "AI4S HTTP"
_stop_port_listener "${UMRS_AI4S_ZMQ_PORT}" "AI4S ZMQ"
[[ "${UMRS_AI4S_ZMQ_PORT}" == "5557" ]] || _stop_port_listener "5557" "legacy AI4S ZMQ"
sleep 2
_stop_port_listener "${UMRS_RCCS_PORT:-8000}" "RCCS HTTP"
_stop_port_listener "${UMRS_RCCS_ZMQ_PORT}" "RCCS ZMQ"
[[ "${UMRS_RCCS_ZMQ_PORT}" == "5555" ]] || _stop_port_listener "5555" "legacy RCCS ZMQ"
_stop_port_listener "${UMRS_AI4S_PORT:-8002}" "AI4S HTTP"
_stop_port_listener "${UMRS_AI4S_ZMQ_PORT}" "AI4S ZMQ"
[[ "${UMRS_AI4S_ZMQ_PORT}" == "5557" ]] || _stop_port_listener "5557" "legacy AI4S ZMQ"

# ─── Build and start UMRS server containers ──────────────────────────────────
# Remove legacy containers that conflict with compose-managed names
for _ctr in umrs-server umrs-server-rccs umrs-server-ai4s; do
    if podman inspect "$_ctr" >/dev/null 2>&1 && \
       [[ -z "$(podman inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "$_ctr" 2>/dev/null)" ]]; then
        echo "  Removing legacy container $_ctr..."
        podman stop "$_ctr" 2>/dev/null || true
        podman rm -f "$_ctr" 2>/dev/null || true
    fi
done

echo ""
echo "Starting UMRS server containers via podman compose..."
echo "  RCCS site config: ../rccs-front/site.yaml"
echo "  AI4S site config: ../ai4s/site.yaml"
echo "  RCCS ZMQ port: ${UMRS_RCCS_ZMQ_PORT}, HTTP port: ${UMRS_RCCS_PORT:-8000}"
echo "  AI4S ZMQ port: ${UMRS_AI4S_ZMQ_PORT}, HTTP port: ${UMRS_AI4S_PORT:-8002}"
echo "  Tunnel mode: ${UMRS_TUNNEL_MODE}"
echo "  Server host: ${UMRS_SERVER_HOST}"

# Export variables needed by docker-compose.yaml
export UMRS_TUNNEL_MODE="${UMRS_TUNNEL_MODE:-false}"
export UMRS_SERVER_HOST="${UMRS_SERVER_HOST:-0.0.0.0}"
export S3_PRESIGN_ENDPOINT="${S3_PRESIGN_ENDPOINT:-http://127.0.0.1:${MINIO_PORT:-9000}}"
export VISION_S3_PRESIGN_ENDPOINT="${VISION_S3_PRESIGN_ENDPOINT:-http://minio-host:19000}"
export UMRS_ADMIN_S3_CONFIGURATION_FILE="${UMRS_ADMIN_S3_CONFIGURATION_FILE:-/etc/umrs/s3cfg}"

# Start UMRS server containers (image already built earlier)
podman compose -f "${SCRIPT_DIR}/docker-compose.yaml" up -d \
    umrs-server-rccs umrs-server-ai4s

# Wait for RCCS server readiness
echo -n "  Waiting for RCCS HTTP bootstrap..."
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${UMRS_RCCS_PORT:-8000}/health" >/dev/null 2>&1; then
        echo " ready."
        break
    fi
    if [[ "$(podman inspect -f '{{.State.Status}}' umrs-server-rccs 2>/dev/null)" == "exited" ]]; then
        echo " failed (container exited)."
        echo "ERROR: RCCS server container exited. Check logs:"
        echo "  podman compose -f docker-compose.yaml logs umrs-server-rccs"
        podman compose -f "${SCRIPT_DIR}/docker-compose.yaml" logs --tail=20 umrs-server-rccs 2>/dev/null || true
        exit 1
    fi
    echo -n "."
    sleep 1
    [[ "$i" -eq 90 ]] && echo " WARNING: RCCS HTTP not ready after 90s"
done

# Wait for AI4S server readiness
echo -n "  Waiting for AI4S HTTP bootstrap..."
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${UMRS_AI4S_PORT:-8002}/health" >/dev/null 2>&1; then
        echo " ready."
        break
    fi
    if [[ "$(podman inspect -f '{{.State.Status}}' umrs-server-ai4s 2>/dev/null)" == "exited" ]]; then
        echo " failed (container exited)."
        echo "ERROR: AI4S server container exited. Check logs:"
        echo "  podman compose -f docker-compose.yaml logs umrs-server-ai4s"
        podman compose -f "${SCRIPT_DIR}/docker-compose.yaml" logs --tail=20 umrs-server-ai4s 2>/dev/null || true
        exit 1
    fi
    echo -n "."
    sleep 1
    [[ "$i" -eq 90 ]] && echo " WARNING: AI4S HTTP not ready after 90s"
done

# ─── Start RAG Service (containerized) ───────────────────────────────────────
RAG_ENABLED=false
if [[ "${UMRS_RAG_ENABLE:-true}" == "true" || "${UMRS_RAG_ENABLE:-true}" == "1" ]]; then
    RAG_ENABLED=true
fi

if [[ "$RAG_ENABLED" == "true" ]]; then
    echo ""
    echo "Starting RAG service (container)..."
    echo "  Port: ${UMRS_RAG_PORT:-8081}"

    # Stop existing RAG container if any
    podman stop umrs-rag 2>/dev/null || true
    podman rm -f umrs-rag 2>/dev/null || true

    # Kill stale native RAG process if any (leftover from pre-container era)
    RAG_BLOCKING_PID=$(ss -tlnp "sport = :${UMRS_RAG_PORT:-8081}" 2>/dev/null \
        | awk -F'pid=' '/pid=/{print $2}' | awk -F',' '{print $1}' | head -1)
    if [[ -n "${RAG_BLOCKING_PID:-}" ]]; then
        echo "  Stopping stale native RAG on port ${UMRS_RAG_PORT:-8081} (PID ${RAG_BLOCKING_PID})..."
        kill "$RAG_BLOCKING_PID" 2>/dev/null || true
        sleep 2
    fi

    podman run -d --name umrs-rag \
        --network=host \
        -v "${SCRIPT_DIR}:/work:ro,z" \
        -v "${HOME}/.config/umrs:/root/.config/umrs:z" \
        -e UMRS_RAG_PORT="${UMRS_RAG_PORT:-8081}" \
        -e UMRS_RAG_BIND_HOST="127.0.0.1" \
        -e UMRS_CONFIG="/work" \
        -e OPENBLAS_NUM_THREADS=1 \
        -e S3_ENDPOINT_URL -e S3_ACCESS_KEY -e S3_SECRET_KEY \
        -e S3_REGION -e S3_BUCKET -e S3_USE_SSL \
        -e DB_HOST="127.0.0.1" \
        -e DB_PORT="${POSTGRES_PORT:-5432}" \
        -e DB_SUPERUSER="${POSTGRES_USER:-umrs}" \
        -e DB_SUPERUSER_PASSWORD="${POSTGRES_PASSWORD}" \
        -e PYTHONUNBUFFERED=1 \
        "$UMRS_IMAGE" python -m umrs.api.rag_service

    echo -n "  Waiting for RAG readiness..."
    for i in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:${UMRS_RAG_PORT:-8081}/docs" >/dev/null 2>&1; then
            echo " ready."
            break
        fi
        if [[ "$(podman inspect -f '{{.State.Status}}' umrs-rag 2>/dev/null)" == "exited" ]]; then
            echo " failed."
            echo "  WARNING: RAG container exited. Check logs:"
            podman logs --tail=10 umrs-rag 2>/dev/null || true
            RAG_ENABLED=false
            break
        fi
        echo -n "."
        sleep 1
        [[ "$i" -eq 120 ]] && echo " WARNING: RAG not ready after 120s"
    done
fi

# ─── Verify servers started ──────────────────────────────────────────────────
echo ""
echo -n "Verifying UMRS containers..."
sleep 3
FAIL=0

RCCS_STATUS="$(podman inspect -f '{{.State.Status}}' umrs-server-rccs 2>/dev/null || echo 'missing')"
if [[ "$RCCS_STATUS" != "running" ]]; then
    echo ""
    echo "ERROR: RCCS server container not running (status: ${RCCS_STATUS})"
    podman compose -f "${SCRIPT_DIR}/docker-compose.yaml" logs --tail=20 umrs-server-rccs 2>/dev/null || true
    FAIL=1
fi

AI4S_STATUS="$(podman inspect -f '{{.State.Status}}' umrs-server-ai4s 2>/dev/null || echo 'missing')"
if [[ "$AI4S_STATUS" != "running" ]]; then
    echo ""
    echo "ERROR: AI4S server container not running (status: ${AI4S_STATUS})"
    podman compose -f "${SCRIPT_DIR}/docker-compose.yaml" logs --tail=20 umrs-server-ai4s 2>/dev/null || true
    FAIL=1
fi

if [[ "$FAIL" -eq 0 ]]; then
    echo " both running."
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  UMRS Dual-Cluster Stack (umrs-front) — containerized       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Tunnel mode:  ${UMRS_TUNNEL_MODE}                                       ║"
echo "║  Server host:  ${UMRS_SERVER_HOST}                                   ║"
echo "║  Postgres:     127.0.0.1:${POSTGRES_PORT:-5432}  (container)             ║"
echo "║  MinIO:        ${MINIO_BIND}:${MINIO_PORT:-9000}  (container)                ║"
echo "║  S3 Presign:   ${S3_PRESIGN_ENDPOINT}          ║"
echo "║                                                              ║"
echo "║  RCCS Cloud:   (container: umrs-server-rccs)                 ║"
echo "║    UMRS HTTP:  0.0.0.0:${UMRS_RCCS_PORT:-8000}                            ║"
echo "║    Site:       rccs-front/site.yaml                          ║"
echo "║    Target:     ${RCCS_SSH_TARGET} (SSH)                              ║"
echo "║                                                              ║"
echo "║  AI4S:         (container: umrs-server-ai4s)                 ║"
echo "║    UMRS HTTP:  0.0.0.0:${UMRS_AI4S_PORT:-8002}                            ║"
echo "║    Site:       ai4s/site.yaml                                ║"
echo "║    Target:     ${AI4S_SSH_TARGET} (SSH)                      ║"
if [[ "$RAG_ENABLED" == "true" ]]; then
echo "║                                                              ║"
echo "║  RAG Service:  127.0.0.1:${UMRS_RAG_PORT:-8081}  (container: umrs-rag)   ║"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Logs:"
echo "  RCCS:  podman compose -f docker-compose.yaml logs -f umrs-server-rccs"
echo "  AI4S:  podman compose -f docker-compose.yaml logs -f umrs-server-ai4s"
if [[ "$RAG_ENABLED" == "true" ]]; then
echo "  RAG:   podman logs -f umrs-rag"
fi
echo ""
echo "Stop:    ./stop.sh"
echo "Restart: ./stop.sh --server && ./start.sh --server"
