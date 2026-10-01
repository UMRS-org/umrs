#!/usr/bin/env bash
# =============================================================================
# umrs-extdb.sh — external Postgres (pgvector) + SeaweedFS (S3) for a UMRS
# project, as rootless podman containers on an HPC login node.
# =============================================================================
# Operator tool for per-project external storage (ADR-077, 2026-10-01): the
# stores live next to the cluster's large filesystem instead of on umrs-front,
# and a UMRS project is pointed at them by URI (postgres) and URL (S3) with
# `r4 project storage set`. Run by hand on the login node — not converged.
#
# Layout (EXTDB_ROOT, on Lustre; 0700):
#   postgres/pgdata         PGDATA (owned by the invoking user via keep-id)
#   seaweedfs/data          SeaweedFS -dir (filer leveldb + volumes)
#   config/                 pg_hba.conf, TLS cert/key, seaweedfs iam.json
#   secrets/extdb.env       generated once: store admin credentials (0600)
#   secrets/project-*.env   per-project URIs for UMRS (0600)
#   podman/                 isolated podman config (see below)
#
# Podman isolation: ~/.config/containers/storage.conf on this site is
# rewritten by every serve-model Slurm job with that job's runroot ($HOME is
# shared with the compute nodes), which orphans long-running containers. All
# podman calls here use CONTAINERS_STORAGE_CONF / CONTAINERS_CONF pointing at
# EXTDB_ROOT/podman, with graphroot and runroot on the node's local disk
# (LOCAL_ROOT, overlay does not work on Lustre).
#
# Login-node caveat: /var/tmp is aged out by systemd-tmpfiles after 30 days
# of no access. `up` and `status` refresh access times under LOCAL_ROOT; run
# `status` at least every two weeks while the stores are in use.
#
# Usage:
#   umrs-extdb.sh up                 init (idempotent) + start both containers
#   umrs-extdb.sh down               stop + remove containers (data kept)
#   umrs-extdb.sh status             container state + health probes
#   umrs-extdb.sh logs pg|s3         tail container logs
#   umrs-extdb.sh psql [args]        psql as the store superuser (in container)
#   umrs-extdb.sh project <name>     role + database + pgvector for <name>,
#                                    writes secrets/project-<name>.env
# Secrets are never printed.
# =============================================================================
set -euo pipefail

EXTDB_ROOT="${EXTDB_ROOT:-/lvs0/dne1/rccs-dmp/${USER}/umrs-extdb}"
LOCAL_ROOT="${EXTDB_LOCAL_ROOT:-/var/tmp/umrs-extdb-$(id -u)}"
PG_IMAGE="${EXTDB_PG_IMAGE:-docker.io/pgvector/pgvector:pg15}"
S3_IMAGE="${EXTDB_S3_IMAGE:-docker.io/chrislusf/seaweedfs:4.47}"
PG_CTR="${EXTDB_PG_CONTAINER:-umrs-extdb-postgres}"
S3_CTR="${EXTDB_S3_CONTAINER:-umrs-extdb-seaweedfs}"
PG_PORT="${EXTDB_PG_PORT:-25432}"
S3_PORT="${EXTDB_S3_PORT:-28900}"
BIND="${EXTDB_BIND:-0.0.0.0}"
# Space-separated CIDRs allowed to reach Postgres (default: umrs-front/data0).
PG_ALLOW="${EXTDB_PG_ALLOW:-10.134.0.8/32}"
# Bounded on the shared project quota: 1 GiB volumes x 200 = 200 GiB.
S3_VOLUME_MAX="${EXTDB_S3_VOLUME_MAX:-200}"
S3_VOLUME_SIZE_MB="${EXTDB_S3_VOLUME_SIZE_MB:-1024}"
# Address UMRS (on the front-end) uses to reach this node; default = the
# source address this node uses towards umrs-front (data0).
ADVERTISE_HOST="${EXTDB_ADVERTISE_HOST:-$(ip -4 route get "${EXTDB_FRONT_IP:-10.134.0.8}" 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p')}"

UID_NUM="$(id -u)"
GID_NUM="$(id -g)"
SECRETS="${EXTDB_ROOT}/secrets/extdb.env"

die() { echo "ERROR: $*" >&2; exit 1; }

podman_env() {
    export CONTAINERS_STORAGE_CONF="${EXTDB_ROOT}/podman/storage.conf"
    export CONTAINERS_CONF="${EXTDB_ROOT}/podman/containers.conf"
    export XDG_RUNTIME_DIR="${LOCAL_ROOT}/xdg"
    export GOMAXPROCS="${GOMAXPROCS:-4}"
}

pm() { podman_env; podman "$@"; }

write_podman_conf() {
    mkdir -p "${EXTDB_ROOT}/podman" "${LOCAL_ROOT}/graph" "${LOCAL_ROOT}/run" \
        "${LOCAL_ROOT}/tmp" "${LOCAL_ROOT}/xdg"
    chmod 700 "${LOCAL_ROOT}" "${LOCAL_ROOT}/xdg"
    cat > "${EXTDB_ROOT}/podman/storage.conf" <<EOF
# Isolated podman storage for umrs-extdb (see umrs-extdb.sh header).
[storage]
driver = "overlay"
graphroot = "${LOCAL_ROOT}/graph"
runroot = "${LOCAL_ROOT}/run"
[storage.options]
mount_program = "/usr/bin/fuse-overlayfs"
[storage.options.overlay]
mountopt = "xattr_permissions=2,threaded=1"
EOF
    cat > "${EXTDB_ROOT}/podman/containers.conf" <<EOF
# Isolated podman engine config for umrs-extdb (see umrs-extdb.sh header).
[containers]
label = false
log_driver = "k8s-file"
log_size_max = 52428800
[engine]
cgroup_manager = "cgroupfs"
events_logger = "file"
tmp_dir = "${LOCAL_ROOT}/tmp"
EOF
}

gen_secret() { openssl rand -hex "$1"; }

write_secrets_once() {
    mkdir -p "${EXTDB_ROOT}/secrets"
    chmod 700 "${EXTDB_ROOT}/secrets"
    [[ -s "$SECRETS" ]] && return 0
    ( umask 077
      {
        echo "# umrs-extdb store credentials — generated $(date -Is); never commit"
        echo "PG_SUPERUSER=umrs_ext"
        echo "PG_SUPERUSER_PASSWORD=$(gen_secret 24)"
        echo "S3_ROOT_USER=$(gen_secret 10 | tr 'a-f' 'A-F')"
        echo "S3_ROOT_PASSWORD=$(gen_secret 20)"
        echo "S3_STS_SIGNING_KEY=$(head -c 32 /dev/urandom | base64 -w0)"
      } > "$SECRETS" )
    echo "  generated store credentials into ${SECRETS} (0600)"
}

load_secrets() {
    [[ -s "$SECRETS" ]] || die "no ${SECRETS}; run: $0 up"
    set -o allexport
    # shellcheck disable=SC1090
    source "$SECRETS"
    set +o allexport
}

# Container env files (0600): secrets must not ride podman's argv, which any
# user of a shared login node can read from the process table.
write_env_files() {
    load_secrets
    ( umask 077
      {
        echo "POSTGRES_USER=${PG_SUPERUSER}"
        echo "POSTGRES_PASSWORD=${PG_SUPERUSER_PASSWORD}"
        echo "POSTGRES_DB=postgres"
        echo "POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256 --auth-local=trust"
        echo "PGDATA=/var/lib/postgresql/data/pgdata"
      } > "${EXTDB_ROOT}/secrets/pg.ctr.env"
      {
        echo "AWS_ACCESS_KEY_ID=${S3_ROOT_USER}"
        echo "AWS_SECRET_ACCESS_KEY=${S3_ROOT_PASSWORD}"
      } > "${EXTDB_ROOT}/secrets/s3.ctr.env" )
}

write_pg_config() {
    local cfg="${EXTDB_ROOT}/config"
    mkdir -p "${cfg}/tls"
    chmod 700 "${cfg}" "${cfg}/tls"
    # Only the UMRS front-end (PG_ALLOW) and this node itself may connect,
    # with SCRAM; everything else is rejected. Rootless pasta forwarding keeps
    # the client source address (verified 2026-10-01: data0 seen as
    # 10.134.0.8). Container-local admin sessions (podman exec) trust.
    local cidr
    {
        echo "# TYPE  DATABASE  USER  ADDRESS            METHOD  (rendered by umrs-extdb.sh)"
        echo "local   all       all                      trust"
        echo "host    all       all   127.0.0.1/32       scram-sha-256"
        echo "host    all       all   ::1/128            scram-sha-256"
        for cidr in ${PG_ALLOW}; do
            echo "host    all       all   ${cidr}    scram-sha-256"
        done
        echo "host    all       all   0.0.0.0/0          reject"
        echo "host    all       all   ::/0               reject"
    } > "${cfg}/pg_hba.conf"
    chmod 644 "${cfg}/pg_hba.conf"
    if [[ ! -s "${cfg}/tls/server.key" ]]; then
        ( umask 077
          openssl req -x509 -newkey rsa:3072 -nodes -days 825 \
              -subj "/CN=umrs-extdb-${ADVERTISE_HOST:-login}" \
              -keyout "${cfg}/tls/server.key" -out "${cfg}/tls/server.crt" \
              >/dev/null 2>&1 )
        chmod 600 "${cfg}/tls/server.key"
        chmod 644 "${cfg}/tls/server.crt"
        echo "  generated self-signed Postgres TLS certificate"
    fi
}

# Same STS shape as umrs-front (ADR-055): role umrs-attempt, trust pinned to
# the env-bootstrapped admin principal ("admin-" + first 8 chars of the access
# key), no s3:ListBucket (SeaweedFS ignores the s3:prefix condition on it).
# The store is dedicated to its project(s), so the base role spans its buckets;
# the per-attempt session policy narrows to the attempt's prefixes.
write_s3_config() {
    load_secrets
    local cfg="${EXTDB_ROOT}/config"
    mkdir -p "${cfg}"
    ( umask 022
      cat > "${cfg}/iam.json" <<EOF
{
  "sts": {
    "tokenDuration": "1h",
    "maxSessionLength": "1h",
    "issuer": "umrs-extdb-sts",
    "signingKey": "${S3_STS_SIGNING_KEY}"
  },
  "policies": [
    {
      "name": "UmrsAttemptRoleBase",
      "document": {
        "Version": "2012-10-17",
        "Statement": [
          {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:GetBucketLocation"],
            "Resource": ["arn:aws:s3:::*"]
          }
        ]
      }
    }
  ],
  "roles": [
    {
      "roleName": "umrs-attempt",
      "roleArn": "arn:aws:iam::role/umrs-attempt",
      "attachedPolicies": ["UmrsAttemptRoleBase"],
      "trustPolicy": {
        "Version": "2012-10-17",
        "Statement": [
          {
            "Effect": "Allow",
            "Principal": {"AWS": "arn:aws:iam::000000000000:user/admin-${S3_ROOT_USER:0:8}"},
            "Action": ["sts:AssumeRole"]
          }
        ]
      }
    }
  ]
}
EOF
    )
    chmod 644 "${cfg}/iam.json"
}

init() {
    [[ -n "$ADVERTISE_HOST" ]] || die "cannot derive the advertise host; set EXTDB_ADVERTISE_HOST"
    mkdir -p "${EXTDB_ROOT}"
    chmod 700 "${EXTDB_ROOT}"
    mkdir -p "${EXTDB_ROOT}/postgres" "${EXTDB_ROOT}/seaweedfs/data"
    chmod 700 "${EXTDB_ROOT}/postgres" "${EXTDB_ROOT}/seaweedfs" "${EXTDB_ROOT}/seaweedfs/data"
    write_podman_conf
    write_secrets_once
    write_env_files
    write_pg_config
    write_s3_config
}

keepalive() {
    # Refresh access times so systemd-tmpfiles' 30-day age-out of /var/tmp
    # does not remove image layers or runtime state of running containers.
    find "${LOCAL_ROOT}" -xdev -exec touch -a -h {} + 2>/dev/null || true
}

ctr_state() { pm inspect -f '{{.State.Status}}' "$1" 2>/dev/null || echo missing; }

ensure_image() { pm image exists "$1" || pm pull -q "$1" >/dev/null; }

# Unix sockets do not work on the rootless fuse-overlayfs layer of this site
# (2026-10-01): Postgres' socket of the initdb temp server survived it and
# every later start failed with "could not bind Unix address ... Address
# already in use"; SeaweedFS' local gRPC sockets in /tmp answered connect()
# with "permission denied". Both socket dirs are tmpfs mounts.
start_pg() {
    load_secrets
    case "$(ctr_state "$PG_CTR")" in
        running)
            # pg_hba.conf is re-rendered by init; apply it without a restart.
            pm exec "$PG_CTR" pg_ctl reload -s -D /var/lib/postgresql/data/pgdata >/dev/null
            echo "  ${PG_CTR}: running (config reloaded)"; return 0 ;;
        missing) ;;
        *) pm rm -f "$PG_CTR" >/dev/null ;;
    esac
    ensure_image "$PG_IMAGE"
    pm run -d --name "$PG_CTR" \
        --userns=keep-id --user "${UID_NUM}:${GID_NUM}" \
        --restart=unless-stopped --no-healthcheck --shm-size=256m \
        --tmpfs /var/run/postgresql:rw,mode=1777,size=1m \
        -p "${BIND}:${PG_PORT}:5432" \
        --env-file "${EXTDB_ROOT}/secrets/pg.ctr.env" \
        -v "${EXTDB_ROOT}/postgres:/var/lib/postgresql/data" \
        -v "${EXTDB_ROOT}/config/pg_hba.conf:/etc/umrs-extdb/pg_hba.conf:ro" \
        -v "${EXTDB_ROOT}/config/tls:/etc/umrs-extdb/tls:ro" \
        "$PG_IMAGE" \
        postgres -c listen_addresses='*' -c max_connections=200 \
            -c hba_file=/etc/umrs-extdb/pg_hba.conf \
            -c ssl=on -c ssl_cert_file=/etc/umrs-extdb/tls/server.crt \
            -c ssl_key_file=/etc/umrs-extdb/tls/server.key \
            -c password_encryption=scram-sha-256 >/dev/null
    echo "  ${PG_CTR}: started"
}

start_s3() {
    load_secrets
    case "$(ctr_state "$S3_CTR")" in
        running) echo "  ${S3_CTR}: running"; return 0 ;;
        missing) ;;
        *) pm rm -f "$S3_CTR" >/dev/null ;;
    esac
    ensure_image "$S3_IMAGE"
    # Only the S3 API is published. Master (9333), volume (8080), filer
    # (8888), IAM (8111) and metrics (9327) stay inside the container netns:
    # the first three are unauthenticated by design.
    pm run -d --name "$S3_CTR" \
        --userns=keep-id --user "${UID_NUM}:${GID_NUM}" \
        --restart=unless-stopped --no-healthcheck \
        --tmpfs /tmp:rw,mode=1777,size=64m \
        -p "${BIND}:${S3_PORT}:9000" \
        --env-file "${EXTDB_ROOT}/secrets/s3.ctr.env" \
        -v "${EXTDB_ROOT}/seaweedfs/data:/data" \
        -v "${EXTDB_ROOT}/config/iam.json:/etc/seaweedfs/iam.json:ro" \
        "$S3_IMAGE" \
        server -dir=/data -ip=127.0.0.1 -ip.bind=0.0.0.0 \
            -volume.max="$S3_VOLUME_MAX" -master.volumeSizeLimitMB="$S3_VOLUME_SIZE_MB" \
            -volume.index=leveldb \
            -s3 -s3.port=9000 -s3.iam -s3.iam.config=/etc/seaweedfs/iam.json \
            -iam -iam.port=8111 -metricsPort=9327 >/dev/null
    echo "  ${S3_CTR}: started"
}

wait_pg() {
    local i
    for i in $(seq 1 120); do
        if pm exec "$PG_CTR" pg_isready -q -h 127.0.0.1 -p 5432 2>/dev/null; then
            echo "  postgres ready (${i}s)"; return 0
        fi
        sleep 1
    done
    pm logs --tail 30 "$PG_CTR" >&2 || true
    die "postgres not ready after 120s"
}

wait_s3() {
    local i
    for i in $(seq 1 180); do
        if curl -sf -m 2 "http://127.0.0.1:${S3_PORT}/status" >/dev/null 2>&1; then
            echo "  seaweedfs ready (${i}s)"; return 0
        fi
        sleep 1
    done
    pm logs --tail 30 "$S3_CTR" >&2 || true
    die "seaweedfs not ready after 180s"
}

up() {
    init
    start_pg
    start_s3
    wait_pg
    wait_s3
    keepalive
    status
}

down() {
    local c
    for c in "$S3_CTR" "$PG_CTR"; do
        if [[ "$(ctr_state "$c")" != missing ]]; then
            pm stop -t 30 "$c" >/dev/null 2>&1 || true
            pm rm -f "$c" >/dev/null
            echo "  ${c}: removed (data kept under ${EXTDB_ROOT})"
        fi
    done
}

status() {
    keepalive
    echo "umrs-extdb on $(hostname -s): advertise ${ADVERTISE_HOST}"
    echo "  postgres  ${PG_CTR}: $(ctr_state "$PG_CTR")  ${ADVERTISE_HOST}:${PG_PORT}  data ${EXTDB_ROOT}/postgres"
    echo "  seaweedfs ${S3_CTR}: $(ctr_state "$S3_CTR")  http://${ADVERTISE_HOST}:${S3_PORT}  data ${EXTDB_ROOT}/seaweedfs/data"
    if pm exec "$PG_CTR" pg_isready -q -h 127.0.0.1 -p 5432 2>/dev/null; then
        echo "  postgres probe: ok"
    else
        echo "  postgres probe: FAIL"
    fi
    if curl -sf -m 3 "http://127.0.0.1:${S3_PORT}/status" >/dev/null 2>&1; then
        echo "  s3 probe: ok"
    else
        echo "  s3 probe: FAIL"
    fi
}

psql_cmd() {
    load_secrets
    pm exec -i "$PG_CTR" psql -v ON_ERROR_STOP=1 -U "$PG_SUPERUSER" -d postgres "$@"
}

# Least privilege for UMRS: a login role that owns one database (pgvector
# installed by the superuser); the store superuser never leaves this node.
project() {
    local name="${1:-}"
    [[ "$name" =~ ^[a-z][a-z0-9_]{1,40}$ ]] || die "project name must match ^[a-z][a-z0-9_]{1,40}\$"
    load_secrets
    local env_file="${EXTDB_ROOT}/secrets/project-${name}.env"
    local pw
    if [[ -s "$env_file" ]]; then
        pw="$(sed -n 's/^PROJECT_PG_PASSWORD=//p' "$env_file")"
    fi
    [[ -n "${pw:-}" ]] || pw="$(gen_secret 24)"
    psql_cmd -q >/dev/null <<EOF
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${name}') THEN
    CREATE ROLE ${name} LOGIN;
  END IF;
END
\$\$;
ALTER ROLE ${name} WITH LOGIN PASSWORD '${pw}';
EOF
    if ! psql_cmd -tAc "SELECT 1 FROM pg_database WHERE datname = '${name}'" | grep -q 1; then
        psql_cmd -q -c "CREATE DATABASE ${name} OWNER ${name};" >/dev/null
    fi
    pm exec -i "$PG_CTR" psql -v ON_ERROR_STOP=1 -U "$PG_SUPERUSER" -d "$name" -q \
        -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null
    ( umask 077
      {
        echo "# UMRS external storage for project '${name}' — generated $(date -Is); never commit"
        echo "PROJECT_PG_PASSWORD=${pw}"
        echo "UMRS_PROJECT_POSTGRES_URI=postgresql://${name}:${pw}@${ADVERTISE_HOST}:${PG_PORT}/${name}?sslmode=require"
        echo "UMRS_PROJECT_S3_URL=http://${S3_ROOT_USER}:${S3_ROOT_PASSWORD}@${ADVERTISE_HOST}:${S3_PORT}"
      } > "$env_file" )
    echo "  project ${name}: role + database + vector ready; URIs in ${env_file} (0600)"
}

case "${1:-}" in
    up) up ;;
    down) down ;;
    status) status ;;
    logs)
        case "${2:-}" in
            pg) pm logs --tail "${3:-50}" "$PG_CTR" ;;
            s3) pm logs --tail "${3:-50}" "$S3_CTR" ;;
            *) die "usage: $0 logs pg|s3 [lines]" ;;
        esac ;;
    psql) shift; psql_cmd "$@" ;;
    project) shift; project "$@" ;;
    podman) shift; pm "$@" ;;
    *) sed -n '2,40p' "$0"; exit 2 ;;
esac
