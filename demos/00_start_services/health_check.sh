#!/usr/bin/env bash
# =============================================================================
# Health check for UMRS services
# =============================================================================
set -Eeuo pipefail

AI_LOGIN_CONTAINER="ai-login"
PG_CONTAINER="demo-postgres"
PG_SUPERUSER="${POSTGRES_USER:-umrs}"

LOG_PREFIX="[health_check]"
log() { echo "$LOG_PREFIX $*"; }
warn() { echo "$LOG_PREFIX WARNING: $*"; }
ok=0
fail=0

check() {
    local name="$1" cmd="$2"
    if eval "$cmd" >/dev/null 2>&1; then
        log "  $name: OK"
        ok=$((ok + 1))
    else
        warn "  $name: FAILED"
        fail=$((fail + 1))
    fi
}

log "Checking UMRS services..."
echo

check "ai-login container" "podman inspect $AI_LOGIN_CONTAINER"
check "postgres container"  "podman inspect $PG_CONTAINER"
check "minio container"     "podman inspect demo-minio"
check "host slurmd"        "pgrep -f slurmd"
check "Slurm node"         "podman exec $AI_LOGIN_CONTAINER sinfo -N -h -o '%N %T' | grep '^compute-01'"
check "pgvector"           "podman exec $PG_CONTAINER pg_isready -U $PG_SUPERUSER"

echo
log "Results: $ok passed, $fail failed"
[ "$fail" -eq 0 ] && log "All healthy!" || warn "Some checks failed — see above"
exit "$fail"
