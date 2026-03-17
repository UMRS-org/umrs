#!/usr/bin/env bash
# =============================================================================
# Start all UMRS services (control plane + host slurmd)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"
SLURM_DEV_DIR="$DEMO_DIR/slurm-dev"
COMPOSE_FILE="$SLURM_DEV_DIR/docker-compose.yaml"

AI_LOGIN_CONTAINER="ai-login"

LOG_PREFIX="[start_all]"
log() { echo "$LOG_PREFIX $*"; }
die() { echo "$LOG_PREFIX ERROR: $*" >&2; exit 1; }

wait_for_condition() {
    local max_attempts="$1" sleep_seconds="$2" description="$3" probe_cmd="$4"
    for i in $(seq 1 "$max_attempts"); do
        if eval "$probe_cmd" >/dev/null 2>&1; then
            log "$description: ready"
            return 0
        fi
        [ "$i" -eq "$max_attempts" ] && return 1
        sleep "$sleep_seconds"
    done
    return 1
}

detect_compose() {
    if command -v podman-compose >/dev/null 2>&1; then
        COMPOSE_CMD=(podman-compose)
    elif command -v podman >/dev/null 2>&1 && podman compose version >/dev/null 2>&1; then
        COMPOSE_CMD=(podman compose)
    else
        die "Neither podman-compose nor podman compose is available."
    fi
}

# ---------- Preflight ----------
log "Checking prerequisites..."
[[ -f "$COMPOSE_FILE" ]] || die "Missing compose file: $COMPOSE_FILE (run init-setup.sh first)"
[[ -f "$SLURM_DEV_DIR/.env" ]] || die "Missing .env file: run 'bash $SLURM_DEV_DIR/scripts/init-setup.sh --auto' first"
detect_compose
log "Compose command: ${COMPOSE_CMD[*]}"

# ---------- Step 1: Start control plane ----------
log ""
log "Step 1: Starting control-plane containers..."
(cd "$SLURM_DEV_DIR" && "${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" up -d)

# ---------- Step 2: Start host slurmd ----------
log ""
log "Step 2: Starting host slurmd..."
if ! pgrep -x slurmd >/dev/null 2>&1; then
    if [[ -f "$SLURM_DEV_DIR/scripts/start-host-slurmd.sh" ]]; then
        sudo bash "$SLURM_DEV_DIR/scripts/start-host-slurmd.sh" --bg
    else
        die "start-host-slurmd.sh not found. Please start slurmd manually."
    fi
else
    log "Host slurmd already running (PID: $(pgrep -x slurmd | head -n 1))"
fi

# ---------- Step 3: Wait for services ----------
log ""
log "Step 3: Waiting for services to become ready..."

wait_for_condition 60 5 "ai-login" \
    "podman inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}running{{end}}' $AI_LOGIN_CONTAINER | grep -qiE 'healthy|running'" || \
    die "ai-login did not become healthy"

wait_for_condition 60 5 "Slurm node registration" \
    "podman exec $AI_LOGIN_CONTAINER sinfo -N -h -o '%N %T' | grep -q '^compute-01'" || \
    die "compute-01 never appeared in sinfo"

# Resume node if drained/down from prior run
node_state="$(podman exec "$AI_LOGIN_CONTAINER" sinfo -N -h -o '%T' -n compute-01 | head -n 1 | tr -d '* ')"
case "$node_state" in
    idle|mixed|allocated)
        log "compute-01 state: $node_state (OK)"
        ;;
    *)
        log "compute-01 state: $node_state — resuming..."
        podman exec "$AI_LOGIN_CONTAINER" scontrol update NodeName=compute-01 State=RESUME Reason="auto-resumed by start_all"
        # Wait for the node to actually become usable
        wait_for_condition 12 5 "compute-01 resume" \
            "podman exec $AI_LOGIN_CONTAINER sinfo -N -h -o '%T' -n compute-01 | tr -d '* ' | grep -qE '^(idle|mixed|allocated)$'" || \
            die "compute-01 did not resume (still $(podman exec "$AI_LOGIN_CONTAINER" sinfo -N -h -o '%T' -n compute-01 | tr -d '* ')). Check: podman exec ai-login sinfo -N"
        ;;
esac

# Release any held jobs left over from prior runs
held_jobs="$(podman exec "$AI_LOGIN_CONTAINER" squeue -h -t PD -o '%i %r' 2>/dev/null | grep -i 'held' | awk '{print $1}' || true)"
if [[ -n "$held_jobs" ]]; then
    for jid in $held_jobs; do
        log "Releasing held job $jid..."
        podman exec "$AI_LOGIN_CONTAINER" scontrol release "$jid"
    done
fi

# ---------- Done ----------
log ""
log "All services started successfully."
log ""
log "Quick checks:"
log "  podman exec ai-login sinfo -N"
log "  umrs-client -c <your_config.json> --server-ip 127.0.0.1 ping"
