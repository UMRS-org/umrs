#!/usr/bin/env bash
# =============================================================================
# Stop all UMRS services
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"
SLURM_DEV_DIR="$DEMO_DIR/slurm-dev"
COMPOSE_FILE="$SLURM_DEV_DIR/docker-compose.yaml"
AI_LOGIN_CONTAINER="ai-login"

LOG_PREFIX="[stop_all]"
log() { echo "$LOG_PREFIX $*"; }

detect_compose() {
    if command -v podman-compose >/dev/null 2>&1; then
        COMPOSE_CMD=(podman-compose)
    elif command -v podman >/dev/null 2>&1 && podman compose version >/dev/null 2>&1; then
        COMPOSE_CMD=(podman compose)
    else
        log "WARNING: No compose command found; trying podman rm"
        return 1
    fi
}

# ---------- Step 1: Cancel Slurm jobs ----------
log "Cancelling active Slurm jobs..."
if podman inspect "$AI_LOGIN_CONTAINER" >/dev/null 2>&1; then
    job_ids="$(podman exec "$AI_LOGIN_CONTAINER" squeue -h -o '%A' 2>/dev/null || true)"
    if [[ -n "$job_ids" ]]; then
        for jid in $job_ids; do
            log "  scancel $jid"
            podman exec "$AI_LOGIN_CONTAINER" scancel "$jid" 2>/dev/null || true
        done
        sleep 3
    fi
fi

# ---------- Step 2: Remove Rivault containers ----------
log "Removing Rivault containers..."
sudo podman ps -qa --filter 'name=rivault_' --format '{{.ID}}' 2>/dev/null | \
    xargs -r sudo podman rm -f 2>/dev/null || true
podman ps -qa --filter 'name=rivault_' --format '{{.ID}}' 2>/dev/null | \
    xargs -r podman rm -f 2>/dev/null || true
sudo podman network rm -f genai-private-net 2>/dev/null || true
podman network rm -f genai-private-net 2>/dev/null || true

# ---------- Step 3: Stop control plane ----------
log "Stopping control-plane containers..."
if detect_compose; then
    (cd "$SLURM_DEV_DIR" && "${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" down 2>/dev/null || true)
fi

# ---------- Step 4: Stop host slurmd ----------
log "Stopping host slurmd and munged..."
sudo killall slurmd 2>/dev/null || true
sudo killall munged 2>/dev/null || true

log "All services stopped."
