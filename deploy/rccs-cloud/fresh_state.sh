#!/usr/bin/env bash
# =============================================================================
# Reset UMRS to a fresh state — server-side and/or client-side.
# =============================================================================
# Stops all services, wipes Postgres + MinIO data, removes user configs,
# cancels Slurm jobs, and cleans demo output.  After this, re-run
# setup_server.sh + start.sh (server) and setup_client.sh (client) to restore.
#
# Usage (on rccs-cloud, for server reset):
#   ./fresh_state.sh --server          # wipe server data only
#
# Usage (on local Mac, for client reset):
#   ./fresh_state.sh --client          # wipe local client config only
#
# Usage (both):
#   ./fresh_state.sh --all             # server + client (must run on server)
#
#   ./fresh_state.sh --dry-run --server  # preview what would be deleted
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${HOME}/.config/umrs"
DRY_RUN=false
DO_SERVER=false
DO_CLIENT=false

usage() {
    cat <<'EOF'
Usage: ./fresh_state.sh [--server|--client|--all] [--dry-run]

Reset UMRS to a fresh state.

Modes:
    --server     Wipe server-side data (Postgres, MinIO volumes, user_files,
                 logs, Slurm jobs, tmp/).  Must run on the server host.
    --client     Wipe client-side config (~/.config/umrs/, demo results).
                 Safe to run on local machine.
    --all        Both --server and --client.
    --dry-run    Print what would be deleted without actually deleting.
    -h, --help   Show this help.

After reset:
    Server: ./setup_server.sh && ./start.sh
    Client: ./setup_client.sh && ./ssh_tunnel.sh
EOF
}

log()  { echo "[fresh] $*"; }
warn() { echo "[fresh] WARNING: $*" >&2; }

run_or_dry() {
    if $DRY_RUN; then
        echo "  [dry-run] $*"
    else
        eval "$@"
    fi
}

# ─── Parse arguments ────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server)  DO_SERVER=true ;;
        --client)  DO_CLIENT=true ;;
        --all)     DO_SERVER=true; DO_CLIENT=true ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
done

# ─── Server-side reset ──────────────────────────────────────────────────────
if $DO_SERVER; then
    log "=== Server-side reset ==="

    # 1. Stop all services (reuse existing stop.sh)
    if [[ -f "${SCRIPT_DIR}/stop.sh" ]]; then
        log "Stopping services..."
        run_or_dry "bash '${SCRIPT_DIR}/stop.sh' --clean 2>&1 || true"
    fi

    # 2. Cancel any remaining Slurm jobs
    if command -v squeue &>/dev/null; then
        SLURM_JOBS=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null \
            | grep -iE 'rivault|serve-model|umrs|vllm|litellm|bentoml' \
            | awk '{print $1}' || true)
        if [[ -n "$SLURM_JOBS" ]]; then
            log "Cancelling Slurm jobs: $SLURM_JOBS"
            run_or_dry "scancel $SLURM_JOBS 2>/dev/null || true"
        else
            log "No UMRS-related Slurm jobs found."
        fi
    fi

    # 3. Remove server user_files
    if [[ -d "${SCRIPT_DIR}/user_files" ]]; then
        log "Removing user_files/..."
        run_or_dry "rm -rf '${SCRIPT_DIR}/user_files'"
    fi

    # 4. Remove server logs
    if [[ -d "${SCRIPT_DIR}/logs" ]]; then
        log "Removing logs/..."
        run_or_dry "rm -rf '${SCRIPT_DIR}/logs'"
    fi

    # 5. Remove tmp directory
    if [[ -d "${SCRIPT_DIR}/tmp" ]]; then
        log "Removing tmp/..."
        run_or_dry "rm -rf '${SCRIPT_DIR}/tmp'"
    fi

    # 6. Remove PID files
    run_or_dry "rm -f '${SCRIPT_DIR}/server.pid' '${SCRIPT_DIR}/rag-service.pid'"

    # 7. Remove server-side UMRS config (will be recreated by setup_server.sh)
    if [[ -d "${CONFIG_DIR}" ]]; then
        log "Removing server config (~/.config/umrs/)..."
        run_or_dry "rm -rf '${CONFIG_DIR}'"
    fi

    # 7b. Remove per-user S3 config files (created during provisioning)
    for s3cfg in "${HOME}/.s3cfg" "${HOME}/.s3cfg_"*; do
        if [[ -f "$s3cfg" ]]; then
            log "Removing $s3cfg"
            run_or_dry "rm -f '$s3cfg'"
        fi
    done

    # 8. Clean demo result directories
    log "Cleaning demo output directories..."
    for results_dir in "${REPO_ROOT}/demos"/*/results; do
        if [[ -d "$results_dir" ]]; then
            log "  Removing $results_dir"
            run_or_dry "rm -rf '$results_dir'"
        fi
    done

    log "=== Server-side reset complete ==="
    echo ""
fi

# ─── Client-side reset ──────────────────────────────────────────────────────
if $DO_CLIENT; then
    log "=== Client-side reset ==="

    # 1. Remove client UMRS config
    if [[ -d "${CONFIG_DIR}" ]]; then
        log "Removing client config (~/.config/umrs/)..."
        run_or_dry "rm -rf '${CONFIG_DIR}'"
    fi

    # 2. Remove S3 config files
    for s3cfg in "${HOME}/.s3cfg" "${HOME}/.s3cfg_"*; do
        if [[ -f "$s3cfg" ]]; then
            log "Removing $s3cfg"
            run_or_dry "rm -f '$s3cfg'"
        fi
    done

    # 3. Clean demo result directories (local)
    log "Cleaning demo output directories..."
    for results_dir in "${REPO_ROOT}/demos"/*/results; do
        if [[ -d "$results_dir" ]]; then
            log "  Removing $results_dir"
            run_or_dry "rm -rf '$results_dir'"
        fi
    done

    # 4. Clean workflow results in parent repo (if present)
    PARENT_DEMOS="${REPO_ROOT}/../../demos"
    if [[ -d "$PARENT_DEMOS" ]]; then
        for results_dir in "${PARENT_DEMOS}"/*/results; do
            if [[ -d "$results_dir" ]]; then
                log "  Removing $results_dir"
                run_or_dry "rm -rf '$results_dir'"
            fi
        done
    fi

    log "=== Client-side reset complete ==="
    echo ""
fi

# ─── Summary ────────────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo "Dry run complete — no changes were made."
else
    echo "Fresh state achieved."
    echo ""
    if $DO_SERVER; then
        echo "Server next steps:"
        echo "  1. ./setup_server.sh     (recreate venv, configs)"
        echo "  2. ./start.sh            (launch postgres, minio, server)"
        echo "  3. Enroll users:  source .venv/bin/activate"
        echo "     umrs-server admin user add user1 --admin-token \$UMRS_ADMIN_TOKEN"
        echo ""
    fi
    if $DO_CLIENT; then
        echo "Client next steps:"
        echo "  1. ./setup_client.sh     (copy configs, install umrs)"
        echo "  2. ./ssh_tunnel.sh       (open SSH tunnels)"
        echo "  3. umrs-client ping      (verify connection)"
        echo "  4. cd ../../demos/00_onboarding && bash run_demo.sh"
    fi
fi
