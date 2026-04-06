#!/usr/bin/env bash
# =============================================================================
# Shared helpers for UMRS demos
# =============================================================================
# Source this from any demo script:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/../lib/common.sh"
# =============================================================================

# --- Default configuration ---
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONF="${UMRS_CONF:-${UMRS_USER_CONF:-${HOME}/.config/umrs/conf_umrs.json}}"
CLIENT="umrs-client -c ${CONF}"

# --- Logging helpers ---
_TAG="${DEMO_TAG:-demo}"
log()  { echo "[${_TAG}] $*"; }
ok()   { echo "[${_TAG}]   ✓ $1"; }
warn() { echo "[${_TAG}] WARNING: $*"; }
die()  { echo "[${_TAG}] ERROR: $*" >&2; exit 1; }

# --- Banner ---
banner() {
    echo "=========================================="
    echo " $*"
    echo "=========================================="
}

# --- Pre‑flight check ---
preflight() {
    which umrs-client >/dev/null 2>&1 || die "umrs-client not in PATH"
    $CLIENT ping >/dev/null 2>&1    || die "Server not reachable (is the SSH tunnel running?)"
    log "Server is ready."
}

# --- Wait for all tasks to finish (simple polling) ---
# Usage: wait_tasks [max_attempts] [sleep_sec]
wait_tasks() {
    local max="${1:-30}" sleep_sec="${2:-10}"
    for i in $(seq 1 "$max"); do
        local raw
        raw="$($CLIENT status all 2>/dev/null || true)"
        if echo "$raw" | grep -qi "completed\|done"; then
            log "Tasks completed."
            return 0
        fi
        if echo "$raw" | grep -qi "error\|failed"; then
            warn "Some tasks errored — check server logs."
            return 1
        fi
        [[ $((i % 6)) -eq 0 ]] && log "  waiting... ($i/$max)"
        sleep "$sleep_sec"
    done
    die "Timeout waiting for tasks ($max × ${sleep_sec}s)."
}
