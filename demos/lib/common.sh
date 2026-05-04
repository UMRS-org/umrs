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

_apply_umrs_profile_defaults() {
    command -v umrs-client >/dev/null 2>&1 || return 0

    local connection_args=()
    if [[ -n "${UMRS_PROFILE:-}" ]]; then
        connection_args+=(--profile "${UMRS_PROFILE}")
    fi
    if [[ -n "${UMRS_ZMQ_HOST:-}" ]]; then
        connection_args+=(--server-ip "${UMRS_ZMQ_HOST}")
    fi
    if [[ -n "${UMRS_ZMQ_PORT:-}" ]]; then
        connection_args+=(--control-port "${UMRS_ZMQ_PORT}")
    fi

    local profile_json
    if [[ ${#connection_args[@]} -gt 0 ]]; then
        if ! profile_json="$(umrs-client "${connection_args[@]}" --output json profile show 2>/dev/null)"; then
            return 0
        fi
    else
        if ! profile_json="$(umrs-client --output json profile show 2>/dev/null)"; then
            return 0
        fi
    fi

    eval "$(PROFILE_JSON="${profile_json}" python - <<'PY'
import json
import os
import shlex
from urllib.parse import urlparse

payload = json.loads(os.environ["PROFILE_JSON"])
connection = payload.get("connection") or {}

minio = urlparse(connection.get("minio_url") or "http://127.0.0.1:9000")
rag = urlparse(connection.get("rag_url") or "http://127.0.0.1:8081")

exports = {
    "UMRS_ZMQ_HOST": connection.get("server_host") or "127.0.0.1",
    "UMRS_ZMQ_PORT": str(connection.get("control_port") or 5555),
    "UMRS_ZMQ_DATA_PORT": str(connection.get("data_base_port") or 5556),
    "UMRS_MINIO_HOST": minio.hostname or "127.0.0.1",
    "UMRS_MINIO_PORT": str(minio.port or 9000),
    "RAG_URL": connection.get("rag_url") or "http://127.0.0.1:8081",
}

for key, value in exports.items():
    print(f"export {key}={shlex.quote(value)}")
PY
)"
}

_apply_umrs_profile_defaults

if [[ -n "${UMRS_CONF:-}" ]]; then
    CONF="${UMRS_CONF}"
elif [[ -n "${UMRS_USER_CONF:-}" ]]; then
    CONF="${UMRS_USER_CONF}"
elif [[ -f "${HOME}/.config/umrs/users/user1.json" ]]; then
    CONF="${HOME}/.config/umrs/users/user1.json"
else
    CONF="${HOME}/.config/umrs/conf_umrs.json"
fi

# Persist the resolved config so subsequent commands don't need -c.
if command -v umrs-client >/dev/null 2>&1 && [[ -f "${CONF}" ]]; then
    umrs-client config use --user "${CONF}" >/dev/null 2>&1 || true
fi

CLIENT_ARGS=(umrs-client)
if [[ -n "${UMRS_PROFILE:-}" ]]; then
    CLIENT_ARGS+=(--profile "${UMRS_PROFILE}")
fi
if [[ -n "${UMRS_ZMQ_HOST:-}" ]]; then
    CLIENT_ARGS+=(--server-ip "${UMRS_ZMQ_HOST}")
fi
if [[ -n "${UMRS_ZMQ_PORT:-}" ]]; then
    CLIENT_ARGS+=(--control-port "${UMRS_ZMQ_PORT}")
fi
CLIENT="${CLIENT_ARGS[*]}"

ADMIN_CLIENT_ARGS=("${CLIENT_ARGS[@]}")
if [[ -n "${UMRS_ADMIN_TOKEN:-}" ]]; then
    ADMIN_CLIENT_ARGS+=(--admin-token "${UMRS_ADMIN_TOKEN}")
fi

# --- Logging helpers ---
_TAG="${DEMO_TAG:-demo}"
log()  { echo "[${_TAG}] $*"; }
ok()   { echo "[${_TAG}]   ✓ $1"; }
warn() { echo "[${_TAG}] WARNING: $*"; }
die()  { echo "[${_TAG}] ERROR: $*" >&2; exit 1; }

admin_hint() {
    if [[ -z "${UMRS_ADMIN_TOKEN:-}" ]]; then
        warn "UMRS_ADMIN_TOKEN is not set; admin-only commands may be denied on auth-enforced deployments."
    fi
}

admin_reload() {
    admin_hint
    "${ADMIN_CLIENT_ARGS[@]}" reload "$@"
}

admin_check() {
    admin_hint
    "${ADMIN_CLIENT_ARGS[@]}" check "$@"
}

admin_update() {
    admin_hint
    "${ADMIN_CLIENT_ARGS[@]}" update "$@"
}

admin_recompute() {
    admin_hint
    "${ADMIN_CLIENT_ARGS[@]}" recompute "$@"
}

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

# --- Onboard active user (idempotent) ---
# Triggers server-side provisioning (bucket + DB creation) for the active
# user config.  Safe to call repeatedly — the server treats reload as a
# no-op when the config hasn't changed.
#
# Usage:
#   onboard              # onboard $CONF (default user)
#   onboard "$OTHER_CONF" # onboard an explicit config
onboard() {
    local conf="${1:-${CONF}}"
    local client_cmd="${CLIENT_ARGS[*]}"
    if [[ "$conf" != "$CONF" ]]; then
        # Use config use to register the explicit config, then use base CLIENT_ARGS.
        umrs-client config use --user "${conf}" >/dev/null 2>&1 || true
    fi

    log "Onboarding user config: ${conf}"
    admin_hint
    "${ADMIN_CLIENT_ARGS[@]}" reload >/dev/null 2>&1 \
        || die "reload failed for ${conf}"
    ok "reload succeeded"

    $client_cmd status all 2>&1 \
        | head -20 | while IFS= read -r line; do log "  $line"; done
    ok "status summary OK"

    $client_cmd list_buckets 2>&1 \
        | head -10 | while IFS= read -r line; do log "  $line"; done
    ok "bucket listing OK"
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
