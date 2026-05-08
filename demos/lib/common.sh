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

UMRS_URL="${UMRS_URL:-${UMRS_HTTP_URL:-http://localhost:8001}}"

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

maybe_admin_reload() {
    if [[ "${DEMO_ADMIN_ACTIONS:-0}" = "1" ]]; then
        admin_reload "$@"
    else
        log "Skipping admin reload — demos assume the site/operator already activated the required config."
        return 0
    fi
}

maybe_admin_recompute() {
    if [[ "${DEMO_ADMIN_ACTIONS:-0}" = "1" ]]; then
        admin_recompute "$@"
    else
        warn "Ignoring recompute request in general-user mode. Recompute is an operator/admin workflow, not part of the normal user demos."
        return 0
    fi
}

# --- Banner ---
banner() {
    echo "=========================================="
    echo " $*"
    echo "=========================================="
}

# --- Pre‑flight check ---
client_ping() {
    UMRS_PING_TIMEOUT="${UMRS_PING_TIMEOUT:-20}" python3 - "${CLIENT_ARGS[@]}" <<'PY'
import os
import subprocess
import sys

timeout = float(os.environ.get("UMRS_PING_TIMEOUT", "20"))
cmd = sys.argv[1:] + ["ping"]

try:
    completed = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
    )
except subprocess.TimeoutExpired:
    raise SystemExit(124)

raise SystemExit(completed.returncode)
PY
}

preflight() {
    which umrs-client >/dev/null 2>&1 || die "umrs-client not in PATH"
    client_ping || die "Server not reachable within ${UMRS_PING_TIMEOUT:-20}s (is the SSH tunnel running?)"
    log "Server is ready."
}

run_client_bounded() {
    local client_timeout="$1"
    shift
    CLIENT_TIMEOUT="$client_timeout" python3 - "${CLIENT_ARGS[@]}" "$@" <<'PY'
import os
import subprocess
import sys

timeout = float(os.environ["CLIENT_TIMEOUT"])
cmd = sys.argv[1:]

try:
    completed = subprocess.run(cmd, timeout=timeout, check=False)
except subprocess.TimeoutExpired:
    print(f"command timed out after {timeout:g}s: {' '.join(cmd)}", file=sys.stderr)
    raise SystemExit(124)

raise SystemExit(completed.returncode)
PY
}

download_results_bounded() {
    run_client_bounded "${RESULT_DOWNLOAD_TIMEOUT:-20}" download_results "$@"
}

http_api_available() {
    local url="${1:-${UMRS_URL}}"
    curl -sf "${url%/}/openapi.json" >/dev/null 2>&1 \
        || curl -sf "${url%/}/docs" >/dev/null 2>&1
}

extract_json_payload() {
    python3 -c '
import json
import sys

text = sys.stdin.read()
decoder = json.JSONDecoder()
for index, char in enumerate(text):
    if char not in "[{":
        continue
    try:
        payload, _ = decoder.raw_decode(text[index:])
    except json.JSONDecodeError:
        continue
    print(json.dumps(payload))
    raise SystemExit(0)

print("{}")
'
}

prepare_mini_rivault_mount_demo() {
        local mount_root="${MINI_RIVAULT_MOUNT_DEMO_ROOT:-${SCRIPT_DIR}/results/mounts}"
        DEMO_MCP_SOURCE="${DEMO_MCP_SOURCE:-${mount_root}/mcp/demo-tools}"
        DEMO_PROJECT_DATA_SOURCE="${DEMO_PROJECT_DATA_SOURCE:-${mount_root}/project-data}"
        DEMO_RAG_INDEX_SOURCE="${DEMO_RAG_INDEX_SOURCE:-${mount_root}/rag-index}"
        DEMO_EGRESS_DOMAINS="${DEMO_EGRESS_DOMAINS:-api.example.com}"

        mkdir -p "${DEMO_MCP_SOURCE}" "${DEMO_PROJECT_DATA_SOURCE}" "${DEMO_RAG_INDEX_SOURCE}"

        cat > "${DEMO_MCP_SOURCE}/config.json" <<'JSON'
{
    "name": "umrs-demo-mounted-tools",
    "version": "0.1.0",
    "capabilities": {
        "tools": ["summarize_mount"],
        "tool_capabilities": {
            "summarize_mount": ["READ_LOCAL_FILE"]
        },
        "egress": {
            "domains": ["api.example.com"],
            "protocols": ["https"]
        }
    },
    "tool_descriptions": {
        "summarize_mount": "Summarize demo project data mounted inside mini-RiVault."
    },
    "tool_schemas": {
        "summarize_mount": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Optional topic to emphasize in the summary."
                }
            }
        }
    },
    "entry_point": "python3 run_tool.py",
    "sandbox": {
        "network": "egress_guard",
        "filesystem": "read_only",
        "max_memory_mb": 256,
        "max_runtime_sec": 10
    }
}
JSON

        cat > "${DEMO_MCP_SOURCE}/run_tool.py" <<'PY'
from __future__ import annotations

import json
import sys

payload = json.load(sys.stdin)
arguments = payload.get("arguments", {}) if isinstance(payload, dict) else {}
topic = arguments.get("topic") or "UMRS mini-RiVault mounts"
print(json.dumps({
        "topic": topic,
        "summary": "UMRS passed custom MCP, project data, RAG index, and egress intent into mini-RiVault.",
}))
PY

        cat > "${DEMO_PROJECT_DATA_SOURCE}/demo-corpus.txt" <<'TEXT'
UMRS demo project corpus.

This directory is mounted read-only at /project inside mini-RiVault. RAG and
custom MCP services consume it inside the endpoint boundary rather than through
a standalone compute-node MCP launcher.
TEXT

        cat > "${DEMO_RAG_INDEX_SOURCE}/manifest.json" <<'JSON'
{
    "name": "umrs-demo-rag-index",
    "description": "Placeholder pre-built RAG index mounted at /rag-index for mini-RiVault endpoint demos.",
    "documents": ["demo-corpus.txt"]
}
JSON
}

mini_rivault_mount_payload() {
        prepare_mini_rivault_mount_demo
        python3 - <<PY
import json

payload = {
        "mcp_servers": [
                {"source": "${DEMO_MCP_SOURCE}", "target": "/mcp-servers/demo-tools"}
        ],
        "project_data": {"source": "${DEMO_PROJECT_DATA_SOURCE}", "target": "/project"},
        "rag_index": {"source": "${DEMO_RAG_INDEX_SOURCE}", "target": "/rag-index"},
}
print(json.dumps(payload, indent=2))
PY
}

print_mini_rivault_mount_summary() {
        prepare_mini_rivault_mount_demo
        log "Mini-RiVault mount request:"
        echo "  MCP server : ${DEMO_MCP_SOURCE} -> /mcp-servers/demo-tools"
        echo "  Project data: ${DEMO_PROJECT_DATA_SOURCE} -> /project"
        echo "  RAG index   : ${DEMO_RAG_INDEX_SOURCE} -> /rag-index"
        echo "  Egress      : ${DEMO_EGRESS_DOMAINS}"
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

    local status_output
    status_output="$($client_cmd status all 2>&1)" || die "status failed for ${conf}"
    while IFS= read -r line; do log "  $line"; done \
        <<< "$(printf '%s\n' "$status_output" | awk 'NR <= 20')"
    ok "status summary OK"

    local bucket_output
    bucket_output="$($client_cmd list_buckets 2>&1)" || die "list_buckets failed for ${conf}"
    while IFS= read -r line; do log "  $line"; done \
        <<< "$(printf '%s\n' "$bucket_output" | awk 'NR <= 10')"
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
