#!/usr/bin/env bash
# assert-offline.sh — Fail-fast if any offline guarantee is missing.
#
# Must run before launching vLLM/LiteLLM inside mini-RiVault containers.
# Exit 0 = all offline invariants satisfied. Exit 1 = one or more violated.
#
# Used by: serve-model.yaml variants, serve-model.slurm, any container
# that will perform offline inference (TRANSFORMERS_OFFLINE=1 required).

set -euo pipefail

FAILED=0

# 1. TRANSFORMERS_OFFLINE must be "1"
if [ "${TRANSFORMERS_OFFLINE:-0}" != "1" ]; then
    echo "FAIL: TRANSFORMERS_OFFLINE is ${TRANSFORMERS_OFFLINE:-not set} (expected 1)"
    FAILED=1
fi

# 2. HF_HUB_OFFLINE must be "1"
if [ "${HF_HUB_OFFLINE:-0}" != "1" ]; then
    echo "FAIL: HF_HUB_OFFLINE is ${HF_HUB_OFFLINE:-not set} (expected 1)"
    FAILED=1
fi

# 3. Check that no http_proxy/https_proxy env vars are set
for var in http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY; do
    if [ -n "${!var:-}" ]; then
        echo "FAIL: $var is set to '${!var}' — proxy must be empty for offline mode"
        FAILED=1
    fi
done

# 4. Runtime probe (ADR-040, opt-in): a DIRECT TCP connect to a public host
#    must FAIL inside an offline inference container. Env/proxy variables only
#    describe intent; this observes the network. Opt-in with OFFLINE_PROBE=1
#    until the HPC e2e leg (tests/run_hpc_tests.sh --slurm-egress-smoke) has
#    run on every venue; then the default flips to on.
_offline_probe_connects() {
    # exit 0 = connected (egress OPEN), exit 1 = refused/unreachable (offline)
    local host="$1" port="$2" tmo="$3"
    if command -v python3 >/dev/null 2>&1; then
        python3 - "$host" "$port" "$tmo" <<'PY'
import socket
import sys

host, port, tmo = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
try:
    with socket.create_connection((host, port), timeout=tmo):
        sys.exit(0)
except OSError:
    sys.exit(1)
PY
        return $?
    fi
    timeout "$tmo" bash -c "exec 3<>/dev/tcp/${host}/${port}" 2>/dev/null
}

if [ "${OFFLINE_PROBE:-0}" = "1" ]; then
    _probe_host="${OFFLINE_PROBE_HOST:-example.com}"
    _probe_port="${OFFLINE_PROBE_PORT:-443}"
    _probe_timeout="${OFFLINE_PROBE_TIMEOUT_SEC:-3}"
    if _offline_probe_connects "$_probe_host" "$_probe_port" "$_probe_timeout"; then
        echo "FAIL: direct egress to ${_probe_host}:${_probe_port} SUCCEEDED — container is not offline"
        FAILED=1
    else
        echo "OK: direct egress to ${_probe_host}:${_probe_port} is blocked"
    fi
fi

if [ "$FAILED" -ne 0 ]; then
    echo "---"
    echo "FAIL: One or more offline invariants violated. Refusing to start inference."
    exit 1
fi

echo "OK: All offline invariants satisfied."
exit 0
