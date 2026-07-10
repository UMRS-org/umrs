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

# 2. HF_HUB_DISABLE_HTTPS must be set (prevent huggingface hub HTTP calls)
if [ "${HF_HUB_DISABLE_HTTPS:-0}" != "1" ]; then
    echo "FAIL: HF_HUB_DISABLE_HTTPS is ${HF_HUB_DISABLE_HTTPS:-not set} (expected 1)"
    FAILED=1
fi

# 3. SYMFONY_NO_VALIDATE_URL must be set (prevent HTTP validation)
if [ "${SYMFONY_NO_VALIDATE_URL:-0}" != "1" ]; then
    echo "FAIL: SYMFONY_NO_VALIDATE_URL is ${SYMFONY_NO_VALIDATE_URL:-not set} (expected 1)"
    FAILED=1
fi

# 4. Check that /etc/resolv.conf exists and points to localhost or private DNS
#    (best-effort; may not apply to all container runtimes)
if [ -f /etc/resolv.conf ]; then
    # If nameservers are public (not 10.x, 172.16-31.x, 192.168.x, or 127.x), warn
    while IFS= read -r line; do
        if echo "$line" | grep -qE '^nameserver\s'; then
            ns=$(echo "$line" | awk '{print $2}')
            if ! echo "$ns" | grep -qE '^(127\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)'; then
                echo "WARN: Public DNS nameserver detected: $ns"
                # Don't fail on this — some runtimes require external DNS for package installs
            fi
        fi
    done < /etc/resolv.conf
fi

# 5. Check that no http_proxy/https_proxy env vars are set
for var in http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY; do
    if [ -n "${!var:-}" ]; then
        echo "FAIL: $var is set to '${!var}' — proxy must be empty for offline mode"
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "---"
    echo "FAIL: One or more offline invariants violated. Refusing to start inference."
    exit 1
fi

echo "OK: All offline invariants satisfied."
exit 0
