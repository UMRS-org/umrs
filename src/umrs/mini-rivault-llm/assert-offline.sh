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

if [ "$FAILED" -ne 0 ]; then
    echo "---"
    echo "FAIL: One or more offline invariants violated. Refusing to start inference."
    exit 1
fi

echo "OK: All offline invariants satisfied."
exit 0
