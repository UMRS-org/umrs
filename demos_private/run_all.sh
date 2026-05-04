#!/usr/bin/env bash
# =============================================================================
# Run All Private UMRS Demos
# =============================================================================
# Runs the active private/proprietary demos that are available in this checkout.
# The Jens direct stack is isolated under demos_jens/ and intentionally omitted
# from the current suite.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="private_run_all" source "${SCRIPT_DIR}/../demos/lib/common.sh"

passed=0
failed=0
skipped=0

run_demo() {
    local num="$1" name="$2" script="$3"
    shift 3
    local extra_check="${1:-}"

    echo ""
    echo "==========================================="
    echo " Private Demo ${num}: ${name}"
    echo "==========================================="

    if [[ -n "$extra_check" ]]; then
        if ! eval "$extra_check" >/dev/null 2>&1; then
            echo "SKIP: prerequisite not met (${extra_check})"
            skipped=$((skipped + 1))
            return 0
        fi
    fi

    if bash "${SCRIPT_DIR}/${script}" 2>&1; then
        echo "=== Private Demo ${num} PASSED ==="
        passed=$((passed + 1))
    else
        echo "=== Private Demo ${num} FAILED ==="
        failed=$((failed + 1))
    fi
}

check_model_visible() {
    local name="$1"
    if [[ "${PRIVATE_VISION_REQUIRE_MODEL:-1}" = "0" ]]; then
        return 0
    fi

    model_list_contains system "$name" || model_list_contains user "$name"
}

model_list_contains() {
    local scope="$1" name="$2"
    MODEL_SCOPE="$scope" MODEL_NAME="$name" MODEL_CHECK_TIMEOUT="${MODEL_CHECK_TIMEOUT:-8}" \
        python3 - "${CLIENT_ARGS[@]}" <<'PY'
import json
import os
import subprocess
import sys

scope = os.environ["MODEL_SCOPE"]
target = os.environ["MODEL_NAME"]
timeout = float(os.environ.get("MODEL_CHECK_TIMEOUT", "8"))
cmd = sys.argv[1:] + ["--output", "json", "model", "list", "--scope", scope]

try:
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=timeout,
        check=False,
    )
except subprocess.TimeoutExpired:
    raise SystemExit(1)

text = completed.stdout
decoder = json.JSONDecoder()
payload = None
for index, char in enumerate(text):
    if char not in "[{":
        continue
    try:
        payload, _ = decoder.raw_decode(text[index:])
    except json.JSONDecodeError:
        continue
    break

def names(value):
    if isinstance(value, dict):
        if isinstance(value.get("name"), str):
            yield value["name"]
        for child in value.values():
            yield from names(child)
    elif isinstance(value, list):
        for item in value:
            yield from names(item)

raise SystemExit(0 if payload is not None and target in set(names(payload)) else 1)
PY
}

check_sam3_model() {
    check_model_visible "${SAM3_MODEL_NAME:-sam3}"
}

check_hmae_model() {
    check_model_visible "${HMAE_MODEL_NAME:-hmae}" \
        || check_model_visible "${HMAE_MODEL_NAME:-hmae-pretrained}"
}

echo "=== UMRS Private Demo Suite Pre-flight ==="
which umrs-client >/dev/null 2>&1 || { echo "ERROR: umrs-client not found in PATH"; exit 1; }
"${CLIENT_ARGS[@]}" ping >/dev/null 2>&1 || { echo "ERROR: Server not reachable. Check connectivity."; exit 1; }
echo "Pre-flight OK — server reachable."

run_demo "04" "HMAE Inference"       "04_hmae_inference/run_demo.sh"
run_demo "09" "Defect Analysis"      "09_defect_analysis/run_demo.sh"
run_demo "10" "HMAE Full Pipeline"   "10_hmae_full_pipeline/run_demo.sh"
run_demo "14" "Opossum Analysis"     "14_opossum_analysis/run_demo.sh"

echo ""
echo "==========================================="
echo " PRIVATE DEMOS COMPLETE"
echo "==========================================="
echo "Passed:  ${passed}"
echo "Failed:  ${failed}"
echo "Skipped: ${skipped}"

if [[ "$failed" -gt 0 ]]; then
    echo ""
    echo "Some private demos failed. Check the output above for details."
    exit 1
fi