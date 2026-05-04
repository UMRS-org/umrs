#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

passed=0
failed=0

run_demo() {
    local num="$1" name="$2" script="$3"

    echo ""
    echo "==========================================="
    echo " Jens Demo ${num}: ${name}"
    echo "==========================================="

    if bash "${SCRIPT_DIR}/${script}"; then
        echo "=== Jens Demo ${num} PASSED ==="
        passed=$((passed + 1))
    else
        echo "=== Jens Demo ${num} FAILED ==="
        failed=$((failed + 1))
    fi
}

run_demo "11" "SAM3 Direct" "11_sam3_direct/run_demo.sh"
run_demo "12" "HMAE Direct" "12_hmae_direct/run_demo.sh"

echo ""
echo "==========================================="
echo " Jens Demo Suite Summary"
echo "==========================================="
echo "Passed: ${passed}"
echo "Failed: ${failed}"

if [[ "$failed" -gt 0 ]]; then
    exit 1
fi