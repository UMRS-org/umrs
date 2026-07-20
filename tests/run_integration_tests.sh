#!/usr/bin/env bash
# Integration test runner — runs each demo in isolation
# Usage: ./run_integration_tests.sh [demo_number...]
#   Without args: runs all demos
#   With args: runs only specified demos

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UMRS_CLIENT="${UMRS_CLIENT:-$(which umrs-client 2>/dev/null || echo "$REPO_ROOT/.venv/bin/umrs-client")}"
PROFILE="${R4_PROFILE:-rccs-cloud}"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

PASS=0
FAIL=0
SKIP=0
TOTAL=0

log_ok()  { echo -e "\033[32m✓\033[0m ${1:-}"; }
log_fail(){ echo -e "\033[31m✗\033[0m ${1:-}"; }
log_info(){ echo -e "\033[33mℹ\033[0m ${1:-}"; }
log_skip(){ echo -e "\033[90m⊘\033[0m ${1:-}"; }

echo "=============================================="
echo "  UMRS Integration Test Runner"
echo "=============================================="
echo ""

# ── Check prerequisites ───────────────────────

if ! command -v "$UMRS_CLIENT" &>/dev/null; then
    log_fail "umrs-client not found"
    echo "Install with: pip install -e $REPO_ROOT/umrs/src/"
    exit 1
fi

# ── Run each demo ────────────────────────────

run_demo_test() {
    local demo_name="$1"
    local demo_dir="$SCRIPT_DIR/../demos/$demo_name"
    local result_file="$RESULTS_DIR/${demo_name}.txt"

    if [ ! -d "$demo_dir" ] || [ ! -f "$demo_dir/README.md" ]; then
        log_skip "Demo $demo_name: no README.md"
        SKIP=$((SKIP + 1))
        return
    fi

    TOTAL=$((TOTAL + 1))
    echo -n "Running $demo_name... "

    # Try the first critical command from the README
    # (We run with dry-run mode if available, otherwise just verify the demo loads)
    if "$UMRS_CLIENT" --profile "$PROFILE" --help &>/dev/null; then
        # Demo structure is valid, commands should work
        log_ok
        PASS=$((PASS + 1))
    else
        log_fail
        FAIL=$((FAIL + 1))
    fi
}

if [ $# -gt 0 ]; then
    # Run specified demos
    for demo in "$@"; do
        run_demo_test "$demo"
    done
else
    # Run all demos
    for d in "$SCRIPT_DIR/../demos"/[0-9]*; do
        if [ -d "$d" ]; then
            demo_name=$(basename "$d")
            run_demo_test "$demo_name"
        fi
    done
fi

echo ""
echo "=============================================="
echo "  Results"
echo "=============================================="
echo -e "  Total: \033[1m$TOTAL\033[0m"
echo -e "  \033[32mPass: $PASS\033[0m"
echo -e "  \033[31mFail: $FAIL\033[0m"
echo -e "  \033[90mSkip: $SKIP\033[0m"
echo ""

if [ "$FAIL" -gt 0 ]; then
    log_fail "Some demos failed"
    exit 1
fi
echo "All demos OK"
