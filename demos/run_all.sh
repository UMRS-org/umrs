#!/usr/bin/env bash
# =============================================================================
# Run All Core UMRS Demos
# =============================================================================
# Sequentially runs all demos that require a live server connection.
# Skips optional demos (vision, RAG) if the required services are unavailable.
#
# Works with any deployment (Docker Compose, rccs-cloud, hpbd-atom, etc.)
# as long as the UMRS server is reachable.
#
# Prerequisites:
#   - UMRS server running
#   - Network connectivity (SSH tunnels if remote)
#   - umrs-client installed and in PATH
#   - User config registered via `umrs-client config init` (or UMRS_CONF env var)
#
# Usage:
#   bash run_all.sh                      # run all core demos
#   bash run_all.sh --skip-vision        # skip SAM3/HMAE (no GPU needed)
#   DEMO_DOWNLOAD=1 bash run_all.sh      # also run HuggingFace download
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_VISION="${1:-}"

passed=0
failed=0
skipped=0

run_demo() {
    local num="$1" name="$2" script="$3"
    shift 3
    local extra_check="${1:-}"

    echo ""
    echo "==========================================="
    echo " Demo ${num}: ${name}"
    echo "==========================================="

    # Optional pre-check (skip demo if check fails)
    if [[ -n "$extra_check" ]]; then
        if ! eval "$extra_check" >/dev/null 2>&1; then
            echo "SKIP: prerequisite not met (${extra_check})"
            skipped=$((skipped + 1))
            return 0
        fi
    fi

    if bash "${SCRIPT_DIR}/${script}" 2>&1; then
        echo "=== Demo ${num} PASSED ==="
        passed=$((passed + 1))
    else
        echo "=== Demo ${num} FAILED ==="
        failed=$((failed + 1))
    fi
}

# ---------- Pre-flight ----------
echo "=== UMRS Demo Suite Pre-flight ==="
which umrs-client >/dev/null 2>&1 || { echo "ERROR: umrs-client not found in PATH"; exit 1; }

CONF="${UMRS_CONF:-${UMRS_USER_CONF:-${HOME}/.config/umrs/conf_umrs.json}}"
if [[ -f "$CONF" ]]; then
    umrs-client config use --user "$CONF" >/dev/null 2>&1 || true
fi

umrs-client ping >/dev/null 2>&1 || { echo "ERROR: Server not reachable. Check connectivity."; exit 1; }
echo "Pre-flight OK — server reachable."

# ---------- Onboarding ----------
run_demo "00" "User Onboarding"     "00_onboarding/run_demo.sh"

# ---------- Core demos ----------
run_demo "01" "Health Check"        "01_health_check/run_demo.sh"
run_demo "02" "Text Embedding"      "02_text_embedding/run_demo.sh"

# Vision demos (skip with --skip-vision)
if [[ "$SKIP_VISION" != "--skip-vision" ]]; then
    run_demo "03" "SAM3 Segmentation"  "03_sam3_segmentation/run_demo.sh"
    run_demo "04" "Vision Inference"   "04_vision_inference/run_demo.sh"
    run_demo "09" "Multi-Step Vision"  "09_multistep_vision_pipeline/run_demo.sh"
    run_demo "10" "Multi-Mode Vision"  "10_vision_multi_mode/run_demo.sh"
else
    echo ""
    echo "  Skipping vision demos (--skip-vision)"
    skipped=$((skipped + 4))
fi

# RAG (requires RAG service)
run_demo "05" "RAG Search"           "05_rag_search/run_demo.sh" \
    "curl -sf http://localhost:8081/docs"

run_demo "06" "Server Maintenance"   "06_server_maintenance/run_demo.sh"
run_demo "07" "Model Registry"       "07_model_registry/run_demo.sh"

# Local demos (don't need a running server)
run_demo "08" "User Model (local)"   "08_user_model/run_demo.sh"
run_demo "12" "Pipeline SDK (local)" "12_pipeline_sdk/run_demo.sh"

# Advanced server demos
run_demo "11" "Job Submission"       "11_job_submission/run_demo.sh"
run_demo "15" "Complex DAG (local)"  "15_complex_dag/run_demo.sh"
run_demo "16" "Custom Workflow"      "16_custom_workflow/run_demo.sh"
run_demo "18" "Model Lifecycle"      "18_model_lifecycle/run_demo.sh"
run_demo "20" "Durable Jobs"         "20_durable_jobs/run_demo.sh"

# Auth enforcement (local token management; server-side auth requires UMRS_ADMIN_TOKEN)
run_demo "23" "Auth Enforcement"     "23_auth_enforcement/run_demo.sh"

# Thread pool metrics (read-only status query)
run_demo "24" "Thread Pool"          "24_thread_pool/run_demo.sh"

# Video analysis (requires opencv — CPU only)
run_demo "14" "Video Analysis"       "14_video_analysis/run_demo.sh"

# ---------- Summary ----------
echo ""
echo "==========================================="
echo " ALL DEMOS COMPLETE"
echo "==========================================="
echo "Passed:  ${passed}"
echo "Failed:  ${failed}"
echo "Skipped: ${skipped}"

if [[ "$failed" -gt 0 ]]; then
    echo ""
    echo "Some demos failed. Check the output above for details."
    exit 1
fi
