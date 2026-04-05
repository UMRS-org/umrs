#!/usr/bin/env bash
# Demo 10: Opossum Motion Analysis (rccs-cloud deployment)
#
# Opossum analysis engine: ../../tmp/opossum/ (see tmp/opossum/src/opossum/analyze.py)
#
# Uploads video files → triggers pipeline-based motion analysis → downloads TSV results.
# The OpossumAnalyzeStep runs frame-differencing to detect motion events from camera videos.
#
# Optional env vars:
#   UMRS_USER_CONF   — user config  (default: ~/.config/umrs/users/user1.json)
#   GEN_TEST_VIDEOS  — "1" to auto-generate synthetic test videos
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DEPLOY_DIR}/../.." && pwd)"
VENV="${DEPLOY_DIR}/.venv"
export PATH="${VENV}/bin:${PATH}"

CONF="${UMRS_USER_CONF:-${HOME}/.config/umrs/users/user1.json}"
EXPERIENCE="Opossum Analysis"
SAMPLE_DIR="${DEPLOY_DIR}/demos/sample_data/videos"
OPOSSUM_DIR="${REPO_ROOT}/tmp/opossum"

echo "=== Opossum Motion Analysis Demo ==="
echo "Opossum engine: ${OPOSSUM_DIR}"

# Pre-flight
umrs-client -c "${CONF}" ping || { echo "ERROR: server not reachable"; exit 1; }

if [[ ! -d "${OPOSSUM_DIR}" ]]; then
    echo "WARNING: Opossum engine not found at ${OPOSSUM_DIR}"
    echo "Expected: tmp/opossum/ with src/opossum/analyze.py"
fi

# Optional: generate synthetic test videos
if [[ "${GEN_TEST_VIDEOS:-0}" == "1" ]]; then
    echo "--- Generating test videos ---"
    mkdir -p "${SAMPLE_DIR}"
    if [[ -f "${OPOSSUM_DIR}/src/demo_gen.py" ]]; then
        python3 "${OPOSSUM_DIR}/src/demo_gen.py" \
            --output-dir "${SAMPLE_DIR}" --num-videos 4
        echo "Generated test videos in ${SAMPLE_DIR}"
    else
        echo "WARNING: demo_gen.py not found — place .mp4 files in ${SAMPLE_DIR}"
    fi
fi

# Check for sample videos
if [[ -d "${SAMPLE_DIR}" ]]; then
    VID_COUNT=$(find "${SAMPLE_DIR}" -maxdepth 1 \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
    if [[ "${VID_COUNT}" -gt 0 ]]; then
        echo "Uploading ${VID_COUNT} video(s) from ${SAMPLE_DIR}"
        umrs-client -c "${CONF}" upload-dir "${SAMPLE_DIR}" \
            --experience "${EXPERIENCE}" --modality video
    else
        echo "No videos in ${SAMPLE_DIR} — upload skipped"
        echo "Place .mp4 files there or use: GEN_TEST_VIDEOS=1 bash $0"
        exit 1
    fi
else
    echo "Sample directory ${SAMPLE_DIR} not found."
    echo "Create it with videos or use: GEN_TEST_VIDEOS=1 bash $0"
    exit 1
fi

# Trigger processing
echo "--- Triggering PROCESS_OBJECTS ---"
umrs-client -c "${CONF}" process-objects --experience "${EXPERIENCE}"

# Wait for output
echo "--- Waiting for results ---"
umrs-client -c "${CONF}" wait-output --experience "${EXPERIENCE}"

# Download
RESULT_DIR="${SCRIPT_DIR}/opossum_results"
mkdir -p "${RESULT_DIR}"
echo "--- Downloading to ${RESULT_DIR} ---"
umrs-client -c "${CONF}" download-results --experience "${EXPERIENCE}" -o "${RESULT_DIR}"

echo "=== Opossum Demo Complete ==="
echo "Opossum engine ref: ${OPOSSUM_DIR}"
echo "  Core:  ${OPOSSUM_DIR}/src/opossum/analyze.py"
echo "  Utils: ${OPOSSUM_DIR}/src/opossum/utils.py"
TSV_COUNT=$(find "${RESULT_DIR}" -name "*.tsv" 2>/dev/null | wc -l)
echo "TSV results: ${TSV_COUNT}"
ls -la "${RESULT_DIR}/" 2>/dev/null || true
