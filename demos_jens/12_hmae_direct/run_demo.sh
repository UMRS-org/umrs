#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="jens_hmae" source "${SCRIPT_DIR}/../../demos/lib/common.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
HMAE_MODE="${HMAE_MODE:-all}"
MAX_IMAGES="${MAX_IMAGES:-1}"
RESULTS_DIR="${SCRIPT_DIR}/results/${RUN_ID}"
GENERATED_EXPERIENCE="${RESULTS_DIR}/experience.json"

if [[ -z "${IMAGE_DIR:-}" ]]; then
    if find "${SCRIPT_DIR}/../internal_genAI/bento/hmaeinfer/spring8concrete" -maxdepth 1 -type f -name "*.png" 2>/dev/null | grep -q .; then
        IMAGE_DIR="${SCRIPT_DIR}/../internal_genAI/bento/hmaeinfer/spring8concrete"
    else
        IMAGE_DIR="${SCRIPT_DIR}/../../demos/sample_data/images"
    fi
fi

mode_task() {
    case "$1" in
        mae) echo "mae_reconstruction" ;;
        reconstruct) echo "reconstruction" ;;
        features) echo "feature_extraction" ;;
        *) die "Unknown HMAE mode: $1" ;;
    esac
}

mode_alias() {
    case "$1" in
        mae) echo "jens-hmae-mae" ;;
        reconstruct) echo "jens-hmae-reconstruct" ;;
        features) echo "jens-hmae-features" ;;
        *) die "Unknown HMAE mode: $1" ;;
    esac
}

selected_modes() {
    case "${HMAE_MODE}" in
        all) printf '%s\n' mae reconstruct features ;;
        mae|reconstruct|features) printf '%s\n' "${HMAE_MODE}" ;;
        *) die "HMAE_MODE must be all, mae, reconstruct, or features" ;;
    esac
}

generate_experience() {
    mkdir -p "${RESULTS_DIR}"
    python3 - "${GENERATED_EXPERIENCE}" <<'PY'
import json
import sys

path = sys.argv[1]
modes = [
    ("mae", "jens-hmae-mae", "mae_reconstruction"),
    ("reconstruct", "jens-hmae-reconstruct", "reconstruction"),
    ("features", "jens-hmae-features", "feature_extraction"),
]
payload = {
    "xp_name": "Jens HMAE Direct",
    "sql_username": "jens_hmae_direct",
    "modalities": [],
}
for mode, bucket_alias, task in modes:
    payload["modalities"].append(
        {
            "modality_name": "image",
            "task": task,
            "s3": {
                "bucket_alias": bucket_alias,
                "need_output_bucket": True,
                "data": {"type": "file", "mime_type": "image/png"},
            },
            "model": {
                "model_name": f"internal-genai-hmae-{mode}",
                "vector_size": "768",
                "model_type": "rivault://bentoml://http://localhost:{LOCAL_PORT}/vision/infer/submit",
                "vision_model_type": "hmae",
                "vision_model_config": {"task": task},
                "return_type": "file",
            },
            "sql": {
                "store_embeddings": "no",
                "table_name": f"jens_hmae_{mode}",
                "meta_data": [],
            },
        }
    )
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
}

stage_images() {
    local image_count upload_dir
    [[ -d "${IMAGE_DIR}" ]] || die "Image directory not found: ${IMAGE_DIR}"
    image_count=$(find "${IMAGE_DIR}" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.tif" -o -name "*.tiff" \) | wc -l | tr -d ' ')
    [[ "${image_count}" -gt 0 ]] || die "No image files found in ${IMAGE_DIR}"

    upload_dir="${SCRIPT_DIR}/.upload_staging/${RUN_ID}"
    rm -rf "${upload_dir}"
    mkdir -p "${upload_dir}"
    find "${IMAGE_DIR}" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.tif" -o -name "*.tiff" \) | sort \
        | head -n "${MAX_IMAGES}" \
        | while IFS= read -r file_path; do
            ln -sf "$(cd "$(dirname "${file_path}")" && pwd)/$(basename "${file_path}")" "${upload_dir}/$(basename "${file_path}")"
        done
    echo "${upload_dir}"
}

banner "Demo 12: internal_genAI HMAE Direct"
echo "Config    : ${CONF}"
echo "Mode      : ${HMAE_MODE}"
echo "Image dir : ${IMAGE_DIR}"
echo "Run ID    : ${RUN_ID}"
echo ""

log "Step 1: Pre-flight checks..."
preflight

log "Step 2: Ensuring generated experience..."
generate_experience
ensure_experience "${GENERATED_EXPERIENCE}" --update

log "Step 3: Staging images..."
UPLOAD_DIR="$(stage_images)"
ok "Images staged in ${UPLOAD_DIR}"

while IFS= read -r mode; do
    alias="$(mode_alias "${mode}")"
    task="$(mode_task "${mode}")"
    mode_results="${RESULTS_DIR}/${mode}"
    log "Step 4/${mode}: Uploading images for task ${task} to ${alias}/${RUN_ID}..."
    $CLIENT upload_dir "${UPLOAD_DIR}" "${alias}/${RUN_ID}"
    ok "Uploaded ${mode} inputs"

    log "Step 5/${mode}: Waiting for result artifacts..."
    mkdir -p "${mode_results}/artifacts"
    if download_results_bounded "${alias}/${RUN_ID}" "${mode_results}/artifacts" --force-overwrite; then
        check_result_files "${mode_results}/artifacts" "${HMAE_REQUIRE_RESULTS:-0}" "HMAE ${mode} results"
    else
        [[ "${HMAE_REQUIRE_RESULTS:-0}" = "1" ]] && die "Could not download HMAE ${mode} results"
        warn "Could not download HMAE ${mode} results yet"
    fi
done < <(selected_modes)

banner "Demo 12 Complete"
find "${RESULTS_DIR}" -type f -maxdepth 4 -print 2>/dev/null | sort || true
