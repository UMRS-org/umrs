#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="jens_sam3" source "${SCRIPT_DIR}/../../demos/lib/common.sh"

BUCKET_ALIAS="${SAM3_BUCKET:-jens-sam3-direct}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
SAM3_PROMPT="${SAM3_PROMPT:-cat}"
MAX_IMAGES="${MAX_IMAGES:-1}"
RESULTS_DIR="${SCRIPT_DIR}/results/${RUN_ID}"
GENERATED_EXPERIENCE="${RESULTS_DIR}/experience.json"

if [[ -z "${IMAGE_DIR:-}" ]]; then
    IMAGE_DIR="${SCRIPT_DIR}/../../demos/sample_data/images"
fi

generate_experience() {
    mkdir -p "${RESULTS_DIR}"
    SAM3_PROMPT="${SAM3_PROMPT}" BUCKET_ALIAS="${BUCKET_ALIAS}" python3 - <<'PY' > "${GENERATED_EXPERIENCE}"
import json
import os

prompt = os.environ["SAM3_PROMPT"]
bucket_alias = os.environ["BUCKET_ALIAS"]
payload = {
    "xp_name": "Jens SAM3 Direct",
    "sql_username": "jens_sam3_direct",
    "modalities": [
        {
            "modality_name": "image",
            "task": "segmentation",
            "s3": {
                "bucket_alias": bucket_alias,
                "need_output_bucket": True,
                "data": {"type": "file", "mime_type": "image/png"},
            },
            "model": {
                "model_name": "internal-genai-sam3",
                "vector_size": "256",
                "model_type": "rivault://bentoml://http://localhost:{LOCAL_PORT}/vision/infer/submit",
                "vision_model_type": "sam3",
                "vision_model_config": {"prompt": prompt},
                "return_type": "file",
            },
            "sql": {
                "store_embeddings": "no",
                "table_name": "jens_sam3_seg",
                "meta_data": [],
            },
        }
    ],
}
print(json.dumps(payload, indent=2))
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

banner "Demo 11: internal_genAI SAM3 Direct"
echo "Config     : ${CONF}"
echo "Bucket     : ${BUCKET_ALIAS}/${RUN_ID}"
echo "Image dir  : ${IMAGE_DIR}"
echo "Prompt     : ${SAM3_PROMPT}"
echo ""

log "Step 1: Pre-flight checks..."
preflight

log "Step 2: Ensuring generated experience..."
generate_experience
ensure_experience "${GENERATED_EXPERIENCE}" --update

log "Step 3: Staging images..."
UPLOAD_DIR="$(stage_images)"
ok "Images staged in ${UPLOAD_DIR}"

log "Step 4: Uploading images through UMRS..."
$CLIENT upload_dir "${UPLOAD_DIR}" "${BUCKET_ALIAS}/${RUN_ID}"
ok "Images uploaded; UMRS server will dispatch internal_genAI"

log "Step 5: Waiting for result artifacts..."
mkdir -p "${RESULTS_DIR}/artifacts"
if download_results_bounded "${BUCKET_ALIAS}/${RUN_ID}" "${RESULTS_DIR}/artifacts" --force-overwrite; then
    check_result_files "${RESULTS_DIR}/artifacts" "${SAM3_REQUIRE_RESULTS:-0}" "SAM3 results"
else
    [[ "${SAM3_REQUIRE_RESULTS:-0}" = "1" ]] && die "Could not download SAM3 results"
    warn "Could not download SAM3 results yet"
fi

banner "Demo 11 Complete"
find "${RESULTS_DIR}/artifacts" -type f -maxdepth 3 -print 2>/dev/null | sort || true
