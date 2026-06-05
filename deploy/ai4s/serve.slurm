#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# AI4S real vLLM serve wrapper (Singularity / GB200 aarch64)
# ─────────────────────────────────────────────────────────────────────────────
# Replaces serve-openai-smoke.slurm with a real mini-RiVault runtime:
# runs the aarch64 vLLM+LiteLLM+NGINX SIF on the allocated GB200 node and
# exposes an OpenAI-compatible endpoint on RIVAULT_LITELLM_PORT (the port the
# umrs-front Gateway probes and proxies to).
#
# Orchestrator contract (remote_slurm_backend._submit_wrapper_job):
#   Injected env: VLLM_MODEL_PATH, VLLM_SERVED_MODEL_NAME, RIVAULT_DIR,
#                 VLLM_TENSOR_PARALLEL_SIZE, VLLM_MAX_MODEL_LEN,
#                 VLLM_GPU_MEMORY_UTILIZATION, LITELLM_MASTER_KEY,
#                 LITELLM_PORT, JOB_INFO_DIR, RIVAULT_CONTROL_DIR.
#   Job-info file rivault-job-<id>.info reports RIVAULT_LITELLM_PORT/NODE/STATUS.
#   A stop-<id>.request file in RIVAULT_CONTROL_DIR triggers graceful shutdown.
#
# NOTE: This wrapper targets SINGLE-NODE tensor parallelism (TP ≤ GPUs on the
# node, i.e. ≤ 4 on GB200 NVL4). Models that exceed one node's aggregate GPU
# memory (e.g. DeepSeek-V4-Pro / GLM-5.1 in full precision) require multi-node
# Ray and are out of scope for this wrapper — see TODO at the bottom.
# ─────────────────────────────────────────────────────────────────────────────

JOB_INFO_DIR="${JOB_INFO_DIR:-/work/hps0/home/ea0022/ai4s-platform/umrs/deploy/ai4s/jobs}"
RIVAULT_CONTROL_DIR="${RIVAULT_CONTROL_DIR:-${JOB_INFO_DIR}/control}"

# External port the Gateway connects to (NGINX inside the container listens here).
EXTERNAL_PORT="${LITELLM_PORT:-${NGINX_LITELLM_DOCKER_PORT:-8001}}"

# Internal container ports (not exposed outside the node).
INNER_VLLM_PORT="${VLLM_PORT:-8000}"
INNER_LITELLM_PORT="4000"

MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-${MODEL_NAME:-rivault-model}}"
HOST_MODEL_PATH="${VLLM_MODEL_PATH:?VLLM_MODEL_PATH is required}"
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:?LITELLM_MASTER_KEY is required}"

# vLLM tuning (orchestrator-provided with safe fallbacks).
VLLM_TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_PP="${VLLM_PIPELINE_PARALLEL_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

# Resolve the SIF image. RIVAULT_DIR may be the .sif itself or its directory.
RIVAULT_DIR="${RIVAULT_DIR:-/work/hps0/home/ea0022/images}"
if [[ "${RIVAULT_DIR}" == *.sif ]]; then
    SIF_IMAGE="${RIVAULT_DIR}"
else
    SIF_IMAGE="${RIVAULT_SIF:-${RIVAULT_DIR%/}/mini-rivault-aarch64.sif}"
fi

# Derive the vLLM tool-call parser from the model name when not explicitly set.
derive_tool_parser() {
    local name_lc
    name_lc="$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')"
    case "${name_lc}" in
        *deepseek*v4*|*deepseek-v4*) echo "deepseek_v4" ;;
        *deepseek*)                  echo "deepseek_v3" ;;
        *glm*)                       echo "glm4_moe" ;;
        *kimi*)                      echo "kimi_k2" ;;
        *)                           echo "hermes" ;;
    esac
}
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-$(derive_tool_parser)}"

mkdir -p "${JOB_INFO_DIR}" "${RIVAULT_CONTROL_DIR}"

RIVAULT_NODE="$(hostname -f 2>/dev/null || hostname)"
JOB_INFO_FILE="${JOB_INFO_DIR}/rivault-job-${SLURM_JOB_ID}.info"
STOP_REQUEST_FILE="${RIVAULT_CONTROL_DIR}/stop-${SLURM_JOB_ID}.request"
CONTAINER_PID=""

write_job_info() {
    local status="$1"
    cat > "${JOB_INFO_FILE}" <<EOF
RIVAULT_JOB_ID=${SLURM_JOB_ID}
RIVAULT_STATUS=${status}
RIVAULT_NODE=${RIVAULT_NODE}
RIVAULT_LITELLM_PORT=${EXTERNAL_PORT}
RIVAULT_VLLM_PORT=${INNER_VLLM_PORT}
RIVAULT_MODEL_PATH=${HOST_MODEL_PATH}
RIVAULT_MODEL_NAME=${MODEL_NAME}
RIVAULT_SIF=${SIF_IMAGE}
RIVAULT_TOOL_PARSER=${VLLM_TOOL_CALL_PARSER}
RIVAULT_START_TIME=$(date -Iseconds)
EOF
}

cleanup() {
    local rc=$?
    trap - EXIT TERM INT
    if [[ -n "${CONTAINER_PID}" ]]; then
        kill "${CONTAINER_PID}" >/dev/null 2>&1 || true
        wait "${CONTAINER_PID}" >/dev/null 2>&1 || true
    fi
    rm -f "${STOP_REQUEST_FILE}" >/dev/null 2>&1 || true
    if [[ -f "${JOB_INFO_FILE}" ]]; then
        sed -i 's/RIVAULT_STATUS=.*/RIVAULT_STATUS=stopped/' "${JOB_INFO_FILE}" || true
        echo "RIVAULT_STOP_TIME=$(date -Iseconds)" >> "${JOB_INFO_FILE}" || true
    fi
    exit "${rc}"
}
trap cleanup EXIT TERM INT

# ── Preflight validation ─────────────────────────────────────────────────────
if [[ ! -f "${SIF_IMAGE}" ]]; then
    echo "[serve] FATAL: SIF image not found: ${SIF_IMAGE}" >&2
    write_job_info failed
    exit 1
fi
if [[ ! -d "${HOST_MODEL_PATH}" ]]; then
    echo "[serve] FATAL: model path not found: ${HOST_MODEL_PATH}" >&2
    write_job_info failed
    exit 1
fi

CONTAINER_CMD="$(command -v singularity || command -v apptainer || true)"
if [[ -z "${CONTAINER_CMD}" ]]; then
    echo "[serve] FATAL: neither singularity nor apptainer found on PATH" >&2
    write_job_info failed
    exit 1
fi

echo "[serve] SLURM_JOB_ID=${SLURM_JOB_ID} node=${RIVAULT_NODE}"
echo "[serve] model=${MODEL_NAME} path=${HOST_MODEL_PATH}"
echo "[serve] sif=${SIF_IMAGE}"
echo "[serve] external_port=${EXTERNAL_PORT} tp=${VLLM_TP} pp=${VLLM_PP} maxlen=${VLLM_MAX_MODEL_LEN}"
echo "[serve] tool_parser=${VLLM_TOOL_CALL_PARSER} runtime=${CONTAINER_CMD}"

write_job_info starting

# ── Launch the mini-RiVault container ────────────────────────────────────────
# Model weights are bind-mounted read-only at /models; the container runs fully
# offline (TRANSFORMERS_OFFLINE=1 / HF_HUB_OFFLINE=1 are baked into the image).
# --writable-tmpfs gives an in-memory overlay so the container's start.sh can
# render its runtime config (litellm-config.yaml, nginx site, supervisord logs)
# into the otherwise read-only SIF filesystem.
"${CONTAINER_CMD}" exec --nv --cleanenv --writable-tmpfs \
    --bind "${HOST_MODEL_PATH}:/models:ro" \
    --env VLLM_MODEL_PATH="/models" \
    --env VLLM_SERVED_MODEL="${MODEL_NAME}" \
    --env VLLM_PORT="${INNER_VLLM_PORT}" \
    --env LITELLM_PORT="${INNER_LITELLM_PORT}" \
    --env NGINX_PORT="${EXTERNAL_PORT}" \
    --env VLLM_TP="${VLLM_TP}" \
    --env VLLM_PP="${VLLM_PP}" \
    --env VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
    --env VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL}" \
    --env VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER}" \
    --env LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY}" \
    "${SIF_IMAGE}" /opt/mini-rivault/start.sh &
CONTAINER_PID=$!

write_job_info running

# ── Supervise: exit on container death or graceful stop request ──────────────
while kill -0 "${CONTAINER_PID}" >/dev/null 2>&1; do
    if [[ -f "${STOP_REQUEST_FILE}" ]]; then
        echo "[serve] graceful stop requested: ${STOP_REQUEST_FILE}"
        exit 0
    fi
    sleep 2
done

wait "${CONTAINER_PID}"

# ─────────────────────────────────────────────────────────────────────────────
# TODO (multi-node): DeepSeek-V4-Pro / GLM-5.1 exceed one GB200 node's GPU
# memory and need TP/PP spanning ≥2 nodes via a Ray cluster:
#   - Start `ray start --head` on $SLURMD_NODENAME, `ray start --address=...`
#     on the others via `srun --ntasks-per-node=1`, then launch vLLM with
#     -tp <gpus_per_node> -pp <node_count> inside the head container.
#   - Requires --nodes>1 (orchestrator node_count) and a 2n4gpu/4n4gpu partition.
# ─────────────────────────────────────────────────────────────────────────────
