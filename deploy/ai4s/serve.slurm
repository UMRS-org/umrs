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
    if [[ -n "${RUNTIME_CACHE:-}" ]]; then
        rm -rf "${RUNTIME_CACHE}" >/dev/null 2>&1 || true
    fi
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

# ── Launch vLLM directly (OpenAI-compatible server) ──────────────────────────
# The mini-RiVault SIF bundles vLLM+LiteLLM+NGINX, but its internal start.sh
# renders config into read-only image paths (/opt/mini-rivault, /etc/nginx),
# which cannot be made writable on AI4S: the site's underlay mount mode
# disables --writable-tmpfs under --fakeroot, and a plain overlay leaves those
# root-owned dirs unwritable for the unprivileged job user.
#
# vLLM already exposes an OpenAI-compatible API with bearer-token auth
# (--api-key), so we run it directly on the external port the Gateway probes.
# All writable runtime state (HF/torch/triton/vLLM caches, HOME) is redirected
# to a per-job /tmp directory; the model weights stay read-only at /models.
RUNTIME_CACHE="${TMPDIR:-/tmp}/rivault-${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_CACHE}/home" "${RUNTIME_CACHE}/cache"

"${CONTAINER_CMD}" exec --nv --cleanenv \
    --bind "${HOST_MODEL_PATH}:/models:ro" \
    --bind "${RUNTIME_CACHE}:${RUNTIME_CACHE}" \
    --env HOME="${RUNTIME_CACHE}/home" \
    --env XDG_CACHE_HOME="${RUNTIME_CACHE}/cache" \
    --env HF_HOME="${RUNTIME_CACHE}/cache/hf" \
    --env VLLM_CACHE_ROOT="${RUNTIME_CACHE}/cache/vllm" \
    --env TRITON_CACHE_DIR="${RUNTIME_CACHE}/cache/triton" \
    --env TORCHINDUCTOR_CACHE_DIR="${RUNTIME_CACHE}/cache/inductor" \
    --env OUTLINES_CACHE_DIR="${RUNTIME_CACHE}/cache/outlines" \
    --env XGRAMMAR_CACHE_DIR="${RUNTIME_CACHE}/cache/xgrammar" \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_HUB_OFFLINE=1 \
    --env VLLM_NO_USAGE_STATS=1 \
    --env VLLM_DO_NOT_TRACK=1 \
    "${SIF_IMAGE}" /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
        --model /models \
        --served-model-name "${MODEL_NAME}" \
        --host 0.0.0.0 \
        --port "${EXTERNAL_PORT}" \
        -tp "${VLLM_TP}" \
        -pp "${VLLM_PP}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser "${VLLM_TOOL_CALL_PARSER}" \
        --api-key "${LITELLM_MASTER_KEY}" &
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
