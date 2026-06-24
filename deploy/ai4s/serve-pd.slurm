#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# AI4S Prefill/Decode Disaggregated serving wrapper (Singularity / GB200 aarch64)
# ─────────────────────────────────────────────────────────────────────────────
#
# PD topology: VLLM_PD_NODE_COUNT nodes total, split evenly:
#   - First half: Prefill workers (kv_producer)
#   - Second half: Decode workers (kv_consumer) + exposes HTTP to Gateway
#
# Architecture (recipe: recipes.vllm.ai/zai-org/GLM-5.2, pd_cluster, FP8):
#   Client → Decode worker (port EXTERNAL_PORT, NIXL kv_consumer)
#   Prefill worker (port PD_PREFILL_PORT, NIXL kv_producer) ← feeds KV to decode
#
# NIXL side-channel: Prefill advertises KV blocks to Decode via UCX side channel
# (VLLM_NIXL_SIDE_CHANNEL_HOST=prefill_head, VLLM_NIXL_SIDE_CHANNEL_PORT=5557).
#
# Requirements:
#   - nixl_packages/ installed at NIXL_PACKAGES_HOST (bind-mounted as PYTHONPATH)
#   - Even SLURM_NNODES (default 4: 2 prefill + 2 decode)
#   - Supported models: FP8 GLM-5.2-FP8 (--kv-cache-dtype fp8)
#   - Orchestrator contract same as serve.slurm; job-info file written to
#     JOB_INFO_DIR/rivault-job-<SLURM_JOB_ID>.info
#
# env_vars in pool YAML:
#   VLLM_PD_PREFILL_NODES: nodes dedicated to prefill (default: NODE_COUNT/2)
#   VLLM_KV_CACHE_DTYPE: kv cache dtype (default: fp8)
#   VLLM_NIXL_SIDE_CHANNEL_PORT: side channel port (default: 5557)
#   PD_PREFILL_PORT: internal vLLM port for prefill worker (default: 8001)
#   RIVAULT_SIF / VLLM_MAX_NUM_SEQS / VLLM_MAX_NUM_BATCHED_TOKENS: same as serve.slurm
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_JOB_INFO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/jobs"
JOB_INFO_DIR="${JOB_INFO_DIR:-${_DEFAULT_JOB_INFO_DIR}}"
RIVAULT_CONTROL_DIR="${RIVAULT_CONTROL_DIR:-${JOB_INFO_DIR}/control}"

# External port: Decode worker exposes HTTP here (Gateway reaches it).
EXTERNAL_PORT="${LITELLM_PORT:-8001}"
# Prefill internal port (not exposed to Gateway).
PD_PREFILL_PORT="${PD_PREFILL_PORT:-8002}"
# NIXL side-channel port for KV block advertising.
NIXL_SIDE_CHANNEL_PORT="${VLLM_NIXL_SIDE_CHANNEL_PORT:-5557}"

SHORT_MODEL_NAME="${MODEL_NAME:-}"
MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-${MODEL_NAME:-rivault-model}}"
if [[ "${SHORT_MODEL_NAME}" == "${MODEL_NAME}" ]]; then SHORT_MODEL_NAME=""; fi
HOST_MODEL_PATH="${VLLM_MODEL_PATH:?VLLM_MODEL_PATH is required}"
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:?LITELLM_MASTER_KEY is required}"

VLLM_TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
NODE_COUNT="${SLURM_NNODES:-4}"

# Require even node count for PD split.
if (( NODE_COUNT % 2 != 0 )); then
    echo "[serve-pd] FATAL: NODE_COUNT=${NODE_COUNT} must be even for PD split" >&2
    exit 1
fi
PD_PREFILL_NODES="${VLLM_PD_PREFILL_NODES:-$(( NODE_COUNT / 2 ))}"
PD_DECODE_NODES=$(( NODE_COUNT - PD_PREFILL_NODES ))
VLLM_GPUS_PER_NODE="${VLLM_TP}"

# TEP for PD: TP = GPUs/node × nodes for each pool (prefill or decode separately).
VLLM_PREFILL_TP=$(( VLLM_TP * PD_PREFILL_NODES ))
VLLM_DECODE_TP=$(( VLLM_TP * PD_DECODE_NODES ))

VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-fp8}"

# Resolve SIF image.
RIVAULT_DIR="${RIVAULT_DIR:-/work/hps0/home/ea0022/images}"
if [[ "${RIVAULT_DIR}" == *.sif ]]; then
    SIF_IMAGE="${RIVAULT_DIR}"
else
    SIF_IMAGE="${RIVAULT_SIF:-${RIVAULT_DIR%/}/mini-rivault-aarch64-v6.sif}"
fi

# nixl packages bind-mounted for NIXL support.
NIXL_PACKAGES_HOST="${NIXL_PACKAGES_HOST:-/work/hps0/home/ea0022/images/nixl_packages}"

FLASHINFER_CUBIN_HOST="${FLASHINFER_CUBIN_DIR_HOST:-/work/hps0/home/ea0022/images/flashinfer_cubins}"
FLASHINFER_CUBIN_TARGET="/opt/venv/lib/python3.12/site-packages/flashinfer_cubin/cubins"

mkdir -p "${JOB_INFO_DIR}" "${RIVAULT_CONTROL_DIR}"

RIVAULT_NODE="$(hostname -f 2>/dev/null || hostname)"
JOB_INFO_FILE="${JOB_INFO_DIR}/rivault-job-${SLURM_JOB_ID}.info"
STOP_REQUEST_FILE="${RIVAULT_CONTROL_DIR}/stop-${SLURM_JOB_ID}.request"

write_job_info() {
    local status="$1"
    cat > "${JOB_INFO_FILE}" <<EOF
RIVAULT_JOB_ID=${SLURM_JOB_ID}
RIVAULT_STATUS=${status}
RIVAULT_NODE=${RIVAULT_NODE}
RIVAULT_LITELLM_PORT=${EXTERNAL_PORT}
RIVAULT_VLLM_PORT=${EXTERNAL_PORT}
RIVAULT_MODEL_PATH=${HOST_MODEL_PATH}
RIVAULT_MODEL_NAME=${MODEL_NAME}
RIVAULT_SIF=${SIF_IMAGE}
RIVAULT_TOOL_PARSER=glm47
RIVAULT_START_TIME=$(date -Iseconds)
RIVAULT_PD_MODE=true
RIVAULT_PD_PREFILL_NODES=${PD_PREFILL_NODES}
RIVAULT_PD_DECODE_NODES=${PD_DECODE_NODES}
EOF
}

CONTAINER_CMD="$(command -v singularity || command -v apptainer || true)"
if [[ -z "${CONTAINER_CMD}" ]]; then
    echo "[serve-pd] FATAL: neither singularity nor apptainer found on PATH" >&2
    write_job_info failed; exit 1
fi
if [[ ! -f "${SIF_IMAGE}" ]]; then
    echo "[serve-pd] FATAL: SIF not found: ${SIF_IMAGE}" >&2
    write_job_info failed; exit 1
fi
if [[ ! -d "${HOST_MODEL_PATH}" ]]; then
    echo "[serve-pd] FATAL: model path not found: ${HOST_MODEL_PATH}" >&2
    write_job_info failed; exit 1
fi

# Get all node hostnames.
mapfile -t ALL_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD_NODE="${ALL_NODES[0]}"
HEAD_IP="$(hostname -i 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+(\.[0-9]+){3}$' | head -1)"
if [[ -z "${HEAD_IP}" ]]; then
    HEAD_IP="$(getent hosts "${HEAD_NODE}" | awk '{print $1; exit}')"
fi

# Split node list: first PD_PREFILL_NODES = prefill, rest = decode.
PREFILL_NODES=("${ALL_NODES[@]:0:${PD_PREFILL_NODES}}")
DECODE_NODES=("${ALL_NODES[@]:${PD_PREFILL_NODES}:${PD_DECODE_NODES}}")
PREFILL_HEAD="${PREFILL_NODES[0]}"
PREFILL_HEAD_IP="$(getent hosts "${PREFILL_HEAD}" | awk '{print $1; exit}')"
DECODE_HEAD="${DECODE_NODES[0]}"
DECODE_HEAD_IP="$(getent hosts "${DECODE_HEAD}" | awk '{print $1; exit}')"

echo "[serve-pd] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[serve-pd] total_nodes=${NODE_COUNT} prefill=${PD_PREFILL_NODES} decode=${PD_DECODE_NODES}"
echo "[serve-pd] prefill_nodes=${PREFILL_NODES[*]} (head=${PREFILL_HEAD} ${PREFILL_HEAD_IP})"
echo "[serve-pd] decode_nodes=${DECODE_NODES[*]} (head=${DECODE_HEAD} ${DECODE_HEAD_IP})"
echo "[serve-pd] model=${MODEL_NAME} sif=${SIF_IMAGE}"
echo "[serve-pd] external_port=${EXTERNAL_PORT} prefill_port=${PD_PREFILL_PORT} nixl_port=${NIXL_SIDE_CHANNEL_PORT}"

write_job_info starting

RUNTIME_CACHE="${TMPDIR:-/tmp}/rivault-${SLURM_JOB_ID}"
# Create runtime cache on all nodes.
srun --nodes="${NODE_COUNT}" --ntasks-per-node=1 \
    mkdir -p "${RUNTIME_CACHE}/home" "${RUNTIME_CACHE}/cache" "${RUNTIME_CACHE}/ray" || true

# Common singularity flags.
SING_COMMON=(
    exec --nv --cleanenv
    --bind "${HOST_MODEL_PATH}:/models:ro"
    --bind "${RUNTIME_CACHE}:${RUNTIME_CACHE}"
    --env HOME="${RUNTIME_CACHE}/home"
    --env XDG_CACHE_HOME="${RUNTIME_CACHE}/cache"
    --env HF_HOME="${RUNTIME_CACHE}/cache/hf"
    --env VLLM_CACHE_ROOT="${RUNTIME_CACHE}/cache/vllm"
    --env TRITON_CACHE_DIR="${RUNTIME_CACHE}/cache/triton"
    --env TORCHINDUCTOR_CACHE_DIR="${RUNTIME_CACHE}/cache/inductor"
    --env OUTLINES_CACHE_DIR="${RUNTIME_CACHE}/cache/outlines"
    --env XGRAMMAR_CACHE_DIR="${RUNTIME_CACHE}/cache/xgrammar"
    --env TRANSFORMERS_OFFLINE=1
    --env HF_HUB_OFFLINE=1
    --env VLLM_NO_USAGE_STATS=1
    --env VLLM_DO_NOT_TRACK=1
    --env RAY_DISABLE_IMPORT_WARNING=1
    --env LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:/opt/venv/lib/python3.12/site-packages/torch/lib"
    # NIXL packages bind-mounted for PD KV transfer.
    --bind "${NIXL_PACKAGES_HOST}:${NIXL_PACKAGES_HOST}:ro"
    --env PYTHONPATH="${NIXL_PACKAGES_HOST}"
    # NCCL settings for multi-node NVLink + InfiniBand KV transfer.
    --env NCCL_CUMEM_ENABLE=1
    --env NCCL_NVLS_ENABLE=1
    --env UCX_NET_DEVICES=all
)
if [[ -d "${FLASHINFER_CUBIN_HOST}" ]]; then
    SING_COMMON+=( --bind "${FLASHINFER_CUBIN_HOST}:${FLASHINFER_CUBIN_TARGET}" )
fi

# GLM: force Triton MoE kernel (see serve.slurm for background).
echo "[serve-pd] GLM: forcing Triton MoE (--moe-backend triton)"
VLLM_MOE_BACKEND="triton"

# Common vLLM args for both prefill and decode.
COMMON_VLLM_ARGS=(
    --model /models
    --served-model-name "${MODEL_NAME}"
    ${SHORT_MODEL_NAME:+"--served-model-name" "${SHORT_MODEL_NAME}"}
    --trust-remote-code
    --api-key "${LITELLM_MASTER_KEY}"
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
    --enforce-eager
    --kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}"
    --moe-backend "${VLLM_MOE_BACKEND}"
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --chat-template-content-format string
    --model-loader-extra-config '{"enable_weights_track": false}'
    --enable-prefix-caching
    --distributed-executor-backend ray
)
if [[ -n "${VLLM_MAX_MODEL_LEN}" && "${VLLM_MAX_MODEL_LEN}" != "0" && "${VLLM_MAX_MODEL_LEN}" != "full" ]]; then
    COMMON_VLLM_ARGS+=( --max-model-len "${VLLM_MAX_MODEL_LEN}" )
fi
if [[ -n "${VLLM_MAX_NUM_SEQS:-}" ]]; then
    COMMON_VLLM_ARGS+=( --max-num-seqs "${VLLM_MAX_NUM_SEQS}" )
fi
if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS:-}" ]]; then
    COMMON_VLLM_ARGS+=( --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" )
fi

# ── Ray cluster for Prefill pool ─────────────────────────────────────────────
RAY_PREFILL_PORT=6379
RAY_PREFILL_ADDRESS="${PREFILL_HEAD_IP}:${RAY_PREFILL_PORT}"

echo "[serve-pd] starting Ray cluster for prefill pool (head=${PREFILL_HEAD_IP})"
"${CONTAINER_CMD}" "${SING_COMMON[@]}" \
    "${SIF_IMAGE}" /bin/sh -c '
        unset VLLM_PORT
        exec /opt/venv/bin/ray start --head \
            --node-ip-address="$1" --port="$2" \
            --num-gpus="$3" --temp-dir="$4" \
            --include-dashboard=false --block' _ \
    "${PREFILL_HEAD_IP}" "${RAY_PREFILL_PORT}" "${VLLM_GPUS_PER_NODE}" "${RUNTIME_CACHE}/ray" &
RAY_PREFILL_PID=$!

for ((i=1; i<${#PREFILL_NODES[@]}; i++)); do
    pnode="${PREFILL_NODES[$i]}"
    srun --nodes=1 --ntasks=1 --nodelist="${pnode}" \
        "${CONTAINER_CMD}" "${SING_COMMON[@]}" "${SIF_IMAGE}" /bin/sh -c '
            unset VLLM_PORT
            exec /opt/venv/bin/ray start --address="$1" --num-gpus="$2" --temp-dir="$3" --block' _ \
        "${RAY_PREFILL_ADDRESS}" "${VLLM_GPUS_PER_NODE}" "${RUNTIME_CACHE}/ray" &
done

# Wait for prefill Ray to have all GPUs.
for _ in $(seq 1 60); do
    have="$("${CONTAINER_CMD}" "${SING_COMMON[@]}" "${SIF_IMAGE}" \
        /opt/venv/bin/python -c "import ray; ray.init(address='${RAY_PREFILL_ADDRESS}'); print(int(ray.cluster_resources().get('GPU',0)))" \
        2>/dev/null | tail -1)"
    [[ "${have}" == "$(( VLLM_GPUS_PER_NODE * PD_PREFILL_NODES ))" ]] && break
    sleep 5
done
echo "[serve-pd] prefill Ray GPUs: ${have}/$(( VLLM_GPUS_PER_NODE * PD_PREFILL_NODES ))"

# ── Ray cluster for Decode pool ──────────────────────────────────────────────
RAY_DECODE_PORT=6380
RAY_DECODE_ADDRESS="${DECODE_HEAD_IP}:${RAY_DECODE_PORT}"

echo "[serve-pd] starting Ray cluster for decode pool (head=${DECODE_HEAD_IP})"
"${CONTAINER_CMD}" "${SING_COMMON[@]}" \
    --env GLOO_SOCKET_IFNAME=ib0 \
    --env NCCL_SOCKET_IFNAME=ib0 \
    "${SIF_IMAGE}" /bin/sh -c '
        unset VLLM_PORT
        exec /opt/venv/bin/ray start --head \
            --node-ip-address="$1" --port="$2" \
            --num-gpus="$3" --temp-dir="$4" \
            --include-dashboard=false --block' _ \
    "${DECODE_HEAD_IP}" "${RAY_DECODE_PORT}" "${VLLM_GPUS_PER_NODE}" "${RUNTIME_CACHE}/ray" &
RAY_DECODE_PID=$!

for ((i=1; i<${#DECODE_NODES[@]}; i++)); do
    dnode="${DECODE_NODES[$i]}"
    srun --nodes=1 --ntasks=1 --nodelist="${dnode}" \
        "${CONTAINER_CMD}" "${SING_COMMON[@]}" "${SIF_IMAGE}" /bin/sh -c '
            unset VLLM_PORT
            exec /opt/venv/bin/ray start --address="$1" --num-gpus="$2" --temp-dir="$3" --block' _ \
        "${RAY_DECODE_ADDRESS}" "${VLLM_GPUS_PER_NODE}" "${RUNTIME_CACHE}/ray" &
done

for _ in $(seq 1 60); do
    have="$("${CONTAINER_CMD}" "${SING_COMMON[@]}" "${SIF_IMAGE}" \
        /opt/venv/bin/python -c "import ray; ray.init(address='${RAY_DECODE_ADDRESS}'); print(int(ray.cluster_resources().get('GPU',0)))" \
        2>/dev/null | tail -1)"
    [[ "${have}" == "$(( VLLM_GPUS_PER_NODE * PD_DECODE_NODES ))" ]] && break
    sleep 5
done
echo "[serve-pd] decode Ray GPUs: ${have}/$(( VLLM_GPUS_PER_NODE * PD_DECODE_NODES ))"

# ── Launch Prefill worker (kv_producer) ──────────────────────────────────────
echo "[serve-pd] launching prefill worker (kv_producer, port=${PD_PREFILL_PORT}, tp=${VLLM_PREFILL_TP})"
"${CONTAINER_CMD}" "${SING_COMMON[@]}" \
    --env RAY_ADDRESS="${RAY_PREFILL_ADDRESS}" \
    --env VLLM_HOST_IP="${PREFILL_HEAD_IP}" \
    --env VLLM_NIXL_SIDE_CHANNEL_HOST="${PREFILL_HEAD_IP}" \
    --env VLLM_NIXL_SIDE_CHANNEL_PORT="${NIXL_SIDE_CHANNEL_PORT}" \
    "${SIF_IMAGE}" /bin/sh -c '
        unset VLLM_PORT
        exec /opt/venv/bin/python -m vllm.entrypoints.openai.api_server "$@"' _ \
    "${COMMON_VLLM_ARGS[@]}" \
    --host 0.0.0.0 \
    --port "${PD_PREFILL_PORT}" \
    --tensor-parallel-size "${VLLM_PREFILL_TP}" \
    --pipeline-parallel-size 1 \
    --nnodes "${PD_PREFILL_NODES}" \
    --node-rank 0 \
    --master-addr "${PREFILL_HEAD_IP}" \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
    &
PREFILL_PID=$!

# Wait for prefill to be ready (poll /v1/models).
echo "[serve-pd] waiting for prefill worker to be ready..."
for _ in $(seq 1 120); do
    if curl -sf -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
        "http://${PREFILL_HEAD_IP}:${PD_PREFILL_PORT}/v1/models" >/dev/null 2>&1; then
        echo "[serve-pd] prefill worker ready"
        break
    fi
    sleep 10
done

# ── Launch Decode worker (kv_consumer) ───────────────────────────────────────
echo "[serve-pd] launching decode worker (kv_consumer, port=${EXTERNAL_PORT}, tp=${VLLM_DECODE_TP})"
write_job_info running
"${CONTAINER_CMD}" "${SING_COMMON[@]}" \
    --env RAY_ADDRESS="${RAY_DECODE_ADDRESS}" \
    --env VLLM_HOST_IP="${DECODE_HEAD_IP}" \
    --env VLLM_NIXL_SIDE_CHANNEL_HOST="${PREFILL_HEAD_IP}" \
    --env VLLM_NIXL_SIDE_CHANNEL_PORT="${NIXL_SIDE_CHANNEL_PORT}" \
    "${SIF_IMAGE}" /bin/sh -c '
        unset VLLM_PORT
        exec /opt/venv/bin/python -m vllm.entrypoints.openai.api_server "$@"' _ \
    "${COMMON_VLLM_ARGS[@]}" \
    --host 0.0.0.0 \
    --port "${EXTERNAL_PORT}" \
    --tensor-parallel-size "${VLLM_DECODE_TP}" \
    --pipeline-parallel-size 1 \
    --nnodes "${PD_DECODE_NODES}" \
    --node-rank 0 \
    --master-addr "${DECODE_HEAD_IP}" \
    --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\",\"kv_load_failure_policy\":\"fail\",\"kv_connector_extra_config\":{\"prefill_addr\":\"${PREFILL_HEAD_IP}\",\"prefill_port\":${PD_PREFILL_PORT},\"nixl_side_channel_host\":\"${PREFILL_HEAD_IP}\",\"nixl_side_channel_port\":${NIXL_SIDE_CHANNEL_PORT}}}" \
    &
DECODE_PID=$!

cleanup() {
    local rc=$?
    trap - EXIT TERM INT
    kill "${PREFILL_PID}" "${DECODE_PID}" "${RAY_PREFILL_PID}" "${RAY_DECODE_PID}" >/dev/null 2>&1 || true
    wait "${PREFILL_PID}" "${DECODE_PID}" >/dev/null 2>&1 || true
    rm -f "${STOP_REQUEST_FILE}" >/dev/null 2>&1 || true
    rm -rf "${RUNTIME_CACHE}" >/dev/null 2>&1 || true
    if [[ -f "${JOB_INFO_FILE}" ]]; then
        sed -i 's/RIVAULT_STATUS=.*/RIVAULT_STATUS=stopped/' "${JOB_INFO_FILE}" || true
        echo "RIVAULT_STOP_TIME=$(date -Iseconds)" >> "${JOB_INFO_FILE}" || true
    fi
    exit "${rc}"
}
trap cleanup EXIT TERM INT

# Supervise: exit on stop request or decode worker death.
while kill -0 "${DECODE_PID}" >/dev/null 2>&1; do
    if [[ -f "${STOP_REQUEST_FILE}" ]]; then
        echo "[serve-pd] graceful stop requested"
        exit 0
    fi
    sleep 2
done
wait "${DECODE_PID}"
