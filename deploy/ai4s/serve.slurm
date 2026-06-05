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
# NOTE: This wrapper supports BOTH single-node tensor parallelism (TP ≤ GPUs on
# the node, i.e. ≤ 4 on GB200 NVL4) and multi-node tensor+pipeline parallelism.
# When Slurm allocates more than one node (SLURM_NNODES > 1, e.g. the 4n4gpu-p
# partition) it bootstraps a Ray cluster across the nodes and launches vLLM with
# -tp <gpus_per_node> -pp <node_count> using the Ray distributed executor, so
# models that exceed one node's aggregate GPU memory (DeepSeek-V4-Pro / GLM-5.1)
# can be served at full context. The SIF must include Ray (see
# deploy/singularity/mini-rivault-aarch64-ray.def).
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
# Tensor parallel = GPUs per node; pipeline parallel = node count. On a
# multi-node allocation Slurm sets SLURM_NNODES, so derive PP from it unless an
# explicit override is provided. World size = TP × PP must equal the total GPU
# count across the allocation (e.g. 4 GPUs/node × 4 nodes = 16 on 4n4gpu-p).
VLLM_TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_PP="${VLLM_PIPELINE_PARALLEL_SIZE:-${SLURM_NNODES:-1}}"
# Empty / 0 / "full" / "native" means: do not pin --max-model-len; let vLLM use
# the model's native maximum context ("full token size").
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
NODE_COUNT="${SLURM_NNODES:-1}"

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
RAY_HEAD_PID=""
WORKER_PIDS=()

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
    # Tear down the Ray cluster (multi-node only). Best-effort: stop the head
    # and any worker srun steps so the allocation releases cleanly.
    if [[ -n "${RAY_HEAD_PID}" ]]; then
        kill "${RAY_HEAD_PID}" >/dev/null 2>&1 || true
    fi
    if [[ ${#WORKER_PIDS[@]} -gt 0 ]]; then
        kill "${WORKER_PIDS[@]}" >/dev/null 2>&1 || true
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
echo "[serve] external_port=${EXTERNAL_PORT} tp=${VLLM_TP} pp=${VLLM_PP} nodes=${NODE_COUNT} maxlen=${VLLM_MAX_MODEL_LEN:-native}"
echo "[serve] tool_parser=${VLLM_TOOL_CALL_PARSER} runtime=${CONTAINER_CMD}"

write_job_info starting

# ── Launch vLLM (OpenAI-compatible server) ───────────────────────────────────
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
#
# Single-node allocations (NODE_COUNT==1) run vLLM directly. Multi-node
# allocations (e.g. 4n4gpu-p) bootstrap a Ray cluster across the nodes first,
# then launch vLLM on the head with -tp <gpus/node> -pp <nodes> and the Ray
# distributed executor backend.
RUNTIME_CACHE="${TMPDIR:-/tmp}/rivault-${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_CACHE}/home" "${RUNTIME_CACHE}/cache" "${RUNTIME_CACHE}/ray"

# Shared singularity exec flags (env + binds) used by every launch path.
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
)

# vLLM server arguments shared across single/multi-node paths.
VLLM_ARGS=(
    --model /models
    --served-model-name "${MODEL_NAME}"
    --host 0.0.0.0
    --port "${EXTERNAL_PORT}"
    -tp "${VLLM_TP}"
    -pp "${VLLM_PP}"
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
    --trust-remote-code
    --enable-auto-tool-choice
    --tool-call-parser "${VLLM_TOOL_CALL_PARSER}"
    --api-key "${LITELLM_MASTER_KEY}"
)
# Pin context length only when explicitly requested; otherwise omit the flag so
# vLLM serves the model's native maximum context ("full token size").
case "$(printf '%s' "${VLLM_MAX_MODEL_LEN}" | tr '[:upper:]' '[:lower:]')" in
    ""|0|full|native|max|auto) : ;;
    *) VLLM_ARGS+=( --max-model-len "${VLLM_MAX_MODEL_LEN}" ) ;;
esac

if [[ "${NODE_COUNT}" -gt 1 ]]; then
    # ── Multi-node: bootstrap a Ray cluster across the allocation ─────────────
    VLLM_ARGS+=( --distributed-executor-backend ray )

    mapfile -t RAY_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    HEAD_NODE="${RAY_NODES[0]}"
    HEAD_IP="$(hostname -i 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+(\.[0-9]+){3}$' | head -1)"
    if [[ -z "${HEAD_IP}" ]]; then
        HEAD_IP="$(getent hosts "${HEAD_NODE}" | awk '{print $1; exit}')"
    fi
    RAY_PORT=6379
    RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    WANT_GPUS=$(( VLLM_TP * NODE_COUNT ))
    echo "[serve] multi-node: nodes=${NODE_COUNT} head=${HEAD_NODE} (${HEAD_IP}) want_gpus=${WANT_GPUS}"

    # Start the Ray head on this (head) node. --block keeps the head process
    # alive in the background while vLLM connects to the cluster separately.
    "${CONTAINER_CMD}" "${SING_COMMON[@]}" \
        "${SIF_IMAGE}" /opt/venv/bin/ray start --head \
            --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \
            --num-gpus="${VLLM_TP}" --temp-dir="${RUNTIME_CACHE}/ray" \
            --include-dashboard=false --block &
    RAY_HEAD_PID=$!

    # Wait for the head GCS to accept connections.
    for _ in $(seq 1 30); do
        if "${CONTAINER_CMD}" "${SING_COMMON[@]}" "${SIF_IMAGE}" \
                /opt/venv/bin/ray status --address "${RAY_ADDRESS}" >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done

    # Start Ray workers on the remaining nodes (one task per node).
    for ((i=1; i<${#RAY_NODES[@]}; i++)); do
        wnode="${RAY_NODES[$i]}"
        echo "[serve] starting ray worker on ${wnode}"
        srun --nodes=1 --ntasks=1 --nodelist="${wnode}" \
            "${CONTAINER_CMD}" "${SING_COMMON[@]}" \
                "${SIF_IMAGE}" /opt/venv/bin/ray start \
                    --address="${RAY_ADDRESS}" --num-gpus="${VLLM_TP}" \
                    --temp-dir="${RUNTIME_CACHE}/ray" --block &
        WORKER_PIDS+=($!)
    done

    # Wait until Ray reports every GPU across the cluster (TP × nodes).
    have=0
    for _ in $(seq 1 60); do
        have="$("${CONTAINER_CMD}" "${SING_COMMON[@]}" "${SIF_IMAGE}" \
            /opt/venv/bin/python -c "import ray; ray.init(address='${RAY_ADDRESS}'); print(int(ray.cluster_resources().get('GPU',0)))" \
            2>/dev/null | tail -1)"
        if [[ "${have}" == "${WANT_GPUS}" ]]; then
            break
        fi
        sleep 5
    done
    echo "[serve] ray cluster GPUs ready: ${have}/${WANT_GPUS}"

    write_job_info running
    "${CONTAINER_CMD}" "${SING_COMMON[@]}" \
        --env RAY_ADDRESS="${RAY_ADDRESS}" \
        "${SIF_IMAGE}" /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
            "${VLLM_ARGS[@]}" &
    CONTAINER_PID=$!
else
    # ── Single-node: run vLLM directly (proven path) ─────────────────────────
    write_job_info running
    "${CONTAINER_CMD}" "${SING_COMMON[@]}" \
        "${SIF_IMAGE}" /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
            "${VLLM_ARGS[@]}" &
    CONTAINER_PID=$!
fi

# ── Supervise: exit on container death or graceful stop request ──────────────
while kill -0 "${CONTAINER_PID}" >/dev/null 2>&1; do
    if [[ -f "${STOP_REQUEST_FILE}" ]]; then
        echo "[serve] graceful stop requested: ${STOP_REQUEST_FILE}"
        exit 0
    fi
    sleep 2
done

wait "${CONTAINER_PID}"
