#!/bin/bash
# =============================================================================
# serve-model-multinode.slurm — BDRT custom-image multinode vLLM launcher
# =============================================================================
# Submitted by SlurmPodmanBackend when deploy_runtime.type=custom-image and
# deploy_runtime.node_count > 1.  Coordinates *node_count* independent sbatch
# jobs (one per compute node) and launches the vendor-supplied vLLM image
# in tensor-parallel mode via Ray on each node.
#
# Usage:
#   sbatch --nodes=<N> --job-name=umrs-rivault serve-model-multinode.slurm
#
# Required environment variables (set by SlurmPodmanBackend or site.yaml):
#   UMRS_DEPLOY_RUNTIME_IMAGE  - Full OCI image reference (e.g. nvcr.io/...)
#   UMRS_DEPLOY_NODE_COUNT     - Number of nodes (default 2)
#   VLLM_MODEL                 - Model path or name
#   VLLM_MAX_MODEL_LEN         - Context window (default 8192)
#   VLLM_GPU_MEM_UTIL          - GPU memory fraction (default 0.97)
#   VLLM_SERVED_MODEL_NAME     - User-facing model name (default from VLLM_MODEL)
#   LITELLM_MASTER_KEY         - Auth key (required)
#   UMRS_SITE_CONFIG           - Path to site.yaml (optional)
#
# Optional:
#   VLLM_EXTRA_ARGS            - Additional vLLM CLI arguments
#   RIVAULT_DIR                - Shared output directory base
# =============================================================================

set -euo pipefail

# ─── Resolve deployment image ────────────────────────────────────────────────
IMAGE="${UMRS_DEPLOY_RUNTIME_IMAGE:?UMRS_DEPLOY_RUNTIME_IMAGE not set}"
NODE_COUNT="${UMRS_DEPLOY_NODE_COUNT:-2}"
MODEL_PATH="${VLLM_MODEL:?VLLM_MODEL not set}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_MEM_UTIL:-0.97}"
SERVED_NAME="${VLLM_SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}"
MASTER_KEY="${LITELLM_MASTER_KEY:?LITELLM_MASTER_KEY not set}"
RIVAULT_DIR="${RIVAULT_DIR:-/tmp/umrs-rivault-${SLURM_JOB_ID}}"
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# ─── Render extra env from UMRS_DEPLOY_EXTRA_* vars ──────────────────────────
_extras=""
for _var in $(env | grep '^UMRS_DEPLOY_EXTRA_' | sed 's/_.*//; s/^UMRS_DEPLOY_EXTRA_//'); do
    _key="$(echo "$_var" | tr '[:upper:]' '[:lower:]')"
    _val="${UMRS_DEPLOY_EXTRA_${_var}:-}"
    if [[ -n "$_val" ]]; then
        _extras="${_extras} -e ${_key}=${_val}"
    fi
done

RUN_ID="$(date +%s)-$$"
RDV="${RIVAULT_DIR}/rendezvous/${RUN_ID}"
mkdir -p "${RDV}"

# ─── Output node addresses for UMRS discovery ────────────────────────────────
echo "=============================="
echo "MULTINODE JOB: ${SLURM_JOB_ID}"
echo "NODES: ${NODE_COUNT}"
echo "IMAGE: ${IMAGE}"
echo "MODEL: ${MODEL_PATH} (served as ${SERVED_NAME})"
echo "Rendezvous: ${RDV}"
echo "=============================="

# ─── Determine our node role ─────────────────────────────────────────────────
# Slurm assigns nodelist as a comma-separated list when --nodes > 1.
# We use the first node as the Ray head; the rest are workers.
IFS=',' read -ra NODELIST <<< "${SLURM_NODELIST:-}"
MY_HOST="$(hostname -f)"

# Find our index in the node list to determine head vs worker
MY_ROLE="worker"
for i in "${!NODELIST[@]}"; do
    if [[ "${NODELIST[$i]}" == "$(hostname -s)" || "${NODELIST[$i]}" == "${MY_HOST}"* ]]; then
        if [[ "$i" -eq 0 ]]; then
            MY_ROLE="head"
        fi
        break
    fi
done

# Resolve our IP on the Slurm network
MY_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | head -1)"
if [[ -z "${MY_IP:-}" ]]; then
    MY_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi

# ─── GPU device args (BDRT: GB10 / Blackwell) ───────────────────────────────
# Detect available NVIDIA devices
GPU_DEVS=""
for _d in $(ls /dev/nvidia* 2>/dev/null | head -64); do
    [[ -e "$_d" ]] && GPU_DEVS="${GPU_DEVS} --device ${_d}:${_d}:rwm"
done

# GPU memory per node for --shm-size (roughly 4 GiB per GPU)
GPU_COUNT=0
if [[ -d /dev/nvidia* ]]; then
    GPU_COUNT="$(ls /dev/nvidia[0-9]* 2>/dev/null | wc -l | tr -d ' ')"
fi
[[ -z "${GPU_COUNT}" || "${GPU_COUNT}" -eq 0 ]] && GPU_COUNT=1

SHM_SIZE="$(( GPU_COUNT * 4 ))g"

# Library mounts needed for Blackwell / GB10 GPU support
LIBS="-v /usr/lib/aarch64-linux-gnu/libcuda.so.1:/usr/lib/aarch64-linux-gnu/libcuda.so.1:ro \
      -v /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:ro \
      -v /usr/lib/aarch64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro"

# ─── Podman storage isolation per job ───────────────────────────────────────
_WHOAMI="$(whoami)"
_XDG="${XDG_RUNTIME_DIR:-/tmp/job-${SLURM_JOB_ID}/xdg}"
mkdir -p "${_XDG}/containers/storage"
_PODGRAPH="/var/tmp/podman_${_WHOAMI}/multinode-${RUN_ID}"
mkdir -p "${_PODGRAPH}"
export CONTAINERS_STORAGE_CONF="${_XDG}/storage-glm.conf"
cat > "${CONTAINERS_STORAGE_CONF}" <<EOF
[storage]
driver="overlay"
runroot = "${_XDG}/containers/storage"
rootless_storage_path="${_PODGRAPH}"
graphroot = "${_PODGRAPH}"
[storage.options]
mount_program = "/usr/bin/fuse-overlayfs"
[storage.options.overlay]
mountopt="xattr_permissions=2,threaded=1"
EOF
podman system migrate >/dev/null 2>&1 || true
echo "[multinode] graphroot=${_PODGRAPH} runroot=${_XDG}/containers/storage"

# ─── Cleanup ─────────────────────────────────────────────────────────────────
CTR="multinode-ctr-${ROLE:-${MY_ROLE}}"
cleanup() {
    if [[ "${MULTINODE_KEEP:-0}" -eq 1 ]]; then
        echo "[multinode] KEEPING ${CTR} for postmortem"
    else
        echo "[multinode] removing ${CTR}"
        podman rm -f "${CTR}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT TERM INT

# ─── HEAD NODE: publish rendezvous IP and launch vLLM ────────────────────────
if [[ "${MY_ROLE}" == "head" ]]; then
    echo "${MY_IP}" > "${RDV}/head.ip"
    echo "[multinode] HEAD published rendezvous: ${RDV}/head.ip (${MY_IP})"

    HEADLOG="${RIVAULT_DIR}/logs/head-${SLURM_JOB_ID}.log"
    mkdir -p "$(dirname "${HEADLOG}")"

    # Launch the custom vLLM image as Ray head + vLLM driver
    podman run -d --name "${CTR}" --replace --network host --shm-size="${SHM_SIZE}" \
        ${GPU_DEVS} ${LIBS} \
        --userns host \
        --cap-add SYS_NICE \
        --log-driver=k8s-file --log-opt "path=${HEADLOG}" \
        -e TRANSFORMERS_OFFLINE=1 \
        -e HF_HUB_OFFLINE=1 \
        -e VLLM_NO_USAGE_STATS=1 \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
        -e RAY_memory_usage_threshold=0.98 \
        -e VLLM_LOGGING_LEVEL=INFO \
        -e NGINX_LITELLM_DOCKER_PORT=8001 \
        ${_extras} \
        "${IMAGE}" \
        /bin/sh -c "
            echo '[head] starting Ray head + vLLM';
            vllm serve '${MODEL_PATH}' \
                --distributed-executor-backend ray \
                --host 0.0.0.0 \
                --port 8000 \
                --tensor-parallel-size ${NODE_COUNT} \
                --max-model-len ${MAX_LEN} \
                --gpu-memory-utilization ${GPU_UTIL} \
                --served-model-name '${SERVED_NAME}' \
                ${EXTRA_ARGS}
        " || { echo "[multinode] FATAL: head container launch failed" >&2; exit 1; }

    # Verify head container started
    sleep 5
    if [[ "$(podman inspect "${CTR}" --format '{{.State.Status}}' 2>/dev/null)" != "running" ]]; then
        echo "[multinode] FATAL: head container not running" >&2
        tail -30 "${HEADLOG}" 2>/dev/null || true
        MULTINODE_KEEP=1
        exit 1
    fi
    echo "[multinode] head log → ${HEADLOG}"

    # Wait for all nodes to join Ray cluster, then confirm vLLM ready
    _up=0
    for _i in $(seq 1 300); do
        if curl -sf -m 5 "http://${MY_IP}:8000/v1/models" >/dev/null 2>&1; then
            _up=1
            break
        fi
        sleep 2
    done
    [[ "${_up}" -eq 1 ]] || { echo "[multinode] FATAL: vLLM not ready" >&2; MULTINODE_KEEP=1; exit 1; }
    echo "[multinode] ✅ vLLM READY at http://${MY_IP}:8000 (model: ${SERVED_NAME}, tp=${NODE_COUNT})"

    # Write endpoint info for UMRS discovery
    cat > "${RIVAULT_DIR}/endpoint.info" <<EOF
RIVAULT_STATUS=running
RIVAULT_NODE=${MY_IP}
RIVAULT_LITELLM_PORT=8001
RIVAULT_VLLM_PORT=8000
RIVAULT_MODEL=${MODEL_PATH}
EOF
else
    # ─── WORKER NODE: discover head IP and join Ray ─────────────────────────
    _wait=0
    while [[ ! -f "${RDV}/head.ip" ]]; do
        sleep 1
        _wait=$((_wait + 1))
        if [[ $_wait -gt 60 ]]; then
            echo "[multinode] FATAL: head rendezvous not ready after ${_wait}s" >&2
            MULTINODE_KEEP=1
            exit 1
        fi
    done

    HEAD_IP="$(cat "${RDV}/head.ip")"
    echo "[multinode] WORKER found head at ${HEAD_IP}"

    WORKERLOG="${RIVAULT_DIR}/logs/worker-${SLURM_JOB_ID}.log"
    mkdir -p "$(dirname "${WORKERLOG}")"

    podman run -d --name "${CTR}" --replace --network host --shm-size="${SHM_SIZE}" \
        ${GPU_DEVS} ${LIBS} \
        --userns host \
        --cap-add SYS_NICE \
        --log-driver=k8s-file --log-opt "path=${WORKERLOG}" \
        -e TRANSFORMERS_OFFLINE=1 \
        -e HF_HUB_OFFLINE=1 \
        -e VLLM_NO_USAGE_STATS=1 \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
        -e RAY_memory_usage_threshold=0.98 \
        -e VLLM_LOGGING_LEVEL=INFO \
        ${_extras} \
        "${IMAGE}" \
        /bin/sh -c "
            echo '[worker] joining Ray cluster at ${HEAD_IP}';
            vllm serve '${MODEL_PATH}' \
                --distributed-executor-backend ray \
                --host 0.0.0.0 \
                --port 8000 \
                --tensor-parallel-size ${NODE_COUNT} \
                --max-model-len ${MAX_LEN} \
                --gpu-memory-utilization ${GPU_UTIL} \
                --served-model-name '${SERVED_NAME}' \
                --tokenizer ${HEAD_IP}:8000 \
                ${EXTRA_ARGS}
        " || { echo "[multinode] FATAL: worker container launch failed" >&2; exit 1; }

    sleep 5
    if [[ "$(podman inspect "${CTR}" --format '{{.State.Status}}' 2>/dev/null)" != "running" ]]; then
        echo "[multinode] FATAL: worker container not running" >&2
        tail -30 "${WORKERLOG}" 2>/dev/null || true
        MULTINODE_KEEP=1
        exit 1
    fi
    echo "[multinode] worker log → ${WORKERLOG}"
fi

# Keep batch step alive until Slurm kills us
wait
