#!/bin/bash
# ─── GLM-5.2-NVFP4 4-node validation (heterogeneous Slurm job) ───────────────
# The GB10 DGX-Spark partitions are four ISOLATED 2-node partitions
# (ng-dgx-m0..m3), so a single homogeneous job caps at 2 nodes. A Slurm
# HETEROGENEOUS job stitches components from several partitions into one
# allocation — validated 2026-07-13:
#   srun -p ng-dgx-m1 -N2 hostname : -p ng-dgx-m3 -N1 hostname   → 3 nodes, OK
# All GB10 nodes share the 10.129.2.x mgmt subnet (RJ45), cross-partition
# ping RTT ≈ 0.8 ms, so Ray/NCCL-socket traffic routes fine.
#
# Topology: 4 nodes × 1 GB10 GPU (121.63 GiB unified each) → TP=1, PP=4.
# Model: nvidia/GLM-5.2-NVFP4 — 753B total / 40B active MoE, NVFP4 weights,
# 465 GB on disk → ~116 GB weights/node at PP=4. VERY tight; we use
# fp8 KV cache + enforce-eager + small max-model-len for first validation.
#
# Topology per user: 2 nodes from ng-dgx-m1 + 2 nodes from ng-dgx-m2.
#
# Submit:  sbatch validate-glm52-nvfp4-4node.slurm
#SBATCH --job-name=glm52nvfp4-val
#SBATCH --account=cloud-users
#SBATCH --time=06:00:00
#SBATCH --output=/lvs0/rccs-dmp/masaru.nagaso/umrs_models/logs/glm52-nvfp4-val-%j.out
#SBATCH --partition=ng-dgx-m1
#SBATCH --nodes=2
#SBATCH hetjob
#SBATCH --partition=ng-dgx-m2
#SBATCH --nodes=2

set -u

MODELS=/lvs0/rccs-dmp/masaru.nagaso/umrs_models
MODEL_DIR="${MODELS}/GLM-5.2-NVFP4"
# cu130-nightly-aarch64 is a FROZEN old build whose transformers does NOT know
# `glm_moe_dsa` (job 208783 crash) and lacks the ModelOpt NVFP4 MoE method.
# :latest served Qwen3.6 NVFP4 on GB10 end-to-end (job 206940) — vLLM ≥0.23,
# transformers ≥5.3.
VLLM_IMAGE="docker.io/vllm/vllm-openai:latest"
RAY_PORT=6379
SERVED_NAME="glm-5.2-nvfp4"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_MEM_UTIL:-0.97}"

echo "[val] === GLM-5.2-NVFP4 4-node hetjob validation ==="

# ── Collect all nodes across het groups (node → het-group mapping kept for srun) ──
ALL_NODES=()
NODE_GROUP=()
if [[ -n "${SLURM_HET_SIZE:-}" ]]; then
    for ((g=0; g<SLURM_HET_SIZE; g++)); do
        _v="SLURM_JOB_NODELIST_HET_GROUP_${g}"
        while IFS= read -r _n; do
            ALL_NODES+=("${_n}")
            NODE_GROUP+=("${g}")
        done < <(scontrol show hostnames "${!_v}")
    done
else
    while IFS= read -r _n; do
        ALL_NODES+=("${_n}"); NODE_GROUP+=(0)
    done < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
fi
echo "[val] Nodes (${#ALL_NODES[@]}): ${ALL_NODES[*]}"
if [[ "${#ALL_NODES[@]}" -lt 4 ]]; then
    echo "[val] FATAL: expected 4 nodes, got ${#ALL_NODES[@]}" >&2
    exit 1
fi

HEAD_NODE="${ALL_NODES[0]}"
HEAD_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^10\.129\.2\.' | head -1)"
HEAD_IP="${HEAD_IP:-$(getent hosts "${HEAD_NODE}" | awk '{print $1; exit}')}"
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
WANT_GPUS=${#ALL_NODES[@]}
echo "[val] Head: ${HEAD_NODE} (${HEAD_IP}); Ray ${RAY_ADDRESS}; PP=${WANT_GPUS}"

# ── Shared podman args (mirrors validated _mn_vllm_args in serve-model.slurm) ──
PODMAN_ARGS=(
    --device /dev/nvidia0 --device /dev/nvidiactl
    --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools
    --device /dev/nvidia-caps/nvidia-cap1 --device /dev/nvidia-caps/nvidia-cap2
    --userns host
    --cap-add SYS_NICE
    -v "${MODELS}/:/root/.cache/huggingface:ro"
    -v "${MODEL_DIR}:/model:ro"
    -v /usr/lib/aarch64-linux-gnu/libcuda.so.1:/usr/lib/aarch64-linux-gnu/libcuda.so.1:ro
    -v /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:ro
    -v /usr/lib/aarch64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro
    -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    -e TRITON_LIBCUDA_PATH=/usr/lib/aarch64-linux-gnu
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
    -e CUDA_VISIBLE_DEVICES=0
    -e NCCL_CUMEM_ENABLE=0
    -e NCCL_DEBUG=INFO
    # GB10 NVFP4 MoE backend (same as the Qwen3.6-NVFP4 fix): without it the
    # experts fall back to the Unquantized MoE backend → KeyError w2_input_scale.
    -e VLLM_MOE_BACKEND=marlin
    -e RAY_memory_usage_threshold=0.98
    -e VLLM_LOGGING_LEVEL=INFO
)
RAY_PIP='export PATH=/usr/local/bin:$PATH; python3 -m pip install -q --no-index --find-links /root/.cache/huggingface/ray_aarch64_env/wheels ray 2>/dev/null || python3 -m pip install -q ray'

# Hetjob gotcha: each component has its OWN raw job ID (leader+1, +2, …) and
# its prolog creates /tmp/job-<component-id> on its nodes — but srun steps
# export the LEADER's XDG_RUNTIME_DIR=/tmp/job-<leader-id>/xdg, which does NOT
# exist on non-leader components' nodes. Rootless podman then dies with
# "error creating temporary file" / "invalid internal status". Fix: point
# XDG_RUNTIME_DIR at a self-created per-user dir on every remote node.
XDG_FIX='export XDG_RUNTIME_DIR=/tmp/xdg-glm-$(id -u); mkdir -p "$XDG_RUNTIME_DIR"; podman system migrate >/dev/null 2>&1 || true'
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no)

cleanup() {
    echo "[val] Cleaning up containers on all nodes..."
    podman rm -f glm-nvfp4-head >/dev/null 2>&1 || true
    for ((i=1; i<${#ALL_NODES[@]}; i++)); do
        # Best effort: ssh first (may be denied once the extern step is gone),
        # then a short srun step. `podman rm -f` also reaps orphaned conmon.
        ssh "${SSH_OPTS[@]}" "${ALL_NODES[$i]}" \
            "export XDG_RUNTIME_DIR=/tmp/xdg-glm-\$(id -u); podman rm -f glm-nvfp4-worker; pkill -u \$(id -u) -f 'ray/raylet' || true" >/dev/null 2>&1 \
        || srun --het-group="${NODE_GROUP[$i]}" --nodes=1 --ntasks=1 \
                --nodelist="${ALL_NODES[$i]}" --overlap \
            bash -c "${XDG_FIX}; podman rm -f glm-nvfp4-worker; pkill -u \$(id -u) -f 'ray/raylet' || true" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT TERM INT

# ── Ray head + co-located vLLM driver on node 0 ──
echo "[val] Starting Ray head + vLLM on ${HEAD_NODE}..."
podman rm -f glm-nvfp4-head >/dev/null 2>&1 || true
podman run -d --name glm-nvfp4-head --network host --shm-size=4g \
    --entrypoint /bin/sh \
    "${PODMAN_ARGS[@]}" \
    "${VLLM_IMAGE}" \
    -c "${RAY_PIP}; ray start --head --node-ip-address='${HEAD_IP}' --port='${RAY_PORT}' --num-gpus=1 --object-store-memory=1073741824 --include-dashboard=false; echo '[head] waiting for ${WANT_GPUS} GPUs...'; i=0; while [ \$i -lt 120 ]; do g=\$(ray status 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | head -1 | sed 's@.*/@@; s@ GPU@@; s@[.].*@@'); [ \"\${g:-0}\" = '${WANT_GPUS}' ] && break; sleep 5; i=\$((i+1)); done; echo \"[head] GPUs=\${g:-0}; launching vLLM\"; exec vllm serve /model --served-model-name '${SERVED_NAME}' -tp 1 -pp ${WANT_GPUS} --distributed-executor-backend ray --host 0.0.0.0 --port 8000 --max-model-len ${MAX_LEN} --gpu-memory-utilization ${GPU_UTIL} --kv-cache-dtype fp8_e4m3 --enforce-eager --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice" \
    || { echo "[val] FATAL: head container launch failed" >&2; exit 1; }

# Wait for Ray GCS
_ready=0
for _ in $(seq 1 300); do
    if (exec 3<>"/dev/tcp/${HEAD_IP}/${RAY_PORT}") 2>/dev/null; then
        exec 3>&- 3<&- 2>/dev/null || true; _ready=1; break
    fi
    sleep 2
done
[[ "${_ready}" -eq 1 ]] || { echo "[val] FATAL: Ray GCS not ready" >&2; podman logs --tail 40 glm-nvfp4-head; exit 1; }
echo "[val] Ray head GCS up."

# Stream the FULL head container log to Lustre — the FATAL 120-line tail window
# lost the primary EngineCore error on job 208788.
HEADLOG="${MODELS}/logs/glm52-nvfp4-head-${SLURM_JOB_ID}.log"
( podman logs -f glm-nvfp4-head >"${HEADLOG}" 2>&1 ) &
echo "[val] Head container log → ${HEADLOG}"

# ── Ray workers on nodes 1..3 (srun + foreground podman, sig-proxy OFF) ──
# Launch-path history (4 failed variants before this):
#   • srun + foreground + default --sig-proxy=true (208788/208793): this site
#     CANCELS srun steps ~75-120s in → SIGTERM to the podman CLIENT → sig-proxy
#     FORWARDS it into the container → `ray start` (PID1, handles SIGTERM) dies.
#     A plain sh sleeper survived the same cancel (PID1 ignores default-action
#     signals) — proving containers escape the step cgroup via conmon.
#   • srun + `podman run -d` (208796): step-end cgroup cleanup reaped the
#     container instantly → Ray never saw 4 GPUs → placement timeout.
#   • ssh + `podman run -d` (208801): container survived but GCS marked the
#     worker nodes dead ~1 min after registration (pam session teardown
#     side-effects) → cluster stuck at 1 GPU.
# THIS variant: srun + FOREGROUND podman + --sig-proxy=false. When the step
# killer fires, only the podman client dies; conmon + container + ray live on
# for the job duration. The srun exit is expected and ignored.
for ((i=1; i<${#ALL_NODES[@]}; i++)); do
    wnode="${ALL_NODES[$i]}"; wgroup="${NODE_GROUP[$i]}"
    echo "[val] $(date +%H:%M:%S) launching Ray worker on ${wnode} (het group ${wgroup}, sig-proxy off)..."
    (
        srun --het-group="${wgroup}" --nodes=1 --ntasks=1 \
             --nodelist="${wnode}" \
            bash -c "${XDG_FIX}; exec podman run \"\$@\"" _ \
                --sig-proxy=false \
                --name glm-nvfp4-worker --replace --network host --shm-size=4g \
                --entrypoint /bin/sh \
                "${PODMAN_ARGS[@]}" \
                "${VLLM_IMAGE}" \
                -c "${RAY_PIP}; python3 -c 'import pynvml; pynvml.nvmlInit(); assert pynvml.nvmlDeviceGetCount() > 0' && exec ray start --address='${RAY_ADDRESS}' --num-gpus=1 --object-store-memory=1073741824 --block"
        echo "[val] $(date +%H:%M:%S) worker srun on ${wnode} returned rc=$? (expected: step killer; container should persist)"
    ) &
done

# ── Wait for vLLM readiness (model load: ~116 GB/node from Lustre) ──
echo "[val] Waiting for vLLM /v1/models (up to 90 min)..."
_up=0
for _ in $(seq 1 540); do
    if curl -sf -m 5 "http://${HEAD_IP}:8000/v1/models" >/dev/null 2>&1; then
        _up=1; break
    fi
    if ! podman ps --filter name=glm-nvfp4-head --format '{{.Names}}' | grep -q glm-nvfp4-head; then
        echo "[val] FATAL: head container died during load" >&2
        podman logs --tail 120 glm-nvfp4-head 2>&1 | sed 's/^/[val][head] /'
        exit 1
    fi
    sleep 10
done
if [[ "${_up}" -ne 1 ]]; then
    echo "[val] FATAL: vLLM never became ready" >&2
    podman logs --tail 120 glm-nvfp4-head 2>&1 | sed 's/^/[val][head] /'
    exit 1
fi
echo "[val] ✅ vLLM READY at http://${HEAD_IP}:8000"

# ── Smoke inference + crude throughput measurement ──
echo "[val] Smoke test:"
_t0=$(date +%s)
curl -s -m 600 "http://${HEAD_IP}:8000/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain in 3 sentences why the sky is blue.\"}],\"max_tokens\":256}" \
    | python3 -c "
import json,sys
r=json.load(sys.stdin)
u=r.get('usage',{})
print('[val] completion_tokens=',u.get('completion_tokens'),' total=',u.get('total_tokens'))
print('[val] text:', r['choices'][0]['message'].get('content','')[:400])
"
_t1=$(date +%s)
echo "[val] wall: $((_t1-_t0))s"

echo "[val] Serving until walltime. Endpoint: http://${HEAD_IP}:8000/v1 (model '${SERVED_NAME}', no API key)"
# Workers are detached containers — nothing to wait on; sleep out the walltime
# while keeping the batch step alive (cleanup trap fires at job end).
sleep infinity
