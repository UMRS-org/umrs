#!/bin/bash
# ─── GLM-5.2-NVFP4 4-node serving: SINGULARITY node script ───────────────────
# One instance of this script = one sbatch job = one node = one Ray member.
# Submit all four via ./submit-glm52-nvfp4.sh (head + 3 workers, Lustre
# rendezvous). See git history for the podman variant and the full saga:
#
#   • Hetjob/srun/ssh worker launches all die on this site (slurm tears down
#     non-batch cgroups in ~40-120s, SIGTERM kills ray; batch step is safe).
#   • Podman itself is unusable here: serve-model.slurm rewrites the SHARED
#     ~/.config/containers/storage.conf per job (dead runroots → "Transport
#     endpoint is not connected", pause-process death), the shared graphroot
#     DB is corrupted, and fresh graphroots can't even pull (lsetxattr /boot
#     fails with fuse-overlayfs on these nodes).
#   • Singularity has NO state DB / daemon / fuse mounts — processes are plain
#     children of the batch step. SIF lives on Lustre, shared by all nodes.
#
# Requires: /lvs0/.../umrs_models/_images/vllm-openai-latest.sif
#           (singularity pull docker://vllm/vllm-openai:latest on login node)
#
# Do not sbatch directly — use ./submit-glm52-nvfp4.sh
#SBATCH --job-name=glm52nvfp4
#SBATCH --account=cloud-users
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --output=/lvs0/rccs-dmp/masaru.nagaso/umrs_models/logs/glm52-nvfp4-%x-%j.out

set -u

MODELS=/lvs0/rccs-dmp/masaru.nagaso/umrs_models
MODEL_DIR="${MODELS}/GLM-5.2-NVFP4"
SIF="${MODELS}/_images/vllm-openai-latest.sif"
RAY_PORT=6379
SERVED_NAME="glm-5.2-nvfp4"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
# GB10 unified memory: OS + page cache + FUSE SIF mount eat ~26GB, so only
# ~95GB of the 121.63GiB pool is free to CUDA at startup (job 208909).
# 0.75 × 121.63 = 91.2GB per node; PP=6 weights ≈ 78GB/node + KV fits.
GPU_UTIL="${VLLM_GPU_MEM_UTIL:-0.75}"
RDV="${MODELS}/logs/rdv"
ROLE="${GLM_ROLE:?GLM_ROLE not set (head|worker)}"
RUN_ID="${GLM_RUN_ID:?GLM_RUN_ID not set}"
TOTAL_GPUS="${GLM_TOTAL_GPUS:-4}"
mkdir -p "${RDV}"

[[ -f "${SIF}" ]] || { echo "[glm] FATAL: SIF not found: ${SIF}" >&2; exit 1; }

MY_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^10\.129\.2\.' | head -1)"
echo "[glm] role=${ROLE} run=${RUN_ID} node=$(hostname -s) ip=${MY_IP} job=${SLURM_JOB_ID} sif=${SIF}"

# Environment for the container (offline, GB10 CUDA quirks, NVFP4 MoE backend)
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_VLLM_NO_USAGE_STATS=1
export SINGULARITYENV_TRITON_LIBCUDA_PATH=/usr/lib/aarch64-linux-gnu
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SINGULARITYENV_RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
export SINGULARITYENV_NCCL_CUMEM_ENABLE=0
export SINGULARITYENV_NCCL_DEBUG=INFO
# The GB10 pairs have RoCE NICs (rocep1s0f1) that connect ONLY within each
# 2-node pair. NCCL auto-picks NET/IB(RoCE) and cross-pair init dies with
# "unhandled system error" (job 208903). Force TCP sockets on the mgmt RJ45
# interface, which routes across all four nodes.
export SINGULARITYENV_NCCL_IB_DISABLE=1
export SINGULARITYENV_NCCL_SOCKET_IFNAME=enP7s7
export SINGULARITYENV_VLLM_MOE_BACKEND=marlin
export SINGULARITYENV_RAY_memory_usage_threshold=0.98
export SINGULARITYENV_VLLM_LOGGING_LEVEL=INFO
# Ray/vLLM write under $HOME and /tmp inside container; keep them node-local.
export SINGULARITYENV_RAY_TMPDIR=/tmp/ray-glm
# Shared $HOME is bind-mounted by default → ~/.local/lib/python3.12/site-packages
# LEAKS into the container (old flashinfer 0.6.12 there broke vllm import:
# "flashinfer-jit-cache version (0.6.13+cu130) does not match"). Kill user-site.
export SINGULARITYENV_PYTHONNOUSERSITE=1
# torchcodec's libtorchcodec_core*.so lack an rpath to torch's libs; docker
# provides them via image ENV, but singularity's env handling drops that.
# Without this, vllm import dies with "Could not load libtorchcodec_core*.so".
export SINGULARITYENV_LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/.singularity.d/libs"

SBIND=(
    # --userns is REQUIRED on GB10: node kernels have NO squashfs support, so
    # the setuid kernel-mount path dies with "bad superblock". Unprivileged
    # userns forces a squashfuse FUSE mount, which works (validated job 208856).
    --userns
    # DO NOT use --nv: it binds ALL host GL/nvidia libs into
    # /.singularity.d/libs, and the host's libGLdispatch.so.0 requires
    # GLIBC_2.38 which the container glibc lacks → every dlopen that touches
    # GL (torchcodec/ffmpeg) dies. Bind ONLY the driver libs the CUDA stack
    # needs (same trio as the validated podman path). /dev is host-bound by
    # default, so the GPU device nodes are already visible.
    --bind /usr/lib/aarch64-linux-gnu/libcuda.so.1:/usr/lib/aarch64-linux-gnu/libcuda.so.1
    --bind /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1
    --bind /usr/lib/aarch64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ptxjitcompiler.so.1
    --bind "${MODELS}:/root/.cache/huggingface:ro"
    --bind "${MODEL_DIR}:/model:ro"
    --writable-tmpfs
)
# Ray is not bundled in the vllm image. Install from the offline wheel cache
# into a node-local --target dir (/tmp = host /tmp; user-site is disabled and
# writable-tmpfs is too small for ray). PYTHONPATH+PATH pick it up.
# LD_LIBRARY_PATH must be set INSIDE the container shell: SINGULARITYENV_
# LD_LIBRARY_PATH is clobbered by --nv processing (validated: in-shell export
# fixes torchcodec, the env-var route does not).
RAY_PIP='export PYTHONNOUSERSITE=1; export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}; _RT=/tmp/raylib-glm; python3 -m pip install -q --no-index --find-links /root/.cache/huggingface/ray_aarch64_env/wheels --target "$_RT" ray 2>/dev/null || python3 -m pip install -q --target "$_RT" ray; export PYTHONPATH="$_RT:${PYTHONPATH:-}" PATH="$_RT/bin:$PATH"'

cleanup() {
    echo "[glm] cleanup (${ROLE})"
    [[ "${ROLE}" == "head" ]] && rm -f "${RDV}/head-${RUN_ID}.ip"
    pkill -u "$(id -u)" -f 'ray/raylet' 2>/dev/null || true
}
trap cleanup EXIT TERM INT

if [[ "${ROLE}" == "head" ]]; then
    RAY_ADDRESS="${MY_IP}:${RAY_PORT}"
    HEADLOG="${MODELS}/logs/glm52-nvfp4-head-${SLURM_JOB_ID}.log"
    echo "${MY_IP}" > "${RDV}/head-${RUN_ID}.ip"
    echo "[glm] rendezvous published: ${RDV}/head-${RUN_ID}.ip"
    echo "[glm] head log → ${HEADLOG}"

    # Ray head + GPU gate + co-located vLLM driver, as a plain background
    # process of the batch step (batch cgroup = safe from the step killer).
    singularity exec "${SBIND[@]}" "${SIF}" /bin/sh -c "
${RAY_PIP}
ray start --head --node-ip-address='${MY_IP}' --port='${RAY_PORT}' --num-gpus=1 --object-store-memory=1073741824 --include-dashboard=false
echo '[head] waiting for ${TOTAL_GPUS} GPUs...'
i=0; while [ \$i -lt 240 ]; do
  g=\$(ray status 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | head -1 | sed 's@.*/@@; s@ GPU@@; s@[.].*@@')
  [ \"\${g:-0}\" = '${TOTAL_GPUS}' ] && break
  sleep 5; i=\$((i+1))
done
echo \"[head] GPUs=\${g:-0}; launching vLLM\"
exec vllm serve /model --served-model-name '${SERVED_NAME}' -tp 1 -pp ${TOTAL_GPUS} --distributed-executor-backend ray --host 0.0.0.0 --port 8000 --max-model-len ${MAX_LEN} --gpu-memory-utilization ${GPU_UTIL} --kv-cache-dtype fp8_e4m3 --enforce-eager --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
" >"${HEADLOG}" 2>&1 &
    HEAD_PID=$!

    # Readiness watch
    _up=0
    for _ in $(seq 1 540); do
        if curl -sf -m 5 "http://${MY_IP}:8000/v1/models" >/dev/null 2>&1; then _up=1; break; fi
        if ! kill -0 "${HEAD_PID}" 2>/dev/null; then
            echo "[glm] FATAL: head process exited" >&2
            tail -60 "${HEADLOG}" | sed 's/^/[glm][head] /'
            exit 1
        fi
        sleep 10
    done
    [[ "${_up}" -eq 1 ]] || { echo "[glm] FATAL: vLLM never ready" >&2; tail -60 "${HEADLOG}" | sed 's/^/[glm][head] /'; exit 1; }
    echo "[glm] ✅ vLLM READY at http://${MY_IP}:8000  (model '${SERVED_NAME}')"

    curl -s -m 600 "http://${MY_IP}:8000/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain in 3 sentences why the sky is blue.\"}],\"max_tokens\":256}" \
        | python3 -c "
import json,sys
r=json.load(sys.stdin)
u=r.get('usage',{})
print('[glm] smoke: completion_tokens=',u.get('completion_tokens'))
print('[glm] text:', r['choices'][0]['message'].get('content','')[:300])
" || echo "[glm] smoke test failed (server may still be usable)"

    echo "[glm] serving until walltime"
    wait "${HEAD_PID}"
    echo "[glm] head process exited; job ends"
else
    echo "[glm] waiting for head rendezvous (up to 15 min)..."
    HEAD_IP=""
    for _ in $(seq 1 180); do
        [[ -f "${RDV}/head-${RUN_ID}.ip" ]] && { HEAD_IP="$(cat "${RDV}/head-${RUN_ID}.ip")"; break; }
        sleep 5
    done
    [[ -n "${HEAD_IP}" ]] || { echo "[glm] FATAL: no head rendezvous for run ${RUN_ID}" >&2; exit 1; }
    RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    echo "[glm] joining Ray cluster at ${RAY_ADDRESS}"

    for _att in 1 2 3 4 5; do
        singularity exec "${SBIND[@]}" "${SIF}" /bin/sh -c "
${RAY_PIP}
python3 -c 'import pynvml; pynvml.nvmlInit(); assert pynvml.nvmlDeviceGetCount() > 0' || exit 7
exec ray start --address='${RAY_ADDRESS}' --num-gpus=1 --object-store-memory=1073741824 --block
"
        rc=$?
        echo "[glm] $(date +%H:%M:%S) worker exited rc=${rc} (attempt ${_att}/5); retry in 15s"
        sleep 15
        [[ -f "${RDV}/head-${RUN_ID}.ip" ]] || { echo "[glm] head rendezvous gone; exiting"; break; }
    done
fi
