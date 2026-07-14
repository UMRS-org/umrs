#!/bin/bash
# ─── GLM-5.2-NVFP4 4-node serving via FOUR INDEPENDENT 1-node sbatch jobs ────
#
# WHY NOT hetjob/srun/ssh (all validated to FAIL on this site, 2026-07-13):
#   This site's Slurm tears down every non-batch cgroup vehicle within
#   ~40-120s, delivering SIGTERM to everything inside:
#     • srun steps: "slurmstepd: *** STEP CANCELLED ***" (~75-120s, even for a
#       plain sleeper — test job 208795); sig-proxy=false does NOT help because
#       the SIGTERM is cgroup-wide, not client-proxied (job 208803).
#     • srun + podman -d: container reaped at step end (job 208796).
#     • ssh (pam_slurm_adopt extern step): worker raylets marked dead ~1 min
#       after registering (job 208801); repeat ssh often denied.
#   Shell PID1 sleepers "survive" these teardowns only because PID1 ignores
#   SIGTERM by default; `ray start` handles SIGTERM and dies.
#   ONLY the BATCH step survives long-term (head container lived 57+ min in
#   208796). So: each node gets its OWN sbatch job, and each job runs its Ray
#   container in its own batch step.
#
# COORDINATION: Lustre rendezvous dir. The head job writes head-<RUN_ID>.ip;
# worker jobs poll for it, then join the Ray cluster. GLM_ROLE + GLM_RUN_ID
# arrive via sbatch --export from submit-glm52-nvfp4.sh.
#
# Do not sbatch this directly — use ./submit-glm52-nvfp4.sh
#SBATCH --job-name=glm52nvfp4
#SBATCH --account=cloud-users
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --output=/lvs0/rccs-dmp/masaru.nagaso/umrs_models/logs/glm52-nvfp4-%x-%j.out

set -u

MODELS=/lvs0/rccs-dmp/masaru.nagaso/umrs_models
MODEL_DIR="${MODELS}/GLM-5.2-NVFP4"
VLLM_IMAGE="docker.io/vllm/vllm-openai:latest"
RAY_PORT=6379
SERVED_NAME="glm-5.2-nvfp4"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_MEM_UTIL:-0.97}"
RDV="${MODELS}/logs/rdv"
ROLE="${GLM_ROLE:?GLM_ROLE not set (head|worker)}"
RUN_ID="${GLM_RUN_ID:?GLM_RUN_ID not set}"
TOTAL_GPUS="${GLM_TOTAL_GPUS:-4}"
mkdir -p "${RDV}"

MY_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^10\.129\.2\.' | head -1)"
echo "[glm] role=${ROLE} run=${RUN_ID} node=$(hostname -s) ip=${MY_IP} job=${SLURM_JOB_ID}"

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
    -e VLLM_MOE_BACKEND=marlin
    -e RAY_memory_usage_threshold=0.98
    -e VLLM_LOGGING_LEVEL=INFO
)
RAY_PIP='export PATH=/usr/local/bin:$PATH; python3 -m pip install -q --no-index --find-links /root/.cache/huggingface/ray_aarch64_env/wheels ray 2>/dev/null || python3 -m pip install -q ray'

CTR="glm-nvfp4-${ROLE}"
KEEP_CTR=0
cleanup() {
    if [[ "${KEEP_CTR}" -eq 1 ]]; then
        echo "[glm] cleanup: KEEPING ${CTR} for postmortem (FATAL path)"
    else
        echo "[glm] cleanup: removing ${CTR}"
        podman rm -f "${CTR}" >/dev/null 2>&1 || true
    fi
    [[ "${ROLE}" == "head" ]] && rm -f "${RDV}/head-${RUN_ID}.ip"
}
trap cleanup EXIT TERM INT

# ── Podman storage setup + stale-state purge (mirrors serve-model.slurm) ─────
# Background: serve-model.slurm rewrites ~/.config/containers/storage.conf
# (home = SHARED Lustre) pinning runroot to ITS job's /tmp/job-<id>/xdg. After
# that job dies, every podman inheriting the file gets a dead runroot →
# "Transport endpoint is not connected", pause-process death, DB runroot
# mismatch ("database run root ... does not match"). Production survives by
# running a purge ritual at job start; we do the same but keep our config
# PRIVATE via CONTAINERS_STORAGE_CONF on the node-local job tmp dir (this is
# a single-node job, so Slurm's XDG_RUNTIME_DIR=/tmp/job-<id>/xdg is valid).
_WHOAMI="$(whoami)"
# CLEAN-SLATE graphroot: the production graphroot (/var/tmp/podman_<user>/storage)
# has a corrupted state DB (runroot pinned to dead /tmp/job-208716, cross-node
# clobbering via the shared-home storage.conf). Attempts to reuse it produced
# "Transport endpoint is not connected" fuse deaths ~2.5 min into every run.
# Use a FRESH node-local graphroot dedicated to GLM serving; costs one image
# pull per node, buys deterministic state. Production storage left untouched.
_PODGRAPH="/var/tmp/podman_${_WHOAMI}/glm-storage"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/job-${SLURM_JOB_ID}/xdg}"
mkdir -p "${XDG_RUNTIME_DIR}/containers/storage" "${_PODGRAPH}"
# NOTE: /tmp/job-<id> itself is root-owned (prolog); only xdg/ is user-writable.
export CONTAINERS_STORAGE_CONF="${XDG_RUNTIME_DIR}/storage-glm.conf"
cat > "${CONTAINERS_STORAGE_CONF}" <<EOF
[storage]
driver="overlay"
runroot = "${XDG_RUNTIME_DIR}/containers/storage"
rootless_storage_path="${_PODGRAPH}"
graphroot = "${_PODGRAPH}"
[storage.options]
mount_program = "/usr/bin/fuse-overlayfs"
[storage.options.overlay]
mountopt="xattr_permissions=2,threaded=1"
EOF
echo "[glm] podman: runroot=${XDG_RUNTIME_DIR}/containers/storage graphroot=${_PODGRAPH}"

# Purge stale container state (keep image layers). Same steps as production.
pkill -9 -u "${_WHOAMI}" conmon 2>/dev/null || true
pkill -9 -u "${_WHOAMI}" crun   2>/dev/null || true
sleep 2
find /tmp/ -maxdepth 1 -name "job-*" -not -name "job-${SLURM_JOB_ID}" -user "${_WHOAMI}" 2>/dev/null \
  | while read -r _jd; do rm -rf "${_jd}/xdg/containers" 2>/dev/null; done || true
if [[ -f "${_PODGRAPH}/overlay-containers/containers.json" ]]; then
    echo "[glm] clearing stale container entries from graphroot"
    printf '[]' > "${_PODGRAPH}/overlay-containers/containers.json"
    find "${_PODGRAPH}/overlay-containers/" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
fi
rm -f "${_PODGRAPH}/db.sql" 2>/dev/null || true
rm -rf "${_PODGRAPH}/libpod" 2>/dev/null || true
mkdir -p "${_PODGRAPH}/libpod"
rm -rf "/run/user/$(id -u)/libpod" 2>/dev/null || true

podman system migrate >/dev/null 2>&1 || true
podman rm -f "${CTR}" >/dev/null 2>&1 || true

if [[ "${ROLE}" == "head" ]]; then
    RAY_ADDRESS="${MY_IP}:${RAY_PORT}"
    echo "${MY_IP}" > "${RDV}/head-${RUN_ID}.ip"
    echo "[glm] rendezvous published: ${RDV}/head-${RUN_ID}.ip"
    HEADLOG="${MODELS}/logs/glm52-nvfp4-head-${SLURM_JOB_ID}.log"

    # Head container: Ray head + GPU gate + co-located vLLM driver.
    # Logging: conmon writes DIRECTLY to file (k8s-file driver). Do NOT use a
    # background `podman logs -f` — the long-lived client can wedge the
    # rootless BoltDB state, making `podman ps` return empty for a healthy
    # container (false liveness misses killed runs 1783891342/1783891440).
    podman run -d --name "${CTR}" --replace --network host --shm-size=4g \
        --log-driver=k8s-file --log-opt "path=${HEADLOG}" \
        --entrypoint /bin/sh \
        "${PODMAN_ARGS[@]}" \
        "${VLLM_IMAGE}" \
        -c "${RAY_PIP}; ray start --head --node-ip-address='${MY_IP}' --port='${RAY_PORT}' --num-gpus=1 --object-store-memory=1073741824 --include-dashboard=false; echo '[head] waiting for ${TOTAL_GPUS} GPUs...'; i=0; while [ \$i -lt 240 ]; do g=\$(ray status 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | head -1 | sed 's@.*/@@; s@ GPU@@; s@[.].*@@'); [ \"\${g:-0}\" = '${TOTAL_GPUS}' ] && break; sleep 5; i=\$((i+1)); done; echo \"[head] GPUs=\${g:-0}; launching vLLM\"; exec vllm serve /model --served-model-name '${SERVED_NAME}' -tp 1 -pp ${TOTAL_GPUS} --distributed-executor-backend ray --host 0.0.0.0 --port 8000 --max-model-len ${MAX_LEN} --gpu-memory-utilization ${GPU_UTIL} --kv-cache-dtype fp8_e4m3 --enforce-eager --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice" \
        || { echo "[glm] FATAL: head container launch failed" >&2; exit 1; }
    # Startup verification: catch an instantly-dying container with real
    # diagnostics (run 1783891342 died silently with an empty log).
    sleep 8
    echo "[glm] post-launch state: $(podman inspect "${CTR}" --format '{{.State.Status}} exit={{.State.ExitCode}}' 2>&1)"
    podman ps -a --filter "name=${CTR}" --format '[glm] ps: {{.Names}} {{.Status}}' || true
    if [[ "$(podman inspect "${CTR}" --format '{{.State.Status}}' 2>/dev/null)" != "running" ]]; then
        echo "[glm] FATAL: head container not running after launch; logs:" >&2
        tail -40 "${HEADLOG}" 2>/dev/null | sed 's/^/[glm][head] /'
        KEEP_CTR=1
        exit 1
    fi
    echo "[glm] head log → ${HEADLOG} (k8s-file driver)"

    # Readiness watch (also keeps batch step alive). `podman ps` can FAIL
    # TRANSIENTLY (BoltDB lock race with the background `podman logs -f`),
    # returning empty while the container is healthy — this false-positive
    # killed run 1783890826. Require 3 consecutive misses (30s) before FATAL.
    _up=0; _miss=0
    for _ in $(seq 1 540); do
        if curl -sf -m 5 "http://${MY_IP}:8000/v1/models" >/dev/null 2>&1; then _up=1; break; fi
        if podman ps --filter "name=${CTR}" --format '{{.Names}}' 2>/dev/null | grep -q "${CTR}"; then
            _miss=0
        else
            _miss=$((_miss+1))
            echo "[glm] liveness miss ${_miss}/3 for ${CTR}"
            if [[ "${_miss}" -ge 3 ]]; then
                echo "[glm] FATAL: head container died" >&2
                echo "[glm] postmortem inspect: $(podman inspect "${CTR}" --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} err={{.State.Error}}' 2>&1)"
                tail -60 "${HEADLOG}" 2>/dev/null | sed 's/^/[glm][head] /'
                KEEP_CTR=1
                exit 1
            fi
        fi
        sleep 10
    done
    [[ "${_up}" -eq 1 ]] || { echo "[glm] FATAL: vLLM never ready" >&2; podman inspect "${CTR}" --format 'state={{.State.Status}}' 2>&1; tail -60 "${HEADLOG}" 2>/dev/null | sed 's/^/[glm][head] /'; KEEP_CTR=1; exit 1; }
    echo "[glm] ✅ vLLM READY at http://${MY_IP}:8000  (model '${SERVED_NAME}')"

    # Smoke test
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
    _miss=0
    while true; do
        if podman ps --filter "name=${CTR}" --format '{{.Names}}' 2>/dev/null | grep -q "${CTR}"; then
            _miss=0
        else
            _miss=$((_miss+1))
            [[ "${_miss}" -ge 3 ]] && break
        fi
        sleep 60
    done
    echo "[glm] head container exited; job ends"
else
    # Worker: wait for rendezvous, then run Ray worker FOREGROUND in batch step.
    echo "[glm] waiting for head rendezvous (up to 15 min)..."
    HEAD_IP=""
    for _ in $(seq 1 180); do
        [[ -f "${RDV}/head-${RUN_ID}.ip" ]] && { HEAD_IP="$(cat "${RDV}/head-${RUN_ID}.ip")"; break; }
        sleep 5
    done
    [[ -n "${HEAD_IP}" ]] || { echo "[glm] FATAL: no head rendezvous for run ${RUN_ID}" >&2; exit 1; }
    RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    echo "[glm] joining Ray cluster at ${RAY_ADDRESS}"

    # FOREGROUND podman in the batch step — survives for the job's life.
    # If ray exits (e.g. head gone), retry a few times before giving up.
    for _att in 1 2 3 4 5; do
        podman run --name "${CTR}" --replace --network host --shm-size=4g \
            --entrypoint /bin/sh \
            "${PODMAN_ARGS[@]}" \
            "${VLLM_IMAGE}" \
            -c "${RAY_PIP}; python3 -c 'import pynvml; pynvml.nvmlInit(); assert pynvml.nvmlDeviceGetCount() > 0' && exec ray start --address='${RAY_ADDRESS}' --num-gpus=1 --object-store-memory=1073741824 --block"
        echo "[glm] $(date +%H:%M:%S) worker container exited rc=$? (attempt ${_att}/5); retry in 15s"
        sleep 15
        [[ -f "${RDV}/head-${RUN_ID}.ip" ]] || { echo "[glm] head rendezvous gone; exiting"; break; }
    done
fi
