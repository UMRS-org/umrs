# shellcheck shell=bash disable=SC2034  # SG_* globals are read by the serving wrapper
# sglang_runner.sh — the multi-node SGLang engine adapter for the serving
# wrapper (source me; never execute me directly).
#
# Why (2026-11-12): nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 is served by
# SGLang (NVIDIA's validated recipe: 4x GB300, TP=4, DSpark speculative
# decoding, Engram tables in host RAM). On rccs it runs on two nodes of the
# `ai-pro6000` partition (aarch64, 2x RTX PRO 6000 Blackwell 96 GB, sm_120,
# rootless podman, no docker, no ssh between nodes): TP=4 across the pair.
# The multi-node vLLM path of serve-model.slurm runs Ray; SGLang needs none.
# Every node runs one `python3 -m sglang.launch_server` with the same flags
# and its own --node-rank; rank 0 serves the OpenAI API.
#
# Topology (one Slurm job, SG_NNODES nodes, SG_GPUS_PER_NODE GPUs each):
#   rank 0     detached container `rivault-sg-r0` on the batch node (the head),
#              serving HTTP on ${SG_PORT}; hosts the torch.distributed store at
#              ${HEAD_IP}:${SG_DIST_PORT}.
#   rank i>0   container `rivault-sg-r<i>` on node i, launched like the
#              multi-node vLLM workers (serve-model.slurm, 2026-11-03):
#              step      (runc sites, rccs) one long-lived backgrounded
#                        `srun --overlap` step per worker prepares the venue,
#                        resets the pause process and runs the container in
#                        the foreground, inside a relaunch loop (a detached
#                        container dies with its short step there: no user
#                        systemd adopts it);
#              detached  (crun sites, bdrt) `podman run -d` from a short step
#                        (the node's user systemd adopts it, and it outlives
#                        the cluster's ~60 s step killer). A dead detached
#                        worker breaks the NCCL world, rank 0 then exits, and
#                        the wrapper's monitor relaunches every rank.
#              RIVAULT_MN_WORKER_MODE overrides; default step iff
#              RIVAULT_PARTITION_OCI_RUNTIME=runc.
# LiteLLM (compose) is repointed at http://${HEAD_IP}:${SG_PORT}/v1.
#
# Contract (bash 4+):
#   source sglang_runner.sh
#   start_sglang_mn            # preflight + launch; sets HEAD_IP, RAY_NODES
#   sg_launch_ranks            # relaunch every rank (the monitor's respawn when
#                              # rivault-sg-r0 exits)
#   sglang_stop                # remove every rank, the worker loops, the key
#   sg_repoint_litellm <cfg>   # http://vllm:8000/v1 -> rank 0
#
# Inputs (roster env_vars):
#   SG_IMAGE          engine image; localhost/* must be staged on every node
#                     (or in RIVAULT_IMAGE_ARCHIVE_DIR), anything else must
#                     carry an @sha256 digest
#   SG_MODEL_DIR      host path of the checkpoint (mounted read-only at /model)
#   SG_RUNTIME        podman (default) | singularity. ai-pro6000 nodes have no
#                     podman (job 468124, 2026-11-12) but singularity-ce: with
#                     singularity SG_IMAGE is a .sif path (a sha256 sidecar
#                     <sif>.sha256 pins it), every rank runs `singularity exec
#                     --nv` in the foreground of a backgrounded process (rank 0
#                     on the head, the others in long-lived srun steps), and
#                     sg_rank0_alive replaces the container check. The UMRS
#                     wrapper's compose stack still needs podman on the head,
#                     so this mode serves the direct driver only
#   SG_STEP_GRES      empty (default) or a --gres value (ai-pro6000: gpu:2) that
#                     every singularity rank's srun step requests. On
#                     ai-pro6000 the batch shell's device cgroup holds ONE of
#                     the node's two GPUs (job 468620, 2026-11-12: TP rank 1
#                     on node 0 failed "invalid device ordinal"), so rank 0
#                     runs in its own step too. GB10 partitions reject --gres
#   SG_MODEL_ROOT     empty (default), or a host directory holding SG_MODEL_DIR
#                     (a Hugging Face repo dir: models--<org>--<name>). It is
#                     mounted at its own path and --model-path is SG_MODEL_DIR
#                     itself, so a snapshot's relative links into ../../blobs
#                     resolve inside the container
#   SG_SERVED_NAME    served model name (default VLLM_SERVED_MODEL_NAME)
#   SG_PORT 30000     rank 0's HTTP port
#   SG_DIST_PORT 29640  torch.distributed store port on rank 0 (--dist-init-addr)
#   SG_NNODES         nodes used (default: the job's node count)
#   SG_GPUS_PER_NODE 2
#   SG_PP 1           pipeline stages (--pp-size). SGLang 314bce7c refuses
#                     pp_size > 1 for DeepSeek-V4.1 (deepseek_v4_hook.py), so
#                     TP=2 x PP=2 only works for other models or a later image.
#   SG_TP             default SG_NNODES*SG_GPUS_PER_NODE/SG_PP (--tp-size);
#                     SG_TP*SG_PP must equal SG_NNODES*SG_GPUS_PER_NODE
#   SG_CONTEXT 262144 --context-length ("" = the checkpoint's)
#   SG_MEM_FRACTION 0.9  --mem-fraction-static ("" = SGLang's choice)
#   SG_KV_DTYPE fp8_e4m3 --kv-cache-dtype
#   SG_ATTN_BACKEND ""   --dsv4-attn-backend (auto|flashmla|flashmla_mega|trtllm;
#                     empty = SGLang picks: flashmla on sm_120)
#   SG_ENGRAM_CONFIG  '{"placement":"cpu","replication":"none","lookup":"auto"}'
#                     (--engram-config; set to "" to omit; DeepSeek-V4.1 only)
#   SG_SPEC DSPARK    --speculative-algorithm (set to "" to disable)
#   SG_SPEC_BLOCK 5   --speculative-dspark-block-size (DSPARK only)
#   SG_REASONING_PARSER auto   SG_TOOL_PARSER auto   ("" omits either)
#   SG_MAX_RUNNING ""  --max-running-requests
#   SG_EXTRA_ARGS ""   extra launch_server flags (word-split), every rank
#   SG_FABRIC_IFACE auto  NCCL/Gloo socket interface and the source of
#                     --dist-init-addr; "auto" = the first up link with an
#                     IPv4 and an RDMA device, else the default-route link.
#                     On a fabric whose TCP is firewalled (rccs ng-dgx QSFP,
#                     2026-11-06) name the management NIC here: the probe then
#                     falls back to the first ACTIVE RoCE v2 HCA for RDMA.
#   SG_RDMA auto      auto = RDMA when every node has a RoCE v2 HCA/GID and an
#                     unlimited memlock; 1 = require it; 0 = sockets only
#   SG_IB_HCA / SG_GID_INDEX   override the sysfs discovery (every node)
#   SG_GPU_FORM auto  cdi (--device nvidia.com/gpu=all) | devices (/dev/nvidia*
#                     plus read-only driver libraries); auto = cdi when the
#                     node has an nvidia CDI spec
#   SG_HOST_LIB_DIR /usr/lib/aarch64-linux-gnu   driver libraries (devices form)
#   SG_SHM_SIZE ""    "" = --ipc host (host /dev/shm); a size = private IPC
#                     with --shm-size (podman refuses --shm-size with --ipc host)
#   SG_WORKER_ATTEMPTS 5  SG_WORKER_RELAUNCH_SLEEP 5   step-mode relaunch loop
#   RIVAULT_IMAGE_ARCHIVE_DIR  site image archives (missing images load from here)
#   RIVAULT_USERNS_MODE        --userns (default keep-id:uid=0,gid=0)
#   VLLM_API_KEY      the wrapper's engine key. SGLang has no key-file flag or
#                     key env var; its --config YAML is merged into argv inside
#                     the process, so the key lives in a 0600 YAML file mounted
#                     read-only on rank 0 and never appears in a command line,
#                     `ps`, or `podman inspect`.

SG_R0_NAME="rivault-sg-r0"
SG_NAME_PREFIX="rivault-sg-r"
SG_VENUE_SH="${SG_VENUE_SH:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/venue.sh}"
SG_KEY_IN_CONTAINER="/run/secrets/sglang-key.yaml"
SG_MODEL_IN_CONTAINER="/model"
SG_DEFAULT_ENGRAM='{"placement":"cpu","replication":"none","lookup":"auto"}'

sg_job_node_count() { echo "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"; }

# `:=` fills unset or empty; `=` fills only unset, so "" switches a flag off.
sg_defaults() {
    : "${SG_SERVED_NAME:=${VLLM_SERVED_MODEL_NAME:-DeepSeek-V4.1-Flash-1017}}"
    : "${SG_PORT:=30000}" "${SG_DIST_PORT:=29640}" "${SG_NNODES:=$(sg_job_node_count)}"
    : "${SG_GPUS_PER_NODE:=2}" "${SG_PP:=1}"
    : "${SG_TP:=$(( SG_NNODES * SG_GPUS_PER_NODE / SG_PP ))}"
    : "${SG_CONTEXT=262144}" "${SG_MEM_FRACTION=0.9}" "${SG_KV_DTYPE=fp8_e4m3}"
    : "${SG_ATTN_BACKEND:=}" "${SG_ENGRAM_CONFIG=${SG_DEFAULT_ENGRAM}}"
    : "${SG_SPEC=DSPARK}" "${SG_SPEC_BLOCK:=5}" "${SG_MAX_RUNNING:=}"
    : "${SG_REASONING_PARSER=auto}" "${SG_TOOL_PARSER=auto}" "${SG_EXTRA_ARGS:=}"
    : "${SG_FABRIC_IFACE:=auto}" "${SG_RDMA:=auto}" "${SG_GPU_FORM:=auto}"
    : "${SG_HOST_LIB_DIR:=/usr/lib/aarch64-linux-gnu}" "${SG_SHM_SIZE:=}"
    : "${SG_WORKER_ATTEMPTS:=5}" "${SG_WORKER_RELAUNCH_SLEEP:=5}" "${SG_MODEL_ROOT:=}"
    sg_model_mount
}

# SG_MODEL_MOUNT (the -v value) and SG_MODEL_IN_CONTAINER (--model-path).
sg_model_mount() {
    SG_MODEL_IN_CONTAINER="/model"
    SG_MODEL_MOUNT="${SG_MODEL_DIR:-}:/model:ro"
    if [[ -n "${SG_MODEL_ROOT}" ]]; then
        SG_MODEL_IN_CONTAINER="${SG_MODEL_DIR:-}"
        SG_MODEL_MOUNT="${SG_MODEL_ROOT}:${SG_MODEL_ROOT}:ro"
    fi
}

# The settings that must be present; prints the first missing one's name.
sg_missing_setting() {
    local name
    for name in SG_IMAGE SG_MODEL_DIR VLLM_API_KEY; do
        if [[ -z "${!name:-}" ]]; then
            echo "${name}"
            return 0
        fi
    done
}

# A localhost/* image is staged per node; any other reference must be pinned
# by digest (nightly tags move; the tensorfold_runner.sh rule).
sg_runtime() { echo "${SG_RUNTIME:-podman}"; }

sg_image_is_pinned() {
    if [[ "$(sg_runtime)" == singularity ]]; then
        [[ "$1" == /*.sif && -s "$1.sha256" ]]
        return
    fi
    case "$1" in
        localhost/*) return 0 ;;
        *@sha256:*) return 0 ;;
    esac
    return 1
}

# rc 1 (one FATAL line) unless the parallel layout fits the allocation.
sg_check_layout() {
    local job_nodes world
    job_nodes="$(sg_job_node_count)"
    world=$(( SG_NNODES * SG_GPUS_PER_NODE ))
    if (( SG_NNODES < 1 || SG_NNODES > job_nodes )); then
        echo "[rivault] FATAL: SG_NNODES=${SG_NNODES} but the job has ${job_nodes} node(s)" >&2
        return 1
    fi
    if (( SG_TP * SG_PP != world )); then
        echo "[rivault] FATAL: SG_TP=${SG_TP} x SG_PP=${SG_PP} != ${SG_NNODES} nodes x ${SG_GPUS_PER_NODE} GPUs" >&2
        return 1
    fi
    if (( SG_PP > 1 )); then
        echo "[rivault] WARNING: SG_PP=${SG_PP}: SGLang 314bce7c refuses pipeline parallelism for DeepSeek-V4.1" >&2
    fi
    return 0
}

# The node-side probe, run through srun on each node. Prints one line:
#   <ipv4> <hca> <gid index> <gpus> <cdi 0|1> <memlock> <iface> <caps> <libs>
# (caps: comma list of /dev/nvidia-caps/*, libs: comma list of the driver
# libraries present in $2; "-" when empty). The RoCE v2 GID matching follows
# tensorfold_runner.sh TF_PROBE_SCRIPT; when the chosen link has no RDMA
# device, the first ACTIVE HCA with an IPv4-mapped RoCE v2 GID is used.
# shellcheck disable=SC2016  # expands on the node
SG_PROBE_SCRIPT='
dev="$1"; libdir="$2"
v2gid() {
  for g in /sys/class/infiniband/"$1"/ports/1/gids/*; do
    i=${g##*/}; v=$(cat "$g" 2>/dev/null)
    case "$2" in
      any) case "$v" in 0000:0000:0000:0000:0000:ffff:*) ;; *) continue ;; esac ;;
      *) [ "$v" = "$2" ] || continue ;;
    esac
    t=$(cat /sys/class/infiniband/"$1"/ports/1/gid_attrs/types/"$i" 2>/dev/null)
    case "$t" in *"v2"*) echo "$i"; return 0 ;; esac
  done
  return 1
}
if [ "$dev" = auto ]; then
  dev=""
  for d in /sys/class/net/*; do
    n=${d##*/}
    [ -e "$d/device/infiniband" ] || continue
    [ "$(cat "$d/operstate" 2>/dev/null)" = up ] || continue
    ip -4 -o addr show dev "$n" 2>/dev/null | grep -q inet || continue
    dev=$n; break
  done
  [ -n "$dev" ] || dev=$(ip -4 route show default 2>/dev/null | awk "{for (i = 1; i < NF; i++) if (\$i == \"dev\") {print \$(i+1); exit}}")
fi
ip=$(ip -4 -o addr show dev "$dev" 2>/dev/null | awk "{sub(/\\/.*/, \"\", \$4); print \$4; exit}")
hca=$(ls "/sys/class/net/$dev/device/infiniband" 2>/dev/null | head -1)
gid="-"
if [ -n "$ip" ] && [ -n "$hca" ]; then
  want=$(printf "0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x" $(echo "$ip" | tr . " "))
  gid=$(v2gid "$hca" "$want") || gid="-"
fi
isib() { [ "$(cat /sys/class/infiniband/"$1"/ports/1/link_layer 2>/dev/null)" = InfiniBand ]; }
if [ -n "$hca" ] && [ "$gid" = "-" ] && isib "$hca"; then gid=ib; fi
if [ -z "$hca" ]; then
  for h in /sys/class/infiniband/*; do
    h=${h##*/}
    grep -q ACTIVE /sys/class/infiniband/"$h"/ports/1/state 2>/dev/null || continue
    if isib "$h"; then hca=$h; gid=ib; break; fi
    g=$(v2gid "$h" any) || continue
    hca=$h; gid=$g; break
  done
fi
gpus=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU")
cdi=0
ls /etc/cdi/*nvidia* /var/run/cdi/*nvidia* >/dev/null 2>&1 && cdi=1
caps=""
for c in /dev/nvidia-caps/*; do [ -e "$c" ] && caps="$caps$c,"; done
libs=""
for f in "$libdir"/libcuda.so.1 "$libdir"/libnvidia-ml.so.1 "$libdir"/libnvidia-ptxjitcompiler.so.1 \
         "$libdir"/libnvidia-nvvm.so.4 "$libdir"/libnvidia-gpucomp.so.*; do
  [ -e "$f" ] && libs="$libs${f##*/},"
done
echo "${ip:--} ${hca:--} ${gid:--} ${gpus:-0} ${cdi} $(ulimit -Hl) ${dev:--} ${caps:--} ${libs:--}"
'

sg_probe_node() {
    srun --nodes=1 --ntasks=1 --nodelist="$1" --overlap \
        bash -c "${SG_PROBE_SCRIPT}" _ "${SG_FABRIC_IFACE}" "${SG_HOST_LIB_DIR}"
}

# Parse one probe line into SG_NODE_*[rank].
sg_record_probe() {
    local rank="$1" line="$2" ip hca gid gpus cdi memlock dev caps libs
    read -r ip hca gid gpus cdi memlock dev caps libs <<< "${line}"
    SG_NODE_IP[rank]="${ip:--}"
    SG_NODE_IFACE[rank]="${dev:-${SG_FABRIC_IFACE}}"
    SG_NODE_HCA[rank]="${SG_IB_HCA:-${hca:--}}"
    SG_NODE_GID[rank]="${SG_GID_INDEX:-${gid:--}}"
    SG_NODE_GPUS[rank]="${gpus:-0}"
    SG_NODE_CDI[rank]="${cdi:-0}"
    SG_NODE_MEMLOCK[rank]="${memlock:-}"
    SG_NODE_CAPS[rank]="${caps:--}"
    SG_NODE_LIBS[rank]="${libs:--}"
}

# rc 1 (and one FATAL line) when a probed node cannot run its rank.
sg_check_probe() {
    local rank="$1" node="$2"
    if [[ "${SG_NODE_IP[rank]}" == "-" ]]; then
        echo "[rivault] FATAL: no IPv4 on ${SG_NODE_IFACE[rank]} (SG_FABRIC_IFACE=${SG_FABRIC_IFACE}) on ${node}" >&2
        return 1
    fi
    if (( SG_NODE_GPUS[rank] < SG_GPUS_PER_NODE )); then
        echo "[rivault] FATAL: ${node} has ${SG_NODE_GPUS[rank]} GPU(s), SG_GPUS_PER_NODE=${SG_GPUS_PER_NODE} (allocate with --gres=gpu:${SG_GPUS_PER_NODE})" >&2
        return 1
    fi
    sg_resolve_gpu_form "${rank}" "${node}"
}

# SG_NODE_FORM[rank] = cdi|devices; rc 1 when the devices form lacks libcuda.
sg_resolve_gpu_form() {
    local rank="$1" node="$2" form="${SG_GPU_FORM}"
    if [[ "${form}" == auto ]]; then
        form=devices
        [[ "${SG_NODE_CDI[rank]}" == 1 ]] && form=cdi
    fi
    SG_NODE_FORM[rank]="${form}"
    if [[ "${form}" == devices && ",${SG_NODE_LIBS[rank]}," != *",libcuda.so.1,"* ]]; then
        echo "[rivault] FATAL: no libcuda.so.1 under ${SG_HOST_LIB_DIR} on ${node} (SG_GPU_FORM=devices; set SG_HOST_LIB_DIR or use cdi)" >&2
        return 1
    fi
    return 0
}

# Prints why node rank $1 cannot do RDMA; empty when it can.
sg_rdma_blocker() {
    local rank="$1"
    if [[ "${SG_NODE_HCA[rank]}" == "-" || "${SG_NODE_GID[rank]}" == "-" ]]; then
        echo "no InfiniBand HCA or RoCE v2 HCA/GID (set SG_IB_HCA/SG_GID_INDEX)"
    elif [[ "${SG_NODE_MEMLOCK[rank]}" != "unlimited" ]]; then
        echo "memlock hard limit is ${SG_NODE_MEMLOCK[rank]}, RDMA needs unlimited"
    fi
}

# SG_RDMA_ON = 0|1 from SG_RDMA and the probes; rc 1 when SG_RDMA=1 cannot hold.
sg_resolve_rdma() {
    local rank why
    SG_RDMA_ON=0
    [[ "${SG_RDMA}" == 0 ]] && return 0
    for (( rank = 0; rank < SG_NNODES; rank++ )); do
        why="$(sg_rdma_blocker "${rank}")"
        [[ -z "${why}" ]] && continue
        if [[ "${SG_RDMA}" == 1 ]]; then
            echo "[rivault] FATAL: ${why} on ${RAY_NODES[rank]} (SG_RDMA=1; or SG_RDMA=0)" >&2
            return 1
        fi
        echo "[rivault] SGLang: RDMA off (${why} on ${RAY_NODES[rank]}); NCCL uses sockets"
        return 0
    done
    (( SG_NNODES > 1 )) && SG_RDMA_ON=1
    return 0
}

# Device and driver-library arguments for rank $1 into SG_GPU_ARGS.
sg_gpu_args() {
    local rank="$1" i dev lib
    if [[ "${SG_NODE_FORM[rank]}" == cdi ]]; then
        SG_GPU_ARGS=( --device nvidia.com/gpu=all )
        return 0
    fi
    SG_GPU_ARGS=()
    for (( i = 0; i < SG_GPUS_PER_NODE; i++ )); do SG_GPU_ARGS+=( --device "/dev/nvidia${i}" ); done
    SG_GPU_ARGS+=( --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools )
    for dev in ${SG_NODE_CAPS[rank]//,/ }; do
        [[ "${dev}" == "-" ]] || SG_GPU_ARGS+=( --device "${dev}" )
    done
    for lib in ${SG_NODE_LIBS[rank]//,/ }; do
        [[ "${lib}" == "-" ]] || SG_GPU_ARGS+=( -v "${SG_HOST_LIB_DIR}/${lib}:${SG_HOST_LIB_DIR}/${lib}:ro" )
    done
    SG_GPU_ARGS+=( -e NVIDIA_VISIBLE_DEVICES=all -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility" )
}

# Container arguments every rank shares, plus rank $1's GPU form, into SG_RUN_ARGS.
sg_common_run_args() {
    local rank="$1"
    SG_RUN_ARGS=( --network host
        --userns "${RIVAULT_USERNS_MODE:-keep-id:uid=0,gid=0}"
        --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864
        --cap-add IPC_LOCK --cap-add SYS_NICE
        -v "${SG_MODEL_MOUNT}"
        --entrypoint python3 )
    if [[ -n "${SG_SHM_SIZE}" ]]; then
        SG_RUN_ARGS+=( --shm-size "${SG_SHM_SIZE}" )
    else
        SG_RUN_ARGS+=( --ipc host )
    fi
    if [[ "${SG_RDMA_ON}" == 1 ]]; then SG_RUN_ARGS+=( --device /dev/infiniband ); fi
    sg_gpu_args "${rank}"
    SG_RUN_ARGS+=( "${SG_GPU_ARGS[@]}" )
}

# Rank $1's environment (NCCL/Gloo on the fabric link) into SG_ENV.
sg_env() {
    local rank="$1" iface="${SG_NODE_IFACE[$1]}"
    SG_ENV=( HF_HUB_OFFLINE=1 "NCCL_SOCKET_IFNAME=${iface}" "GLOO_SOCKET_IFNAME=${iface}"
        "SGLANG_HOST_IP=${SG_NODE_IP[rank]}" )
    if [[ "${SG_RDMA_ON}" == 1 ]]; then
        SG_ENV+=( "NCCL_IB_HCA=${SG_NODE_HCA[rank]}" )
        # InfiniBand (ai-pro6000: mlx5_0, NDR 400) needs no GID index; RoCE v2 does
        if [[ "${SG_NODE_GID[rank]}" != ib ]]; then SG_ENV+=( "NCCL_IB_GID_INDEX=${SG_NODE_GID[rank]}" ); fi
    else
        SG_ENV+=( NCCL_IB_DISABLE=1 )
    fi
}

# Append "FLAG VALUE" to SG_CMD when VALUE is non-empty.
sg_add_opt() {
    [[ -n "$2" ]] && SG_CMD+=( "$1" "$2" )
    return 0
}

# Model and serving flags shared by every rank (appended to SG_CMD).
sg_model_flags() {
    sg_add_opt --context-length "${SG_CONTEXT}"
    sg_add_opt --mem-fraction-static "${SG_MEM_FRACTION}"
    sg_add_opt --kv-cache-dtype "${SG_KV_DTYPE}"
    sg_add_opt --dsv4-attn-backend "${SG_ATTN_BACKEND}"
    sg_add_opt --engram-config "${SG_ENGRAM_CONFIG}"
    sg_add_opt --speculative-algorithm "${SG_SPEC}"
    if [[ "${SG_SPEC}" == DSPARK ]]; then sg_add_opt --speculative-dspark-block-size "${SG_SPEC_BLOCK}"; fi
    sg_add_opt --max-running-requests "${SG_MAX_RUNNING}"
    sg_add_opt --reasoning-parser "${SG_REASONING_PARSER}"
    sg_add_opt --tool-call-parser "${SG_TOOL_PARSER}"
}

# Rank $1's launch_server argv (after the python3 entrypoint) into SG_CMD.
sg_server_cmd() {
    local rank="$1"
    SG_CMD=( -m sglang.launch_server --model-path "${SG_MODEL_IN_CONTAINER}"
        --served-model-name "${SG_SERVED_NAME}" --trust-remote-code
        --host 0.0.0.0 --port "${SG_PORT}" --tp-size "${SG_TP}" )
    if (( SG_PP > 1 )); then SG_CMD+=( --pp-size "${SG_PP}" ); fi
    if (( SG_NNODES > 1 )); then
        SG_CMD+=( --nnodes "${SG_NNODES}" --node-rank "${rank}"
            --dist-init-addr "${SG_NODE_IP[0]}:${SG_DIST_PORT}" )
    fi
    sg_model_flags
    # The key reaches rank 0 only, and only as a file (see VLLM_API_KEY).
    # launch_server merges --config only when spelled as two words.
    if (( rank == 0 )); then SG_CMD+=( --config "${SG_KEY_IN_CONTAINER}" ); fi
    if [[ -n "${SG_EXTRA_ARGS}" ]]; then
        local -a extra
        read -r -a extra <<< "${SG_EXTRA_ARGS}"
        SG_CMD+=( "${extra[@]}" )
    fi
}

# Full `podman run` argv for rank $1 into SG_ARGV. $2 = detached|step
# (step runs in the foreground). Workers replace a leftover container.
sg_rank_argv() {
    local rank="$1" mode="${2:-detached}" x
    sg_common_run_args "${rank}"; sg_env "${rank}"; sg_server_cmd "${rank}"
    SG_ARGV=( podman run )
    if [[ "${mode}" != step ]]; then SG_ARGV+=( -d ); fi
    SG_ARGV+=( --name "${SG_NAME_PREFIX}${rank}" )
    if (( rank > 0 )); then SG_ARGV+=( --replace ); fi
    SG_ARGV+=( "${SG_RUN_ARGS[@]}" )
    if (( rank == 0 )); then SG_ARGV+=( -v "${SG_KEY_FILE}:${SG_KEY_IN_CONTAINER}:ro" ); fi
    for x in "${SG_ENV[@]}"; do SG_ARGV+=( -e "${x}" ); done
    SG_ARGV+=( "${SG_IMAGE}" "${SG_CMD[@]}" )
}

# `singularity exec` argv for rank $1 into SG_ARGV (always foreground).
sg_rank_argv_sif() {
    local rank="$1" x
    sg_env "${rank}"; sg_server_cmd "${rank}"
    SG_ARGV=( singularity exec --nv --bind "${SG_MODEL_MOUNT}" )
    if (( rank == 0 )); then SG_ARGV+=( --bind "${SG_KEY_FILE}:${SG_KEY_IN_CONTAINER}:ro" ); fi
    for x in "${SG_ENV[@]}"; do SG_ARGV+=( --env "${x}" ); done
    SG_ARGV+=( "${SG_IMAGE}" python3 "${SG_CMD[@]}" )
}

# A per-job log file for rank $1 beside the key file.
sg_log_file() { echo "${RIVAULT_WRAPPER_DIR:-.}/jobs/sg-r$1.${SLURM_JOB_ID:-0}.log"; }

# The srun options of a singularity rank's step on node $1.
sg_sif_step_opts() {
    SG_STEP_OPTS=( --nodes=1 --ntasks=1 --nodelist="$1" --overlap )
    if [[ -n "${SG_STEP_GRES:-}" ]]; then SG_STEP_OPTS+=( "--gres=${SG_STEP_GRES}" ); fi
}

# Singularity: workers as long-lived steps (relaunched like step mode), then
# rank 0 in a step of its own on the head (see SG_STEP_GRES).
sg_launch_ranks_sif() {
    local rank cmd
    sg_stop_ranks_sif
    for (( rank = 1; rank < SG_NNODES; rank++ )); do
        sg_rank_argv_sif "${rank}"
        cmd="exec $(printf '%q ' "${SG_ARGV[@]}")"
        sg_sif_worker_loop "${rank}" "${RAY_NODES[rank]}" "${cmd}" &
        SG_WORKER_PIDS+=( "$!" )
        WORKER_PIDS+=( "$!" )
    done
    sg_rank_argv_sif 0
    sg_sif_step_opts "${RAY_NODES[0]}"
    srun "${SG_STEP_OPTS[@]}" "${SG_ARGV[@]}" >> "$(sg_log_file 0)" 2>&1 &
    SG_R0_PID=$!
    echo "[rivault] SGLang ranks started (singularity): ${SG_NNODES} node(s), TP=${SG_TP}, RDMA ${SG_RDMA_ON}, r0 pid ${SG_R0_PID} (weights load in background)"
}

sg_sif_worker_loop() {
    local rank="$1" node="$2" cmd="$3" attempt
    sg_sif_step_opts "${node}"
    for (( attempt = 1; attempt <= SG_WORKER_ATTEMPTS; attempt++ )); do
        srun "${SG_STEP_OPTS[@]}" bash -c "${cmd}" >> "$(sg_log_file "${rank}")" 2>&1 || true
        echo "[rivault] SGLang rank ${rank} on ${node} exited (attempt ${attempt}/${SG_WORKER_ATTEMPTS})" >&2
        sleep "${SG_WORKER_RELAUNCH_SLEEP}"
    done
}

sg_stop_ranks_sif() {
    local pid
    for pid in "${SG_WORKER_PIDS[@]}"; do pkill -TERM -P "${pid}" 2>/dev/null; kill "${pid}" 2>/dev/null || true; done
    SG_WORKER_PIDS=()
    if [[ -n "${SG_R0_PID:-}" ]]; then kill "${SG_R0_PID}" 2>/dev/null || true; SG_R0_PID=""; fi
}

# rc 0 while rank 0 runs (either runtime).
sg_rank0_alive() {
    if [[ "$(sg_runtime)" == singularity ]]; then
        [[ -n "${SG_R0_PID:-}" ]] && kill -0 "${SG_R0_PID}" 2>/dev/null
        return
    fi
    [[ "$(podman inspect "${SG_R0_NAME}" --format '{{.State.Status}}' 2>/dev/null)" == running ]]
}

# The worker launch mode (serve-model.slurm's rule).
sg_worker_mode() {
    if [[ -n "${RIVAULT_MN_WORKER_MODE:-}" ]]; then
        echo "${RIVAULT_MN_WORKER_MODE}"
    elif [[ "${RIVAULT_PARTITION_OCI_RUNTIME:-}" == "runc" ]]; then
        echo step
    else
        echo detached
    fi
}

# A node-side shell line that recreates this job's private podman conf there.
sg_venue_prepare_cmd() {
    printf "source %q && rivault_venue_prepare %q %q" "${SG_VENUE_SH}" \
        "${SLURM_JOB_ID:-$$}" "${RIVAULT_CGROUP_MANAGER:-cgroupfs}"
}

# The last state and log lines of worker $1 on node $2 (before --replace
# destroys them).
sg_worker_postmortem() {
    local rank="$1" node="$2" name="${SG_NAME_PREFIX}$1"
    srun --nodes=1 --ntasks=1 --nodelist="${node}" --overlap bash -c \
        "podman inspect '${name}' --format 'exit={{.State.ExitCode}} oom={{.State.OOMKilled}} err={{.State.Error}}' 2>&1; podman logs --tail 40 '${name}' 2>&1 | tail -40" \
        2>&1 | sed "s/^/[rivault][sg-r${rank}-postmortem:${node}] /" || true
}

# Step mode: one long-lived step per attempt owns the pause process and the
# container; it returns when the worker exits. Runs in the background.
sg_worker_step_loop() {
    local rank="$1" node="$2" attempt cmd
    sg_rank_argv "${rank}" step
    cmd="$(sg_venue_prepare_cmd) >/dev/null 2>&1; podman system migrate >/dev/null 2>&1; exec $(printf '%q ' "${SG_ARGV[@]}")"
    for (( attempt = 1; attempt <= SG_WORKER_ATTEMPTS; attempt++ )); do
        srun --nodes=1 --ntasks=1 --nodelist="${node}" --overlap bash -c "${cmd}" || true
        sg_worker_postmortem "${rank}" "${node}"
        echo "[rivault] SGLang rank ${rank} on ${node} exited (attempt ${attempt}/${SG_WORKER_ATTEMPTS})" >&2
        sleep "${SG_WORKER_RELAUNCH_SLEEP}"
    done
    echo "[rivault] FATAL: SGLang rank ${rank} on ${node} failed to stay up after ${SG_WORKER_ATTEMPTS} attempts" >&2
}

# Detached mode: prepare the venue and reset the pause process in short
# steps, then start the container with -d.
sg_start_worker_detached() {
    local rank="$1" node="$2"
    srun --nodes=1 --ntasks=1 --nodelist="${node}" --overlap bash -c "$(sg_venue_prepare_cmd)" >/dev/null 2>&1 \
        || echo "[rivault] WARN: venue prepare failed on ${node}; podman will use inherited paths" >&2
    srun --nodes=1 --ntasks=1 --nodelist="${node}" --overlap podman system migrate >/dev/null 2>&1 || true
    sg_rank_argv "${rank}" detached
    if ! srun --nodes=1 --ntasks=1 --nodelist="${node}" --overlap "${SG_ARGV[@]}" >/dev/null; then
        echo "[rivault] FATAL: SGLang rank ${rank} did not start on ${node}" >&2
        return 1
    fi
}

sg_start_worker() {
    local rank="$1" node="${RAY_NODES[$1]}"
    if [[ "$(sg_worker_mode)" == step ]]; then
        sg_worker_step_loop "${rank}" "${node}" &
        SG_WORKER_PIDS+=( "$!" )
        WORKER_PIDS+=( "$!" )   # the wrapper's cleanup kills these too
        return 0
    fi
    sg_start_worker_detached "${rank}" "${node}"
}

# Stop the step loops first (no relaunch), then remove each worker container
# (which also ends a step-mode foreground step).
sg_stop_workers() {
    local rank pid
    for pid in "${SG_WORKER_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
    SG_WORKER_PIDS=()
    for (( rank = 1; rank < ${#RAY_NODES[@]} && rank < ${SG_NNODES:-0}; rank++ )); do
        srun --nodes=1 --ntasks=1 --nodelist="${RAY_NODES[rank]}" --overlap \
            podman rm -f -t 30 "${SG_NAME_PREFIX}${rank}" >/dev/null 2>&1 || true
    done
}

# Relaunch every rank: workers first, then rank 0 here (the head).
sg_launch_ranks() {
    local rank mode
    if [[ "$(sg_runtime)" == singularity ]]; then sg_launch_ranks_sif; return; fi
    mode="$(sg_worker_mode)"
    sg_stop_workers
    podman rm -f "${SG_R0_NAME}" >/dev/null 2>&1 || true
    for (( rank = 1; rank < SG_NNODES; rank++ )); do
        sg_start_worker "${rank}" || return 1
    done
    sg_rank_argv 0 detached
    if ! "${SG_ARGV[@]}" >/dev/null; then
        echo "[rivault] FATAL: SGLang rank 0 did not start on ${RAY_NODES[0]}" >&2
        return 1
    fi
    echo "[rivault] SGLang ranks started: ${SG_NNODES} node(s), TP=${SG_TP} PP=${SG_PP}, workers ${mode}, RDMA ${SG_RDMA_ON} (weights load in background)"
}

# Present on the node, or loaded there from the site's image archive
# (venue.sh rivault_ensure_image).
# shellcheck disable=SC2016  # expands on the node
sg_ensure_image_on() {
    if [[ "$(sg_runtime)" == singularity ]]; then
        srun --nodes=1 --ntasks=1 --nodelist="$1" --overlap test -s "${SG_IMAGE}"
        return
    fi
    srun --nodes=1 --ntasks=1 --nodelist="$1" --overlap bash -c \
        'source "$1" && rivault_ensure_image "$2" "$3"' _ \
        "${SG_VENUE_SH}" "${SG_IMAGE}" "${RIVAULT_IMAGE_ARCHIVE_DIR:-}"
}

# rc 1 unless the image is pinned and present on every node, and the model
# directory exists on every node.
sg_check_image_and_model() {
    local node
    if ! sg_image_is_pinned "${SG_IMAGE}"; then
        echo "[rivault] FATAL: SG_IMAGE ${SG_IMAGE} must be localhost/* (staged) or pinned @sha256" >&2
        return 1
    fi
    for node in "${RAY_NODES[@]}"; do
        if ! sg_ensure_image_on "${node}"; then
            echo "[rivault] FATAL: ${SG_IMAGE} is not staged on ${node} (nor in ${RIVAULT_IMAGE_ARCHIVE_DIR:-the archive dir})" >&2
            return 1
        fi
        if ! srun --nodes=1 --ntasks=1 --nodelist="${node}" --overlap test -d "${SG_MODEL_DIR}"; then
            echo "[rivault] FATAL: ${SG_MODEL_DIR} missing on ${node} (stage the checkpoint first)" >&2
            return 1
        fi
    done
}

# RAY_NODES = the first SG_NNODES nodes of the job; the batch step must be on
# the first.
sg_resolve_nodes() {
    mapfile -t RAY_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" | head -n "${SG_NNODES}")
    if [[ "${SLURMD_NODENAME:-$(hostname -s)}" != "${RAY_NODES[0]:-}" ]]; then
        echo "[rivault] FATAL: batch step not on the head node ${RAY_NODES[0]:-?}" >&2
        return 1
    fi
}

sg_probe_all() {
    local rank
    for (( rank = 0; rank < SG_NNODES; rank++ )); do
        sg_record_probe "${rank}" "$(sg_probe_node "${RAY_NODES[rank]}")"
        sg_check_probe "${rank}" "${RAY_NODES[rank]}" || return 1
    done
    sg_resolve_rdma
}

sg_preflight() {
    local missing
    missing="$(sg_missing_setting)"
    if [[ -n "${missing}" ]]; then
        echo "[rivault] FATAL: SGLang needs ${missing}" >&2
        return 1
    fi
    sg_check_layout || return 1
    sg_resolve_nodes || return 1
    sg_check_image_and_model || return 1
    sg_probe_all
}

# The engine key as a 0600 YAML file for launch_server --config (the only
# way SGLang reads a key from outside argv). JSON-style escaping is valid
# inside a YAML double-quoted scalar.
sg_write_key_file() {
    local key="${VLLM_API_KEY//\\/\\\\}"
    key="${key//\"/\\\"}"
    SG_KEY_FILE="${RIVAULT_WRAPPER_DIR:-.}/jobs/sg-api-key.${SLURM_JOB_ID:-0}.yaml"
    mkdir -p "$(dirname "${SG_KEY_FILE}")"
    ( umask 077; printf 'api-key: "%s"\n' "${key}" > "${SG_KEY_FILE}" )
}

# Point the job's LiteLLM at rank 0 (it was rendered for the compose vllm).
sg_repoint_litellm() {
    local cfg="$1"
    sed -i "s@http://vllm:8000/v1@http://${HEAD_IP}:${SG_PORT}/v1@g" "${cfg}"
    echo "[rivault] LiteLLM repointed to SGLang rank 0 at http://${HEAD_IP}:${SG_PORT}/v1"
}

sg_log_fabric() {
    local rank
    for (( rank = 0; rank < SG_NNODES; rank++ )); do
        echo "[rivault] SGLang node ${rank}: ${RAY_NODES[rank]}=${SG_NODE_IP[rank]} via ${SG_NODE_IFACE[rank]}, ${SG_NODE_GPUS[rank]} GPU(s) ${SG_NODE_FORM[rank]}, hca ${SG_NODE_HCA[rank]} gid ${SG_NODE_GID[rank]}"
    done
}

start_sglang_mn() {
    declare -g -a SG_NODE_IP=() SG_NODE_IFACE=() SG_NODE_HCA=() SG_NODE_GID=() SG_NODE_GPUS=()
    declare -g -a SG_NODE_CDI=() SG_NODE_MEMLOCK=() SG_NODE_CAPS=() SG_NODE_LIBS=() SG_NODE_FORM=()
    declare -g -a RAY_NODES=() SG_WORKER_PIDS=()
    sg_defaults
    sg_preflight || return 1
    HEAD_IP="${SG_NODE_IP[0]}"
    sg_log_fabric
    sg_write_key_file
    sg_launch_ranks
}

sglang_stop() {
    if [[ "$(sg_runtime)" == singularity ]]; then
        sg_stop_ranks_sif
        rm -f "${SG_KEY_FILE:-}" >/dev/null 2>&1 || true
        return 0
    fi
    sg_stop_workers
    podman stop -t 30 "${SG_R0_NAME}" >/dev/null 2>&1 || true
    podman rm -f "${SG_R0_NAME}" >/dev/null 2>&1 || true
    rm -f "${SG_KEY_FILE:-}" >/dev/null 2>&1 || true
}
