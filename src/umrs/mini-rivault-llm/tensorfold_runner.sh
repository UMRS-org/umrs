# shellcheck shell=bash disable=SC2034  # TF_* globals are read by serve-model.slurm
# tensorfold_runner.sh — the TensorFold engine adapter (ADR-106 D-8) for the
# serving wrapper (source me; never execute me directly).
#
# Why (2026-11-01): the maintainer chose DeepSeek-V4.1-Flash-1017 for the
# `svc-chat` role. Its only 2x DGX Spark route is the MiaAI-Lab recipe
# (github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks @0e7d421e):
# the TensorFold engine (BertholomusAI dsv41 fork) serving an EXL3 2.9-bpw
# quant, TP=2 over the 200G CX-7 link. Upstream's start.sh needs docker and
# passwordless ssh from the head to the worker. Neither exists here: the
# nodes have rootless podman only, and every per-node action goes through
# `srun --overlap`. This file rebuilds upstream's two run commands (start.sh
# "SERVE", "ENGINE_ENV", "ENV0", "FOLLOW") for rootless podman, with every
# optional upstream "port" feature (spill tier, KV sharding, ABLIT) off.
#
# Topology (one Slurm job, exactly 2 nodes):
#   rank 1  detached container `rivault-tf-r1` on the worker, started through
#           srun (a detached container escapes bdrt's ~60 s srun-step killer,
#           like the multi-node vLLM path);
#   rank 0  detached container `rivault-tf-r0` on the batch node (the head),
#           serving the OpenAI API on ${TF_PORT} with an API key file;
#   memguard  a per-node watcher that kills the local rank before a GB10 runs
#           out of unified memory (an OOM freezes the whole Spark; upstream
#           memguard.sh, with podman instead of docker).
# LiteLLM (compose) is repointed at http://<head fabric IP>:${TF_PORT}/v1.
#
# Contract (bash 4+):
#   source tensorfold_runner.sh
#   start_tensorfold        # preflight + launch; sets HEAD_IP, RAY_NODES
#   tf_launch_ranks         # relaunch both ranks (the monitor's respawn)
#   tensorfold_stop         # remove both ranks and the memguards
#
# Inputs (roster env_vars; defaults are upstream's unless noted):
#   TF_IMAGE              engine image; a localhost/* tag must be staged on
#                         both nodes (never pulled), anything else must carry
#                         an @sha256 digest
#   TF_HF_CACHE           host HF cache mounted at /hf:ro (default
#                         VLLM_PATH_TO_HUGGINGFACE)
#   TF_MODEL_SUB          EXL3 snapshot under /hf (hub/models--…/snapshots/<rev>)
#   TF_DATA_SUB           prepared data dir under /hf (tensorfold-dsv41/<rev8>)
#   TF_KIT_SUB            kernel kit dir under /hf (tensorfold-dsv41/kit-<ref8>)
#   TF_ENGRAM_SUB         Engram snapshot under /hf (fallback when no local copy)
#   TF_ENGRAM_DIR         node-local Engram copy on BOTH nodes (recommended;
#                         over NFS prefill drops ~2-3x)
#   TF_WORKER_CACHE_DIR   host dir holding rank 1's rank cache
#   TF_SERVED_NAME        served model name (default VLLM_SERVED_MODEL_NAME)
#   TF_PORT 8888  TF_MASTER_PORT 29620  TF_CONTEXT 262144 (ours; upstream 1M)
#   TF_PARALLEL 4  TF_MAX_TOKENS 32768  TF_TEMPERATURE 1.0  TF_TOP_P 0.95
#   TF_TOP_K 20  TF_THINKING 1  TF_REASONING_EFFORT ""  TF_DRAFTS 1
#   TF_GRAPHS 1  TF_GRAPH_CAP 128  TF_SIDE 1  TF_PREFETCH 1  TF_ENGRAM_AIO 1
#   TF_RDMA 1  TF_REPLAY_FLOOR 1  TF_RDMA_IDLE_S 2  TF_RDMA_NAP_US 500
#   TF_PREFILL_INTERLEAVE_MS 250  TF_CONNECT_S 1800 (ours; upstream 300)
#   TF_MIN_AVAIL_GIB 112  TF_MIN_AVAIL_WAIT_S 120
#   TF_MEMGUARD_GIB 2.5  TF_MEMGUARD_HARD_GIB 1  TF_MEMGUARD_HOLD_S 4
#   TF_FABRIC_IFACE       default VLLM_FABRIC_IFACE, then enp1s0f1np1; "auto" =
#                         the first up link with a RoCE device and an IPv4 (rccs
#                         QSFP pairs; the chosen name is logged per node)
#   RIVAULT_IMAGE_ARCHIVE_DIR  site image archives (missing images load from here)
#   TF_IB_HCA / TF_GID_INDEX   override the sysfs discovery (both nodes)
#   TF_EXTRA_SERVE_ARGS / TF_EXTRA_FOLLOW_ARGS   extra rank 0 / rank 1 flags
#   VLLM_API_KEY          the wrapper's engine key (written to a 0600 file)

TF_R0_NAME="rivault-tf-r0"
TF_VENUE_SH="${TF_VENUE_SH:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/venue.sh}"
TF_R1_NAME="rivault-tf-r1"
TF_KEY_IN_CONTAINER="/run/secrets/tensorfold-api-key"

tf_defaults() {
    : "${TF_HF_CACHE:=${VLLM_PATH_TO_HUGGINGFACE:-}}"
    : "${TF_SERVED_NAME:=${VLLM_SERVED_MODEL_NAME:-DeepSeek-V4.1-Flash-1017}}"
    : "${TF_PORT:=8888}" "${TF_MASTER_PORT:=29620}" "${TF_CONTEXT:=262144}"
    : "${TF_PARALLEL:=4}" "${TF_MAX_TOKENS:=32768}" "${TF_TEMPERATURE:=1.0}"
    : "${TF_TOP_P:=0.95}" "${TF_TOP_K:=20}" "${TF_THINKING:=1}" "${TF_DRAFTS:=1}"
    : "${TF_REASONING_EFFORT:=}" "${TF_GRAPHS:=1}" "${TF_GRAPH_CAP:=128}"
    : "${TF_SIDE:=1}" "${TF_PREFETCH:=1}" "${TF_ENGRAM_AIO:=1}" "${TF_RDMA:=1}"
    : "${TF_REPLAY_FLOOR:=1}" "${TF_RDMA_IDLE_S:=2}" "${TF_RDMA_NAP_US:=500}"
    : "${TF_PREFILL_INTERLEAVE_MS:=250}" "${TF_CONNECT_S:=1800}"
    : "${TF_MIN_AVAIL_GIB:=112}" "${TF_MIN_AVAIL_WAIT_S:=120}"
    : "${TF_MEMGUARD_GIB:=2.5}" "${TF_MEMGUARD_HARD_GIB:=1}" "${TF_MEMGUARD_HOLD_S:=4}"
    : "${TF_FABRIC_IFACE:=${VLLM_FABRIC_IFACE:-enp1s0f1np1}}"
    : "${TF_EXTRA_SERVE_ARGS:=}" "${TF_EXTRA_FOLLOW_ARGS:=}"
}

# The settings that must be present; prints the first missing one's name.
tf_missing_setting() {
    local name
    for name in TF_IMAGE TF_HF_CACHE TF_MODEL_SUB TF_DATA_SUB TF_KIT_SUB \
        TF_WORKER_CACHE_DIR VLLM_API_KEY; do
        if [[ -z "${!name:-}" ]]; then
            echo "${name}"
            return 0
        fi
    done
    if [[ -z "${TF_ENGRAM_DIR:-}" && -z "${TF_ENGRAM_SUB:-}" ]]; then
        echo "TF_ENGRAM_DIR or TF_ENGRAM_SUB"
    fi
}

# A localhost/* image is staged per node; any other reference must be pinned
# by digest (community images drift; ADR-106 D-8).
tf_image_is_pinned() {
    case "$1" in
        localhost/*) return 0 ;;
        *@sha256:*) return 0 ;;
    esac
    return 1
}

# IPv4 a.b.c.d -> the RoCE v2 GID text sysfs shows for it.
tf_ipv4_gid() {
    local a b c d
    IFS=. read -r a b c d <<< "$1"
    printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' "$a" "$b" "$c" "$d"
}

# The node-side probe, run through srun on each node. Prints one line:
#   <ipv4> <hca> <gid index> <MemAvailable GiB> <memlock hard limit>
# shellcheck disable=SC2016  # expands on the node
TF_PROBE_SCRIPT='
dev="$1"
if [ "$dev" = auto ]; then
  dev=""
  for d in /sys/class/net/*; do
    n=${d##*/}
    [ -e "$d/device/infiniband" ] || continue
    [ "$(cat "$d/operstate" 2>/dev/null)" = up ] || continue
    ip -4 -o addr show dev "$n" 2>/dev/null | grep -q inet || continue
    dev=$n; break
  done
fi
ip=$(ip -4 -o addr show dev "$dev" 2>/dev/null | awk "{sub(/\\/.*/, \"\", \$4); print \$4; exit}")
hca=$(ls "/sys/class/net/$dev/device/infiniband" 2>/dev/null | head -1)
gid="-"
if [ -n "$ip" ] && [ -n "$hca" ]; then
  want=$(printf "0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x" $(echo "$ip" | tr . " "))
  for g in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
    i=${g##*/}
    [ "$(cat "$g" 2>/dev/null)" = "$want" ] || continue
    t=$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"$i" 2>/dev/null)
    case "$t" in *"v2"*) gid=$i; break ;; esac
  done
fi
mem=$(awk "/^MemAvailable:/ {print int(\$2/1048576)}" /proc/meminfo)
echo "${ip:--} ${hca:--} ${gid} ${mem:-0} $(ulimit -Hl) ${dev:--}"
'

tf_probe_node() {
    srun --nodes=1 --ntasks=1 --nodelist="$1" --overlap \
        bash -c "${TF_PROBE_SCRIPT}" _ "${TF_FABRIC_IFACE}"
}

# Parse one probe line into TF_NODE_{IP,HCA,GID,MEM,MEMLOCK}[rank].
tf_record_probe() {
    local rank="$1" line="$2" ip hca gid mem memlock dev
    read -r ip hca gid mem memlock dev <<< "${line}"
    TF_NODE_IP[rank]="${ip}"
    TF_NODE_IFACE[rank]="${dev:-${TF_FABRIC_IFACE}}"
    TF_NODE_HCA[rank]="${TF_IB_HCA:-${hca}}"
    TF_NODE_GID[rank]="${TF_GID_INDEX:-${gid}}"
    TF_NODE_MEM[rank]="${mem:-0}"
    TF_NODE_MEMLOCK[rank]="${memlock:-}"
}

# rc 1 (and one FATAL line) when a probed node cannot run its rank.
tf_check_probe() {
    local rank="$1" node="$2"
    if [[ "${TF_NODE_IP[rank]}" == "-" || -z "${TF_NODE_IP[rank]}" ]]; then
        echo "[rivault] FATAL: no IPv4 on ${TF_FABRIC_IFACE} on ${node}" >&2
        return 1
    fi
    if [[ "${TF_RDMA}" == 1 && ( "${TF_NODE_HCA[rank]}" == "-" || "${TF_NODE_GID[rank]}" == "-" ) ]]; then
        echo "[rivault] FATAL: no RoCE v2 HCA/GID for ${TF_FABRIC_IFACE} on ${node} (set TF_IB_HCA/TF_GID_INDEX, or TF_RDMA=0)" >&2
        return 1
    fi
    if [[ "${TF_RDMA}" == 1 && "${TF_NODE_MEMLOCK[rank]}" != "unlimited" ]]; then
        echo "[rivault] FATAL: memlock hard limit is ${TF_NODE_MEMLOCK[rank]} on ${node}; RDMA needs unlimited (or TF_RDMA=0)" >&2
        return 1
    fi
    return 0
}

# Both nodes must have TF_MIN_AVAIL_GIB free (98.9 GiB weights + the
# engine's reserve). Re-probes until TF_MIN_AVAIL_WAIT_S has passed.
tf_await_memory() {
    local deadline=$(( SECONDS + TF_MIN_AVAIL_WAIT_S )) rank short
    while true; do
        short=""
        for rank in 0 1; do
            tf_record_probe "${rank}" "$(tf_probe_node "${RAY_NODES[rank]}")"
            if (( TF_NODE_MEM[rank] < TF_MIN_AVAIL_GIB )); then
                short="${RAY_NODES[rank]}=${TF_NODE_MEM[rank]}GiB"
            fi
        done
        [[ -z "${short}" ]] && return 0
        if (( SECONDS >= deadline )); then
            echo "[rivault] FATAL: ${short} available, ${TF_MIN_AVAIL_GIB} GiB needed per node" >&2
            return 1
        fi
        sleep 5
    done
}

tf_engram_host_dir() {
    echo "${TF_ENGRAM_DIR:-${TF_HF_CACHE}/${TF_ENGRAM_SUB}}"
}

# Arguments both ranks' containers share (rootless GB10 device passthrough,
# the same device and driver-library set the multi-node vLLM path uses).
tf_common_run_args() {
    TF_RUN_ARGS=(
        --network host --ipc host
        --userns "${RIVAULT_USERNS_MODE:-keep-id:uid=0,gid=0}"
        --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864
        --cap-add IPC_LOCK --cap-add SYS_NICE
        --device /dev/nvidia0 --device /dev/nvidiactl
        --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools
        --device /dev/nvidia-caps/nvidia-cap1 --device /dev/nvidia-caps/nvidia-cap2
        -v /usr/lib/aarch64-linux-gnu/libcuda.so.1:/usr/lib/aarch64-linux-gnu/libcuda.so.1:ro
        -v /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:ro
        -v /usr/lib/aarch64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro
        -e NVIDIA_VISIBLE_DEVICES=all -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
        -e CUDA_VISIBLE_DEVICES=0
        -v "${TF_HF_CACHE}:/hf:ro"
    )
    if [[ "${TF_RDMA}" == 1 ]]; then
        TF_RUN_ARGS+=( --device /dev/infiniband )
    fi
}

# upstream ENGINE_ENV: the same on both ranks.
tf_engine_env() {
    TF_ENGINE_ENV=(
        "TF_DS_REPLAY_FLOOR=${TF_REPLAY_FLOOR}" "TF_DS_STREAMS=${TF_PARALLEL}"
        "TF_RDMA_IDLE_S=${TF_RDMA_IDLE_S}" "TF_RDMA_NAP_US=${TF_RDMA_NAP_US}"
        "HF_HUB_OFFLINE=1" "TF_DS_CONNECT_S=${TF_CONNECT_S}"
    )
    if (( TF_PREFILL_INTERLEAVE_MS != 0 )); then
        TF_ENGINE_ENV+=( "TF_DS_PREFILL_INTERLEAVE_MS=${TF_PREFILL_INTERLEAVE_MS}" )
    fi
}

# NCCL over the fabric link for rank $1 (upstream rank_nccl_env).
tf_nccl_env() {
    local rank="$1"
    TF_NCCL_ENV=( "NCCL_SOCKET_IFNAME=${TF_NODE_IFACE[rank]:-${TF_FABRIC_IFACE}}" "NCCL_CUMEM_ENABLE=0" )
    if [[ "${TF_RDMA}" == 1 ]]; then
        TF_NCCL_ENV+=( "NCCL_IB_HCA=${TF_NODE_HCA[rank]}" "NCCL_IB_GID_INDEX=${TF_NODE_GID[rank]}"
                       "TF_RDMA_GID_INDEX=${TF_NODE_GID[rank]}" )
    else
        TF_NCCL_ENV+=( "NCCL_IB_DISABLE=1" )
    fi
}

# rank 0's engine command (upstream SERVE).
tf_serve_cmd() {
    TF_SERVE=( tensorfold-native serve "/hf/${TF_DATA_SUB}/model" --backend cuda
        --host 0.0.0.0 --port "${TF_PORT}" --name "${TF_SERVED_NAME}"
        --context "${TF_CONTEXT}" --parallel "${TF_PARALLEL}"
        --max-tokens "${TF_MAX_TOKENS}" --temperature "${TF_TEMPERATURE}"
        --top-p "${TF_TOP_P}" --top-k "${TF_TOP_K}"
        --api-key-file "${TF_KEY_IN_CONTAINER}" )
    if [[ "${TF_THINKING}" == 1 ]]; then TF_SERVE+=( --thinking ); else TF_SERVE+=( --no-thinking ); fi
    if [[ -n "${TF_REASONING_EFFORT}" ]]; then TF_SERVE+=( --reasoning-effort "${TF_REASONING_EFFORT}" ); fi
    if [[ "${TF_DRAFTS}" != 1 ]]; then TF_SERVE+=( --no-drafts ); fi
    if [[ -n "${TF_EXTRA_SERVE_ARGS}" ]]; then
        local -a extra
        read -r -a extra <<< "${TF_EXTRA_SERVE_ARGS}"
        TF_SERVE+=( "${extra[@]}" )
    fi
}

# rank 1's engine command (upstream "tf-dsv41-lanes … $FOLLOW"). Its pool and
# streams must equal rank 0's context and parallel, or the ranks wait on each
# other every round.
tf_follow_cmd() {
    TF_FOLLOW=( tf-dsv41-lanes "/hf/${TF_MODEL_SUB}" /rank-cache 1 2
        "${TF_NODE_IP[0]}" "${TF_MASTER_PORT}" "/hf/${TF_KIT_SUB}/kit"
        --engram /engram --token-map "/hf/${TF_DATA_SUB}/dsv41_token_map.json"
        --pool "${TF_CONTEXT}" --streams "${TF_PARALLEL}" --drafts "${TF_DRAFTS}"
        --graphs "${TF_GRAPHS}" --side "${TF_SIDE}" --prefetch "${TF_PREFETCH}"
        --aio "${TF_ENGRAM_AIO}" )
    if [[ "${TF_GRAPHS}" != 0 ]]; then TF_FOLLOW+=( --graph-cap "${TF_GRAPH_CAP}" ); fi
    if [[ "${TF_RDMA}" == 1 ]]; then TF_FOLLOW+=( --rdma "${TF_NODE_HCA[1]}" ); fi
    if [[ -n "${TF_EXTRA_FOLLOW_ARGS}" ]]; then
        local -a extra
        read -r -a extra <<< "${TF_EXTRA_FOLLOW_ARGS}"
        TF_FOLLOW+=( "${extra[@]}" )
    fi
}

# Full `podman run` argv for rank 1 (on the worker) into TF_R1_ARGV.
tf_rank1_argv() {
    local x
    tf_common_run_args; tf_engine_env; tf_nccl_env 1; tf_follow_cmd
    TF_R1_ARGV=( podman run -d --name "${TF_R1_NAME}" "${TF_RUN_ARGS[@]}"
        -v "$(tf_engram_host_dir):/engram:ro" -v "${TF_WORKER_CACHE_DIR}:/rank-cache:ro" )
    for x in "${TF_ENGINE_ENV[@]}" "${TF_NCCL_ENV[@]}"; do TF_R1_ARGV+=( -e "${x}" ); done
    TF_R1_ARGV+=( "${TF_IMAGE}" "${TF_FOLLOW[@]}" )
}

# Full `podman run` argv for rank 0 (on the head) into TF_R0_ARGV.
tf_rank0_argv() {
    local x engram="/hf/${TF_ENGRAM_SUB:-}"
    tf_common_run_args; tf_engine_env; tf_nccl_env 0; tf_serve_cmd
    TF_R0_ARGV=( podman run -d --name "${TF_R0_NAME}" "${TF_RUN_ARGS[@]}"
        -v "${TF_KEY_FILE}:${TF_KEY_IN_CONTAINER}:ro" )
    if [[ -n "${TF_ENGRAM_DIR:-}" ]]; then
        TF_R0_ARGV+=( -v "${TF_ENGRAM_DIR}:/engram:ro" )
        engram="/engram"
    fi
    for x in TF_TP_WORLD=2 "TF_TP_MASTER=${TF_NODE_IP[0]}" "TF_TP_PORT=${TF_MASTER_PORT}" \
        "TF_DS_RANK_CACHE=/hf/${TF_DATA_SUB}/rank-cache" "TF_DS_KIT=/hf/${TF_KIT_SUB}/kit" \
        "TENSORFOLD_CUDA_KERNELS=/hf/${TF_KIT_SUB}/kit/aot" "TF_DS_ENGRAM=${engram}" \
        "TF_DS_TOKEN_MAP=/hf/${TF_DATA_SUB}/dsv41_token_map.json" \
        "TF_DS_GRAPHS=${TF_GRAPHS}" "TF_DS_GRAPH_CAP=${TF_GRAPH_CAP}" "TF_DS_HC_SIDE=${TF_SIDE}" \
        "TF_DS_L2_PREFETCH=${TF_PREFETCH}" "TF_DS_ENGRAM_AIO=${TF_ENGRAM_AIO}" \
        "${TF_ENGINE_ENV[@]}" "${TF_NCCL_ENV[@]}"; do
        TF_R0_ARGV+=( -e "${x}" )
    done
    if [[ "${TF_RDMA}" == 1 ]]; then TF_R0_ARGV+=( -e "TF_RDMA_DEVICES=${TF_NODE_HCA[0]}" ); fi
    TF_R0_ARGV+=( "${TF_IMAGE}" "${TF_SERVE[@]}" )
}

# The engine key as a 0600 file next to the job (the engine rereads it on
# change; never on a command line, never in `ps`).
tf_write_key_file() {
    TF_KEY_FILE="${RIVAULT_WRAPPER_DIR:-.}/jobs/tf-api-key.${SLURM_JOB_ID:-0}"
    mkdir -p "$(dirname "${TF_KEY_FILE}")"
    ( umask 077; printf '%s\n' "${VLLM_API_KEY}" > "${TF_KEY_FILE}" )
}

# Per-node memguard (upstream memguard.sh, podman instead of docker): kill the
# local rank when MemAvailable stays under TF_MEMGUARD_GIB for
# TF_MEMGUARD_HOLD_S, or under TF_MEMGUARD_HARD_GIB for two samples. The
# monitor then sees the head rank exited and relaunches the pair.
# shellcheck disable=SC2016  # expands on the node
TF_MEMGUARD_SCRIPT='
ctr="$1"; soft_kb=$(awk -v g="$2" "BEGIN{print int(g*1048576)}"); hard_kb=$(awk -v g="$3" "BEGIN{print int(g*1048576)}")
hold=$(( $4 * 4 )); low=0; hard=0
sleep 30
while podman container exists "$ctr" 2>/dev/null; do
  kb=$(awk "/^MemAvailable:/ {print \$2}" /proc/meminfo)
  if [ "$kb" -lt "$hard_kb" ]; then hard=$((hard+1)); else hard=0; fi
  if [ "$kb" -lt "$soft_kb" ]; then low=$((low+1)); else low=0; fi
  if [ "$hard" -ge 2 ] || [ "$low" -ge "$hold" ]; then
    echo "memguard: MemAvailable ${kb} kB; killing $ctr"
    timeout 20 podman kill "$ctr"; exit 0
  fi
  sleep 0.25
done
'

tf_memguard_unit() { echo "rivault-tf-memguard-${SLURM_JOB_ID:-0}-$1"; }

tf_start_memguard() {
    local rank="$1" node="$2" ctr="$3"
    srun --nodes=1 --ntasks=1 --nodelist="${node}" --overlap \
        systemd-run --user --unit="$(tf_memguard_unit "${rank}")" --collect \
        bash -c "${TF_MEMGUARD_SCRIPT}" _ "${ctr}" "${TF_MEMGUARD_GIB}" \
        "${TF_MEMGUARD_HARD_GIB}" "${TF_MEMGUARD_HOLD_S}" >/dev/null 2>&1 || \
        echo "[rivault] WARNING: memguard not started on ${node} (rank ${rank})"
}

# Launch rank 1 on the worker, then rank 0 here, each with its memguard.
tf_launch_ranks() {
    srun --nodes=1 --ntasks=1 --nodelist="${RAY_NODES[1]}" --overlap \
        podman rm -f "${TF_R1_NAME}" >/dev/null 2>&1 || true
    podman rm -f "${TF_R0_NAME}" >/dev/null 2>&1 || true
    tf_rank1_argv
    if ! srun --nodes=1 --ntasks=1 --nodelist="${RAY_NODES[1]}" --overlap "${TF_R1_ARGV[@]}" >/dev/null; then
        echo "[rivault] FATAL: rank 1 did not start on ${RAY_NODES[1]}" >&2
        return 1
    fi
    tf_rank0_argv
    if ! "${TF_R0_ARGV[@]}" >/dev/null; then
        echo "[rivault] FATAL: rank 0 did not start on ${RAY_NODES[0]}" >&2
        return 1
    fi
    tf_start_memguard 1 "${RAY_NODES[1]}" "${TF_R1_NAME}"
    tf_start_memguard 0 "${RAY_NODES[0]}" "${TF_R0_NAME}"
    echo "[rivault] TensorFold ranks started: r0=${RAY_NODES[0]} r1=${RAY_NODES[1]} (weights load in background)"
}

# rc 1 unless the image is present on every node of the job.
tf_check_image() {
    local node
    if ! tf_image_is_pinned "${TF_IMAGE}"; then
        echo "[rivault] FATAL: TF_IMAGE ${TF_IMAGE} must be localhost/* (staged) or pinned @sha256" >&2
        return 1
    fi
    for node in "${RAY_NODES[@]}"; do
        if ! tf_ensure_image_on "${node}"; then
            echo "[rivault] FATAL: ${TF_IMAGE} is not staged on ${node} (nor in ${RIVAULT_IMAGE_ARCHIVE_DIR:-the archive dir})" >&2
            return 1
        fi
    done
}

# Present on the node, or loaded there from the site's image archive
# (venue.sh rivault_ensure_image; RCCS node image stores are node-local, so a
# localhost/* image reaches a node only as <dir>/<name>-<tag>-<arch>.tar).
# shellcheck disable=SC2016  # expands on the node
tf_ensure_image_on() {
    srun --nodes=1 --ntasks=1 --nodelist="$1" --overlap bash -c \
        'source "$1" && rivault_ensure_image "$2" "$3"' _ \
        "${TF_VENUE_SH}" "${TF_IMAGE}" "${RIVAULT_IMAGE_ARCHIVE_DIR:-}"
}

# rc 1 unless every prepared path exists (checked on the worker: NFS is shared,
# the local Engram copy and rank 1's cache are per node).
tf_check_paths() {
    local -a paths=( "${TF_HF_CACHE}/${TF_MODEL_SUB}" "${TF_HF_CACHE}/${TF_DATA_SUB}/model"
        "${TF_HF_CACHE}/${TF_DATA_SUB}/rank-cache" "${TF_HF_CACHE}/${TF_KIT_SUB}/kit"
        "${TF_HF_CACHE}/${TF_DATA_SUB}/dsv41_token_map.json" "$(tf_engram_host_dir)"
        "${TF_WORKER_CACHE_DIR}" )
    local node p
    for node in "${RAY_NODES[@]}"; do
        for p in "${paths[@]}"; do
            if ! srun --nodes=1 --ntasks=1 --nodelist="${node}" --overlap test -e "${p}"; then
                echo "[rivault] FATAL: ${p} missing on ${node} (stage it first; see the service-model runbook)" >&2
                return 1
            fi
        done
    done
}

# Resolve both nodes and run every check that needs no GPU.
tf_preflight() {
    local missing node_count="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}" rank
    missing="$(tf_missing_setting)"
    if [[ -n "${missing}" ]]; then
        echo "[rivault] FATAL: TensorFold needs ${missing}" >&2
        return 1
    fi
    if [[ "${node_count}" -ne 2 ]]; then
        echo "[rivault] FATAL: TensorFold DeepSeek-V4.1 needs exactly two nodes, got ${node_count}" >&2
        return 1
    fi
    mapfile -t RAY_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST:-}")
    if [[ "${SLURMD_NODENAME:-$(hostname -s)}" != "${RAY_NODES[0]}" ]]; then
        echo "[rivault] FATAL: batch step not on the head node ${RAY_NODES[0]}" >&2
        return 1
    fi
    tf_check_image || return 1
    tf_check_paths || return 1
    for rank in 0 1; do
        tf_record_probe "${rank}" "$(tf_probe_node "${RAY_NODES[rank]}")"
        tf_check_probe "${rank}" "${RAY_NODES[rank]}" || return 1
    done
}

# Point the job's LiteLLM at rank 0 (it was rendered for the compose vllm).
tf_repoint_litellm() {
    local cfg="$1"
    sed -i "s@http://vllm:8000/v1@http://${HEAD_IP}:${TF_PORT}/v1@g" "${cfg}"
    echo "[rivault] LiteLLM repointed to TensorFold rank 0 at http://${HEAD_IP}:${TF_PORT}/v1"
}

start_tensorfold() {
    declare -g -a TF_NODE_IP=() TF_NODE_IFACE=() TF_NODE_HCA=() TF_NODE_GID=() TF_NODE_MEM=() TF_NODE_MEMLOCK=()
    declare -g -a RAY_NODES=()
    tf_defaults
    tf_preflight || return 1
    tf_await_memory || return 1
    HEAD_IP="${TF_NODE_IP[0]}"
    echo "[rivault] TensorFold fabric: ${RAY_NODES[0]}=${TF_NODE_IP[0]} via ${TF_NODE_IFACE[0]}, ${RAY_NODES[1]} via ${TF_NODE_IFACE[1]} (${TF_NODE_HCA[0]} gid ${TF_NODE_GID[0]}), ${RAY_NODES[1]}=${TF_NODE_IP[1]} (${TF_NODE_HCA[1]} gid ${TF_NODE_GID[1]})"
    tf_write_key_file
    tf_launch_ranks
}

tensorfold_stop() {
    local rank node
    for rank in 0 1; do
        node="${RAY_NODES[rank]:-}"
        [[ -n "${node}" ]] || continue
        srun --nodes=1 --ntasks=1 --nodelist="${node}" --overlap bash -c \
            "systemctl --user stop '$(tf_memguard_unit "${rank}").service' >/dev/null 2>&1; podman stop -t 30 '${TF_R0_NAME}' '${TF_R1_NAME}' >/dev/null 2>&1; podman rm -f '${TF_R0_NAME}' '${TF_R1_NAME}' >/dev/null 2>&1" \
            >/dev/null 2>&1 || true
    done
    rm -f "${TF_KEY_FILE:-}" >/dev/null 2>&1 || true
}
