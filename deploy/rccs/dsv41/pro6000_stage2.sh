#!/bin/bash
# ai-pro6000 stage 2 (run by pro6000_entry.sh inside the queued job): DeepSeek-V4.1-Flash-1017 NVFP4 on SGLang TP=4
# over both nodes via the branch's sglang_runner.sh, serving benchmark, hardware microbenchmarks, then serve until the
# walltime. Every phase logs to the job log; results land in ~/tmp/dsv41/bench.
set -uo pipefail
D=$HOME/tmp/dsv41
MR=$D/mr
B=$D/bench
export RIVAULT_WRAPPER_DIR=$D/run
mkdir -p "$RIVAULT_WRAPPER_DIR/jobs" "$B/hw" "$B/serve"
REPO=$HOME/umrs_models_hs/hub/models--nvidia--DeepSeek-V4.1-Flash-1017-NVFP4
export SG_IMAGE=docker.io/lmsysorg/sglang@sha256:bc4aab534886dfc9e2c1d7425eca14e1f9d03a518a29074928c4f385649d9844
export SG_MODEL_ROOT=$REPO SG_MODEL_DIR=$REPO/snapshots/3ae9c0d3fbbea5370bd06e679d22044db15883f1
export SG_GPUS_PER_NODE=2 SG_TP=4 SG_CONTEXT=${SG_CONTEXT:-131072} SG_KV_DTYPE=fp8_e4m3
export SG_MEM_FRACTION=${SG_MEM_FRACTION:-0.88} SG_SPEC=${SG_SPEC-DSPARK} SG_SPEC_BLOCK=5
export SG_ENGRAM_CONFIG='{"placement":"cpu","replication":"none","lookup":"auto"}'
export SG_SERVED_NAME=DeepSeek-V4.1-Flash-1017 SG_RDMA=${SG_RDMA:-auto} SG_GPU_FORM=${SG_GPU_FORM:-auto}
export RIVAULT_MN_WORKER_MODE=step RIVAULT_USERNS_MODE=host
# rendezvous on the default-route NIC (a fast link may drop TCP, as the rccs QSFP does); RDMA still from the probe
SG_FABRIC_IFACE=${SG_FABRIC_IFACE:-$(ip route show default | awk '{for (i=1;i<NF;i++) if ($i=="dev") {print $(i+1); exit}}')}
export SG_FABRIC_IFACE
KEYF=$D/tf-api-key
[ -s "$KEYF" ] || ( umask 077; head -c 24 /dev/urandom | base64 | tr -d '/+=' > "$KEYF" )
VLLM_API_KEY=$(cat "$KEYF"); export VLLM_API_KEY
log() { echo "$(date -Is) $*"; }
log "stage2 job=$SLURM_JOB_ID fabric=$SG_FABRIC_IFACE spec=$SG_SPEC"
[ -s "$SG_MODEL_DIR/model.safetensors.index.json" ] || { log "checkpoint not staged"; echo "$SLURM_JOB_ID checkpoint" > "$D/run/pro6000.FAILED"; exit 1; }
# shellcheck disable=SC1091
source "$MR/venue.sh"
rivault_venue_prepare "$SLURM_JOB_ID" cgroupfs
export SG_VENUE_SH=$MR/venue.sh
# shellcheck disable=SC1091
source "$MR/sglang_runner.sh"
trap 'log stopping; sglang_stop; exit 0' TERM INT

wait_ready() { # timeout_s
  local t0=$SECONDS
  while (( SECONDS - t0 < $1 )); do
    if curl -s -m 5 -H "Authorization: Bearer $VLLM_API_KEY" "http://${HEAD_IP}:${SG_PORT}/v1/models" | grep -q DeepSeek; then
      log "READY after $((SECONDS - t0)) s"; return 0; fi
    st=$(podman inspect rivault-sg-r0 --format '{{.State.Status}}' 2>/dev/null)
    if [ "$st" != running ]; then log "rank 0 is '${st:-gone}'"; podman logs --tail 80 rivault-sg-r0 2>&1 | sed 's/^/  r0| /'; return 1; fi
    sleep 20
  done
  log "not ready after $1 s"; podman logs --tail 40 rivault-sg-r0 2>&1 | sed 's/^/  r0| /'; return 1
}
mem_report() { srun --ntasks-per-node=1 --overlap bash -c 'echo "$(hostname) $(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tr "\n" " ") host_avail=$(awk "/MemAvailable/ {printf \"%.0fG\", \$2/1048576}" /proc/meminfo)"'; }

start_sglang_mn || { log "START FAILED"; sglang_stop; echo "$SLURM_JOB_ID start" > "$D/run/pro6000.FAILED"; exit 1; }
echo "http://${HEAD_IP}:${SG_PORT}" > "$D/run/endpoint-sg.$SLURM_JOB_ID"
if [ "${BENCH:-1}" != 1 ]; then
  wait_ready 7200; mem_report
elif wait_ready 7200; then
  mem_report
  python3 "$B/serve_bench.py" --url "http://${HEAD_IP}:${SG_PORT}" --model DeepSeek-V4.1-Flash-1017 --key-file "$KEYF" \
    --label "pro6000-sglang-nvfp4-tp4-spec${SG_SPEC:-off}" --conc 1,2,4,8 \
    --out "$B/serve/pro6000-sglang-tp4-spec${SG_SPEC:-off}.$SLURM_JOB_ID.json" \
    --meta "{\"platform\":\"ai-pro6000 2x(2x RTX PRO 6000 96GB)\",\"engine\":\"sglang 314bce7c\",\"weights\":\"nvidia NVFP4 3ae9c0d3\",\"tp\":4,\"spec\":\"${SG_SPEC:-off}\",\"fabric\":\"${SG_FABRIC_IFACE}\",\"rdma\":\"${SG_RDMA_ON:-?}\"}" \
    2>&1 | tail -40
  mem_report
fi
[ "${BENCH:-1}" = 1 ] && sglang_stop
# hardware microbenchmarks in the SGLang image (torch + NCCL): intra-node (2 GPUs over PCIe) and inter-node (4 ranks)
mapfile -t N < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
M0=$(ip -4 -o addr show dev "$SG_FABRIC_IFACE" | awk '{sub(/\/.*/,"",$4); print $4; exit}')
hw_run() { # label world nodes... -- env...
  local label=$1 world=$2; shift 2
  local -a nodes=(); while [ "$1" != -- ]; do nodes+=("$1"); shift; done; shift
  local port=$((29900 + RANDOM % 90)) rank=0 n g
  for n in "${nodes[@]}"; do
    for g in 0 1; do
      srun -N1 -n1 -w "$n" --overlap bash -c 'source "$1" && rivault_venue_prepare "$2" cgroupfs >/dev/null 2>&1; shift 2; exec "$@"' _ "$MR/venue.sh" "$SLURM_JOB_ID" \
        podman run --rm --network host --ipc host --userns host --ulimit memlock=-1:-1 "${SG_GPU_ARGS[@]}" \
        ${IBDEV[@]+"${IBDEV[@]}"} -v "$B:/bench" -e RANK=$rank -e WORLD_SIZE="$world" -e LOCAL_RANK=$g \
        -e MASTER_ADDR="$M0" -e MASTER_PORT=$port -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET "$@" \
        --entrypoint python3 "$SG_IMAGE" /bench/hw_bench.py --label "$label" --out "/bench/hw/$label.$SLURM_JOB_ID.json" \
        > "$B/hw/$label.$SLURM_JOB_ID.rank$rank.log" 2>&1 &
      rank=$((rank + 1))
    done
  done
  wait
  grep -h "allreduce_8KiB\|NET/IB\|NET/Socket\|Using network" "$B/hw/$label.$SLURM_JOB_ID".rank*.log | sort -u | head -6
}
IBENV=(); [ -n "${SG_IB_HCA:-${SG_NODE_HCA[0]:-}}" ] && IBENV=(-e NCCL_IB_HCA="${SG_IB_HCA:-${SG_NODE_HCA[0]}}" -e NCCL_IB_GID_INDEX="${SG_GID_INDEX:-${SG_NODE_GID[0]:-3}}")
sg_gpu_args 0
IBDEV=(); [ -e /dev/infiniband ] && IBDEV=(--device /dev/infiniband)
if [ "${BENCH:-1}" = 1 ]; then
hw_run pro6000-intra 2 "${N[0]}" -- -e NCCL_SOCKET_IFNAME="$SG_FABRIC_IFACE"
hw_run pro6000-inter-ib 4 "${N[@]}" -- -e NCCL_SOCKET_IFNAME="$SG_FABRIC_IFACE" "${IBENV[@]}"
hw_run pro6000-inter-tcp 4 "${N[@]}" -- -e NCCL_SOCKET_IFNAME="$SG_FABRIC_IFACE" -e NCCL_IB_DISABLE=1
# serve for the rest of the walltime
start_sglang_mn || { log "RELAUNCH FAILED"; echo "$SLURM_JOB_ID relaunch" > "$D/run/pro6000.FAILED"; exit 1; }
wait_ready 7200
fi
relaunches=0
while true; do
  sleep 30
  st=$(podman inspect rivault-sg-r0 --format '{{.State.Status}}' 2>/dev/null)
  if [ "$st" != running ]; then
    relaunches=$((relaunches + 1)); log "rank 0 '${st:-gone}', relaunch $relaunches"
    [ "$relaunches" -le 3 ] || break
    sg_launch_ranks && wait_ready 7200
  fi
done
sglang_stop
