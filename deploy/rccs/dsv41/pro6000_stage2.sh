#!/bin/bash
# ai-pro6000 stage 2 (run by pro6000_entry.sh inside the queued job): DeepSeek-V4.1-Flash-1017 NVFP4 on SGLang TP=4
# over both nodes via the branch's sglang_runner.sh in its singularity mode (the nodes have no podman: job 468124),
# serving benchmarks (IB, TCP, speculative decoding off), hardware microbenchmarks, then serve until the walltime.
# BENCH=0 (the keep-alive successors) skips straight to serving. Results land in ~/tmp/dsv41/bench.
set -uo pipefail
D=$HOME/tmp/dsv41
MR=$D/mr
B=$D/bench
export RIVAULT_WRAPPER_DIR=$D/run
mkdir -p "$RIVAULT_WRAPPER_DIR/jobs" "$B/hw" "$B/serve"
REPO=$HOME/umrs_models_hs/hub/models--nvidia--DeepSeek-V4.1-Flash-1017-NVFP4
export SG_RUNTIME=singularity SG_IMAGE=$HOME/umrs_models_hs/images/sglang-314bce7c-arm64.sif
export SG_MODEL_ROOT=$REPO SG_MODEL_DIR=$REPO/snapshots/3ae9c0d3fbbea5370bd06e679d22044db15883f1
export SG_GPUS_PER_NODE=2 SG_TP=4 SG_CONTEXT=${SG_CONTEXT:-131072} SG_KV_DTYPE=fp8_e4m3
export SG_MEM_FRACTION=${SG_MEM_FRACTION:-0.88} SG_SPEC=${SG_SPEC-DSPARK} SG_SPEC_BLOCK=5
export SG_ENGRAM_CONFIG='{"placement":"cpu","replication":"none","lookup":"auto"}'
export SG_SERVED_NAME=DeepSeek-V4.1-Flash-1017 SG_RDMA=${SG_RDMA:-auto} SG_GPU_FORM=devices
export SG_WORKER_ATTEMPTS=3 RIVAULT_MN_WORKER_MODE=step
# the batch shell holds only one of a node's two GPUs (job 468620): every rank and microbench step asks for both
export SG_STEP_GRES=gpu:2
# rendezvous on the default-route NIC (a fast link may drop TCP, as rccs's QSFP does); the probe falls back to the
# first ACTIVE HCA for NCCL (ai-pro6000: mlx5_0, InfiniBand NDR 400)
SG_FABRIC_IFACE=${SG_FABRIC_IFACE:-$(ip route show default | awk '{for (i=1;i<NF;i++) if ($i=="dev") {print $(i+1); exit}}')}
export SG_FABRIC_IFACE
KEYF=$D/tf-api-key
[ -s "$KEYF" ] || ( umask 077; head -c 24 /dev/urandom | base64 | tr -d '/+=' > "$KEYF" )
VLLM_API_KEY=$(cat "$KEYF"); export VLLM_API_KEY
log() { echo "$(date -Is) $*"; }
fail() { log "$1 FAILED"; echo "$SLURM_JOB_ID $1" > "$D/run/pro6000.FAILED"; exit 1; }
log "stage2 job=$SLURM_JOB_ID fabric=$SG_FABRIC_IFACE spec=$SG_SPEC runtime=$SG_RUNTIME bench=${BENCH:-1}"
command -v singularity >/dev/null || fail "no singularity on $(hostname -s):"
[ -s "$SG_MODEL_DIR/model.safetensors.index.json" ] || fail checkpoint
# shellcheck disable=SC1091
source "$MR/venue.sh"
export SG_VENUE_SH=$MR/venue.sh
# shellcheck disable=SC1091
source "$MR/sglang_runner.sh"
trap 'log stopping; sglang_stop; exit 0' TERM INT

wait_ready() { # timeout_s
  local t0=$SECONDS
  while (( SECONDS - t0 < $1 )); do
    if curl -s -m 5 -H "Authorization: Bearer $VLLM_API_KEY" "http://${HEAD_IP}:${SG_PORT}/v1/models" | grep -q DeepSeek; then
      log "READY after $((SECONDS - t0)) s"; return 0; fi
    if ! sg_rank0_alive; then log "rank 0 exited"; tail -60 "$(sg_log_file 0)" | sed 's/^/  r0| /'; tail -20 "$(sg_log_file 1)" 2>/dev/null | sed 's/^/  r1| /'; return 1; fi
    sleep 20
  done
  log "not ready after $1 s"; tail -40 "$(sg_log_file 0)" | sed 's/^/  r0| /'; return 1
}
mem_report() { srun --ntasks-per-node=1 --overlap bash -c 'echo "$(hostname -s) $(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tr "\n" " ") host_avail=$(awk "/MemAvailable/ {printf \"%.0fG\", \$2/1048576}" /proc/meminfo)"'; }
bench() { # label conc
  python3 "$B/serve_bench.py" --url "http://${HEAD_IP}:${SG_PORT}" --model DeepSeek-V4.1-Flash-1017 --key-file "$KEYF" \
    --label "$1" --conc "$2" --out "$B/serve/$1.$SLURM_JOB_ID.json" \
    --meta "{\"platform\":\"ai-pro6000 2x(2x RTX PRO 6000 Blackwell Max-Q 96GB)\",\"engine\":\"sglang 314bce7c\",\"weights\":\"nvidia NVFP4 3ae9c0d3\",\"tp\":4,\"spec\":\"${SG_SPEC:-off}\",\"rdma\":\"${SG_RDMA_ON:-?}\",\"hca\":\"${SG_NODE_HCA[0]:-}\",\"fabric\":\"${SG_FABRIC_IFACE}\"}" \
    2>&1 | tail -40
}
# one benchmarked configuration: launch, wait, measure, stop
run_config() { # label conc
  local t0=$SECONDS
  start_sglang_mn || { log "$1: start failed"; return 1; }
  if wait_ready 7200; then
    log "$1: load+ready $((SECONDS - t0)) s"; mem_report; bench "$1" "$2"; mem_report
  fi
  sglang_stop; sleep 10
}

if [ "${BENCH:-1}" = 1 ]; then
  run_config "pro6000-sglang-tp4-ib-spec" 1,2,4,8
  SG_RDMA=0 run_config "pro6000-sglang-tp4-tcp-spec" 1,2,4
  SG_SPEC="" run_config "pro6000-sglang-tp4-ib-nospec" 1,2,4
  # hardware microbenchmarks in the SIF (torch + NCCL): intra-node (2 GPUs over PCIe) and inter-node (4 ranks)
  mapfile -t N < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  M0=$(ip -4 -o addr show dev "$SG_FABRIC_IFACE" | awk '{sub(/\/.*/,"",$4); print $4; exit}')
  hw_run() { # label world nodes... -- env...
    local label=$1 world=$2; shift 2
    local -a nodes=(); while [ "$1" != -- ]; do nodes+=("$1"); shift; done; shift
    local port=$((29900 + RANDOM % 90)) rank=0 n g
    for n in "${nodes[@]}"; do
      for g in 0 1; do
        srun -N1 -n1 -w "$n" --overlap --gres=gpu:2 singularity exec --nv --bind "$B:/bench" \
          --env RANK=$rank --env WORLD_SIZE="$world" --env LOCAL_RANK=$g --env MASTER_ADDR="$M0" --env MASTER_PORT=$port \
          --env NCCL_DEBUG=INFO --env NCCL_DEBUG_SUBSYS=INIT,NET --env NCCL_SOCKET_IFNAME="$SG_FABRIC_IFACE" "$@" \
          "$SG_IMAGE" python3 /bench/hw_bench.py --label "$label" --out "/bench/hw/$label.$SLURM_JOB_ID.json" \
          > "$B/hw/$label.$SLURM_JOB_ID.rank$rank.log" 2>&1 &
        rank=$((rank + 1))
      done
    done
    wait
    grep -h "allreduce_8KiB\|NET/IB\|NET/Socket\|Using network" "$B/hw/$label.$SLURM_JOB_ID".rank*.log | sort -u | head -6
  }
  hw_run pro6000-intra 2 "${N[0]}" --
  hw_run pro6000-inter-ib 4 "${N[@]}" -- --env NCCL_IB_HCA=mlx5_0
  hw_run pro6000-inter-tcp 4 "${N[@]}" -- --env NCCL_IB_DISABLE=1
  log "benchmarks done"
fi
# serve for the rest of the walltime
start_sglang_mn || fail relaunch
wait_ready 7200 || fail ready
relaunches=0
while true; do
  sleep 30
  if ! sg_rank0_alive; then
    relaunches=$((relaunches + 1)); log "rank 0 exited, relaunch $relaunches"
    [ "$relaunches" -le 3 ] || fail serve
    sglang_stop; sleep 10; start_sglang_mn && wait_ready 7200
  fi
done
