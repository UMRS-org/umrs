#!/bin/bash
#SBATCH -J dsv41-tf-direct
#SBATCH -N 2
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 23:00:00
#SBATCH -o /home/users/masaru.nagaso/tmp/dsv41/logs/tf-direct-%j.log
# First-launch receipt + benchmark target for DeepSeek-V4.1-Flash-1017 (MiaAI TensorFold recipe) on one ng-dgx-m pair,
# driven by the branch's tensorfold_runner.sh directly (no UMRS stack, no LiteLLM): the same adapter code the roster
# entry deepseek-v4.1-flash-1017 runs. Submit: sbatch -p ng-dgx-m2 --export=ALL,IMG=<tag>[,TF_DRAFTS=1,...] tf_direct.sbatch
set -uo pipefail
: "${IMG:?set IMG}"
D=$HOME/tmp/dsv41
MR=$D/mr                       # copies of src/umrs/mini-rivault-llm/{tensorfold_runner.sh,venue.sh,gb10_reclaim.py}
export RIVAULT_WRAPPER_DIR=$D/run
mkdir -p "$RIVAULT_WRAPPER_DIR/jobs"
HF=/lvs0/rccs-dmp/masaru.nagaso/umrs_models
export TF_IMAGE=$IMG
export RIVAULT_IMAGE_ARCHIVE_DIR=$HF/images
export TF_HF_CACHE=$HF
export TF_MODEL_SUB=hub/models--Mia-AiLab--DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw/snapshots/45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b
export TF_DATA_SUB=tensorfold-dsv41/45ee5d0f
export TF_KIT_SUB=tensorfold-dsv41/kit-abe07c7d
export TF_ENGRAM_SUB=hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
export TF_ENGRAM_DIR=/var/tmp/umrs-tf/engram-1bae0e10
export TF_HEAD_CACHE_DIR=/var/tmp/umrs-tf/rank-cache-45ee5d0f TF_WORKER_CACHE_DIR=/var/tmp/umrs-tf/rank-cache-45ee5d0f
export TF_CONTEXT=${TF_CONTEXT:-131072} TF_PARALLEL=${TF_PARALLEL:-4} TF_DRAFTS=${TF_DRAFTS:-1}
export TF_MIN_AVAIL_GIB=${TF_MIN_AVAIL_GIB:-100} TF_MIN_AVAIL_WAIT_S=300 TF_PRESTART_RECLAIM_GIB=100
export TF_FABRIC_IFACE=${TF_FABRIC_IFACE:-enP2p1s0f1np1} TF_MASTER_IFACE=${TF_MASTER_IFACE:-enP7s7}
export TF_RDMA=${TF_RDMA:-1} TF_MEMORY_RESERVE_GIB=${TF_MEMORY_RESERVE_GIB:-11.5}
export TF_WORKER_MODE=step RIVAULT_MN_WORKER_MODE=step RIVAULT_USERNS_MODE=host RCCS_SKIP_CDI=1
export TF_SERVED_NAME=DeepSeek-V4.1-Flash-1017
# KEEP=1: keep-alive chain (stopgap until the roster's service block is converged; stop it with
# `scancel -n dsv41-tf-direct`). LEAD_S before the walltime, the successor goes to the staged pair
# ($D/run/staged-pairs) with the earliest start: another pair that can start within LEAD_S overlaps this job (no gap),
# else this pair, queued behind this job (singleton; a gap of one load, ~2 min). Maintainer 2026-11-13: "if m2 pair is
# still occupied, we can use another ng-dgx-m*".
LEAD_S=${LEAD_S:-3000}
secs_left() { squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1); if (n>2) s+=3600*$(n-2); if (n>3) s+=86400*$(n-3); print s}'; }
start_epoch() { sbatch --test-only -p "$1" -N 2 --exclusive --mem=0 -t 23:00:00 --wrap=true 2>&1 |
  grep -oP 'to start at \K\S+' | head -1 | xargs -r -I{} date -d {} +%s; }
pick_pair() { # prints "<partition> <start epoch>" of the earliest other staged pair
  local p t best="" bt=9999999999
  while read -r p; do
    [ -n "$p" ] && [ "$p" != "$SLURM_JOB_PARTITION" ] || continue
    t=$(start_epoch "$p"); [ -n "$t" ] || continue
    if (( t < bt )); then bt=$t; best=$p; fi
  done < <(sort -u "$D/run/staged-pairs" 2>/dev/null)
  [ -n "$best" ] && echo "$best $bt"
}
submit_successor() {
  local pick p t args=(-J dsv41-tf-direct "--export=ALL,IMG=$IMG,BENCH=0,ABLATE=0,KEEP=1")
  pick=$(pick_pair); read -r p t <<< "$pick"
  if [ -n "$p" ] && (( t <= $(date +%s) + LEAD_S )); then
    echo "$(date -Is) [keep] successor on $p (starts $(date -d @"$t" +%H:%M), overlapping this job)"
    sbatch -p "$p" "${args[@]}" "$D/tf_direct.sbatch" | sed 's/^/[keep] /'
  else
    echo "$(date -Is) [keep] no other staged pair free in time; successor queued on $SLURM_JOB_PARTITION"
    sbatch --dependency=singleton -p "$SLURM_JOB_PARTITION" "${args[@]}" "$D/tf_direct.sbatch" | sed 's/^/[keep] /'
  fi
}
SUCCESSOR=0
KEYF=$D/tf-api-key
[ -s "$KEYF" ] || ( umask 077; head -c 24 /dev/urandom | base64 | tr -d '/+=' > "$KEYF" )
VLLM_API_KEY=$(cat "$KEYF"); export VLLM_API_KEY
echo "$(date -Is) job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST image=$IMG drafts=$TF_DRAFTS ctx=$TF_CONTEXT rdma=$TF_RDMA"
# shellcheck disable=SC1091
source "$MR/venue.sh"
rivault_venue_prepare "$SLURM_JOB_ID" cgroupfs
export TF_VENUE_SH=$MR/venue.sh
# the engine image into every node's store first, one node at a time (an in-driver archive load once returned
# empty output and no image, job 468248, likely two venue prepares re-pinning the runroot at once)
ARCH=$HF/images/tensorfold-dsv41-${IMG##*:}-aarch64.tar
for n in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
  srun -N1 -n1 -w "$n" --overlap bash -c 'source "$1" && rivault_venue_prepare "$2" cgroupfs >/dev/null 2>&1;
    podman image exists "$3" || podman load -q -i "$4" >/dev/null; podman image exists "$3" && echo "$(hostname -s) image ok"' \
    _ "$MR/venue.sh" "$SLURM_JOB_ID" "$IMG" "$ARCH"
done
# shellcheck disable=SC1091
source "$MR/tensorfold_runner.sh"
trap 'echo "$(date -Is) stopping"; tensorfold_stop; exit 0' TERM INT
start_tensorfold || { echo "$(date -Is) START FAILED"; tensorfold_stop; exit 1; }
echo "http://${HEAD_IP}:${TF_PORT}" > "$D/run/endpoint-tf.$SLURM_JOB_ID"
echo "$(date -Is) endpoint http://${HEAD_IP}:${TF_PORT}"
B=$D/bench; mkdir -p "$B/serve"
min_avail() { local n m low=999; for n in "${RAY_NODES[@]}"; do
  m=$(srun -N1 -n1 -w "$n" --overlap awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo); echo "$(date -Is) $n MemAvailable ${m} GiB" >&2
  (( m < low )) && low=$m; done; echo "$low"; }
# wait for READY (or a dead rank 0) up to $1 s
wait_ready() { local t0=$SECONDS st
  while (( SECONDS - t0 < $1 )); do
    st=$(podman inspect "$TF_R0_NAME" --format '{{.State.Status}}' 2>/dev/null)
    if [ "$st" != running ]; then
      echo "$(date -Is) rank 0 is '${st:-gone}': last log lines"; podman logs --tail 60 "$TF_R0_NAME" 2>&1 | sed 's/^/  r0| /'
      tail -40 "$(tf_log_file r1)" 2>/dev/null | sed 's/^/  r1| /'; return 1; fi
    if curl -s -m 5 -H "Authorization: Bearer $VLLM_API_KEY" "http://${HEAD_IP}:${TF_PORT}/v1/models" | grep -q DeepSeek; then
      echo "$(date -Is) READY after $((SECONDS - t0)) s"; return 0; fi
    sleep 15
  done; echo "$(date -Is) not READY after $1 s"; return 1; }
bench() { # label
  python3 "$B/serve_bench.py" --url "http://${HEAD_IP}:${TF_PORT}" --model "$TF_SERVED_NAME" --key-file "$KEYF" \
    --label "$1" --conc 1,2,4 --out "$B/serve/$1.$SLURM_JOB_ID.json" \
    --meta "{\"platform\":\"rccs ng-dgx-m pair 2x GB10\",\"engine\":\"tensorfold recipe c7f2b851\",\"weights\":\"EXL3 EAQ v1 2.9bpw\",\"tp\":2,\"drafts\":\"$TF_DRAFTS\",\"rdma\":\"$TF_RDMA\",\"ctx\":$TF_CONTEXT,\"nodes\":\"$SLURM_JOB_NODELIST\"}" 2>&1 | tail -30; }
relaunch() { echo "$(date -Is) relaunch: drafts=$TF_DRAFTS rdma=$TF_RDMA"; tf_launch_ranks && wait_ready 5400; }
if wait_ready 5400; then
  LOW=$(min_avail)
  [ "${BENCH:-1}" = 1 ] && bench "gb10-tf-tp2-rdma-drafts${TF_DRAFTS}"
  if [ "${ABLATE:-1}" = 1 ]; then
    TF_RDMA=0; export TF_RDMA
    relaunch && bench "gb10-tf-tp2-tcp10g-drafts${TF_DRAFTS}"
    TF_RDMA=1
    # the drafter costs ~3.5 GiB a rank: only with headroom over memguard's floor plus prefill workspace
    if [ "$TF_DRAFTS" = 0 ] && (( LOW >= 9 )); then
      TF_DRAFTS=1; relaunch && { min_avail >/dev/null; bench "gb10-tf-tp2-rdma-drafts1"; }
      TF_DRAFTS=0
    else echo "$(date -Is) drafts=1 ablation skipped: min MemAvailable ${LOW} GiB after load"; fi
    relaunch
  fi
fi
relaunches=0
while true; do
  sleep 30
  if [ "${KEEP:-0}" = 1 ] && [ "$SUCCESSOR" = 0 ] && (( $(secs_left) <= LEAD_S )); then
    submit_successor; SUCCESSOR=1
  fi
  st=$(podman inspect "$TF_R0_NAME" --format '{{.State.Status}}' 2>/dev/null)
  if [ "$st" != running ]; then
    relaunches=$((relaunches + 1)); echo "$(date -Is) rank 0 '${st:-gone}', relaunch $relaunches"
    [ "$relaunches" -le 3 ] || { tensorfold_stop; exit 1; }
    relaunch
  fi
done
