#!/bin/bash
#SBATCH -J dsv41-hw-m
#SBATCH -N 2
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -o /home/users/masaru.nagaso/tmp/dsv41/logs/hw-m-%j.log
# GB10 pair microbenchmarks: per-GPU memory/GEMM, NCCL all_reduce over RoCE (QSFP) and over TCP (enP7s7), raw RDMA.
# sbatch -p ng-dgx-m2 --export=ALL,IMG=localhost/tensorfold-dsv41:exl3-tp2-8a98ae503d85 m_hw.sbatch
set -uo pipefail
: "${IMG:?}"
D=$HOME/tmp/dsv41; OUT=$D/bench/hw; mkdir -p "$OUT"
ARCH=/lvs0/rccs-dmp/masaru.nagaso/umrs_models/images/tensorfold-dsv41-${IMG##*:}-aarch64.tar
mapfile -t N < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
FAB=enP2p1s0f1np1; MGMT=enP7s7
on() { local n=$1; shift; srun -N1 -n1 -w "$n" --overlap "$@"; }
# shellcheck disable=SC2016
PREP='R=/var/tmp/tfbuild-$USER; RR=$XDG_RUNTIME_DIR/tfbuild; mkdir -p $R $RR; rm -rf $R/db.sql $R/libpod $R/overlay-containers; P="podman --root $R --runroot $RR --storage-driver overlay"; $P image exists '"$IMG"' || $P load -q -i '"$ARCH"' >/dev/null; echo "$(hostname) image ok"'
for n in "${N[@]}"; do on "$n" bash -c "$PREP" & done; wait
ipof() { on "$1" ip -4 -o addr show dev "$2" | awk '{sub(/\/.*/,"",$4); print $4; exit}'; }
gidof() { # node ip -> RoCE v2 gid index on roceP2p1s0f1
  on "$1" bash -c 'want=$(printf "0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x" $(echo '"$2"' | tr . " ")); for g in /sys/class/infiniband/roceP2p1s0f1/ports/1/gids/*; do i=${g##*/}; [ "$(cat $g 2>/dev/null)" = "$want" ] || continue; case "$(cat /sys/class/infiniband/roceP2p1s0f1/ports/1/gid_attrs/types/$i 2>/dev/null)" in *v2*) echo $i; exit;; esac; done'
}
M0=$(ipof "${N[0]}" $MGMT); F0=$(ipof "${N[0]}" $FAB); F1=$(ipof "${N[1]}" $FAB)
G0=$(gidof "${N[0]}" "$F0"); G1=$(gidof "${N[1]}" "$F1")
echo "nodes ${N[*]} mgmt0=$M0 fab=$F0/$F1 gid=$G0/$G1"
# raw RDMA (host perftest, if installed)
if on "${N[0]}" which ib_write_bw >/dev/null 2>&1; then
  for t in ib_write_bw ib_write_lat; do
    on "${N[0]}" timeout 60 $t -d roceP2p1s0f1 -x "$G0" -F --report_gbits -a > "$OUT/${t}-server.$SLURM_JOB_ID.txt" 2>&1 &
    sleep 3
    on "${N[1]}" timeout 60 $t -d roceP2p1s0f1 -x "$G1" -F --report_gbits -a "$M0" > "$OUT/${t}.$SLURM_JOB_ID.txt" 2>&1
    wait
  done
fi
run_pair() { # label  extra-env...
  local label=$1; shift
  local port=$((29700 + RANDOM % 200)) r
  for r in 0 1; do
    on "${N[$r]}" bash -c 'R=/var/tmp/tfbuild-$USER; RR=$XDG_RUNTIME_DIR/tfbuild; exec podman --root $R --runroot $RR --storage-driver overlay run --rm --network host --ipc host --userns host --ulimit memlock=-1:-1 --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-caps/nvidia-cap1 --device /dev/nvidia-caps/nvidia-cap2 --device /dev/infiniband -v /usr/lib/aarch64-linux-gnu/libcuda.so.1:/usr/lib/aarch64-linux-gnu/libcuda.so.1:ro -v /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:ro -v /usr/lib/aarch64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro -v '"$D"'/bench:/bench -e RANK='"$r"' -e WORLD_SIZE=2 -e LOCAL_RANK=0 -e MASTER_ADDR='"$M0"' -e MASTER_PORT='"$port"' -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET "$@" --entrypoint python3 '"$IMG"' /bench/hw_bench.py --label '"$label"' --out /bench/hw/'"$label"'.'"$SLURM_JOB_ID"'.json' _ "$@" > "$OUT/$label.$SLURM_JOB_ID.rank$r.log" 2>&1 &
  done
  wait
  grep -h "NET/\|Using network\|allreduce_8KiB" "$OUT/$label.$SLURM_JOB_ID".rank*.log | sort -u | head -8
}
run_pair gb10-pair-roce -e NCCL_SOCKET_IFNAME=$MGMT -e NCCL_IB_HCA=roceP2p1s0f1 -e "NCCL_IB_GID_INDEX=$G0" -e NCCL_IB_DISABLE=0
run_pair gb10-pair-tcp10g -e NCCL_SOCKET_IFNAME=$MGMT -e NCCL_IB_DISABLE=1
echo "$(date -Is) done"
