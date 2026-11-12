#!/bin/bash
# ai-pro6000 job entry (read when the queued job starts). Stage 1: probe both nodes and keep the record.
set -uo pipefail
D=$HOME/tmp/dsv41
mkdir -p "$D/bench/hw"
# keep-alive stopgap until the roster service block is converged: queue the next 23 h job (singleton, serve only)
if [ -e "$D/run/pro6000.FAILED" ]; then echo "previous run failed ($(cat "$D/run/pro6000.FAILED")); not chaining, releasing the nodes"; exit 1; fi
sbatch --dependency=singleton -J dsv41-pro6000 --export=ALL,BENCH=0 "$D/pro6000_hold.sbatch" | sed 's/^/[keep] /'
echo "$(date -Is) job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST"
srun --ntasks-per-node=1 --overlap bash -c '
echo "== $(hostname)"; nvidia-smi --query-gpu=index,name,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader
nvidia-smi topo -m 2>&1 | head -12
ip -4 -o addr | awk "{print \$2, \$4}"
for d in /sys/class/infiniband/*; do [ -e "$d" ] && echo "$(basename $d) $(cat $d/ports/1/state) $(cat $d/ports/1/rate) $(cat $d/ports/1/link_layer)"; done
ls /etc/cdi /var/run/cdi 2>&1; podman --version; df -h / /var/tmp | tail -2; free -g | head -2; ulimit -Hl
' 2>&1 | tee "$D/bench/hw/pro6000-probe.$SLURM_JOB_ID.txt"
if [ -x "$D/pro6000_stage2.sh" ]; then exec "$D/pro6000_stage2.sh"; fi
echo "$(date -Is) no stage 2; releasing the nodes"
