#!/bin/bash
# Submit GLM-5.2-NVFP4 as SIX independent 1-node sbatch jobs (PP=6 over RJ45).
# Head+w1 on ng-dgx-m1, w2+w3 on ng-dgx-m2, w4+w5 on ng-dgx-m0.
# PP=4 does NOT fit: 465GB/4 = ~116GB weights/node vs ~95GB free on the head
# (job 208909: "Free memory on device cuda:0 (95.56/121.63 GiB) ... less than
# desired"). PP=6 → ~78GB/node. See glm52-nvfp4-node-sif.slurm for the full
# srun/podman/singularity saga.
#
# Usage: ./submit-glm52-nvfp4.sh [run-id]
set -eu
cd "$(dirname "$0")"
RUN_ID="${1:-$(date +%s)}"
SCRIPT=glm52-nvfp4-node-sif.slurm
TOTAL=6

echo "run-id: ${RUN_ID}"
HEAD_JOB=$(sbatch --parsable --partition=ng-dgx-m1 --job-name=glm52nvfp4-head \
    --export=ALL,GLM_ROLE=head,GLM_RUN_ID="${RUN_ID}",GLM_TOTAL_GPUS=${TOTAL} "${SCRIPT}")
echo "head:    job ${HEAD_JOB} (ng-dgx-m1)"
_w=1
for part in ng-dgx-m1 ng-dgx-m2 ng-dgx-m2 ng-dgx-m0 ng-dgx-m0; do
    W=$(sbatch --parsable --partition="${part}" --job-name="glm52nvfp4-w${_w}" \
        --export=ALL,GLM_ROLE=worker,GLM_RUN_ID="${RUN_ID}" "${SCRIPT}")
    echo "worker${_w}: job ${W} (${part})"
    _w=$((_w+1))
done
echo
echo "watch:   tail -f /lvs0/rccs-dmp/masaru.nagaso/umrs_models/logs/glm52-nvfp4-glm52nvfp4-head-${HEAD_JOB}.out"
echo "teardown: scancel \$(squeue -u \$USER -h -o %i -n glm52nvfp4-head,glm52nvfp4-w1,glm52nvfp4-w2,glm52nvfp4-w3,glm52nvfp4-w4,glm52nvfp4-w5 | tr \"\\n\" \" \")"
