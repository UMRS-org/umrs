#!/bin/bash
#SBATCH -J dsv41-nvfp4-dl
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH -t 03:00:00
#SBATCH -o /home/users/masaru.nagaso/tmp/dsv41/logs/nvfp4-dl-%j.log
# nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 @3ae9c0d3 into the home file system (/hs/work0; the /lvs0 project quota is
# full). Compute nodes download ~8x faster than the login node (Engram: 203 GB in ~6 min a node).
set -uo pipefail
H=$HOME/umrs_models_hs
for try in 1 2 3 4 5; do
  echo "$(date -Is) try $try on $(hostname -s)"
  if "$HOME/venvs/hf-dl-arm/bin/hf" download nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 --revision 3ae9c0d3fbbea5370bd06e679d22044db15883f1 \
      --cache-dir "$H/hub" --max-workers 16 > "$H/_dl_logs/nvfp4-node.log" 2>&1; then
    echo "$(date -Is) NVFP4 DONE $(du -sLh "$H/hub/models--nvidia--DeepSeek-V4.1-Flash-1017-NVFP4/snapshots/3ae9c0d3fbbea5370bd06e679d22044db15883f1" | cut -f1)"; exit 0
  fi
  tail -2 "$H/_dl_logs/nvfp4-node.log"; sleep 30
done
echo "$(date -Is) NVFP4 FAILED"
