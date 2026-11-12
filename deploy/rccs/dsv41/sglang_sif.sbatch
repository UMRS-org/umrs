#!/bin/bash
#SBATCH -J dsv41-sglang-sif
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH -o /home/users/masaru.nagaso/tmp/dsv41/logs/sglang-sif-%j.log
# The digest-pinned SGLang image as a Singularity SIF for ai-pro6000, whose nodes have no podman (job 468124,
# 2026-11-12). Built on an aarch64 GB10 node; node-local temp space; the SIF lives in home (/hs/work0).
set -euo pipefail
REF=docker://lmsysorg/sglang@sha256:bc4aab534886dfc9e2c1d7425eca14e1f9d03a518a29074928c4f385649d9844
OUT=$HOME/umrs_models_hs/images/sglang-314bce7c-arm64.sif
mkdir -p "$(dirname "$OUT")"
export SINGULARITY_TMPDIR=/var/tmp/sif-tmp-$USER SINGULARITY_CACHEDIR=/var/tmp/sif-cache-$USER
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"
echo "$(date -Is) $(hostname -s) pulling $REF"
singularity pull --arch arm64 "$OUT.part" "$REF"
mv "$OUT.part" "$OUT"
sha256sum "$OUT" > "$OUT.sha256"
singularity exec "$OUT" python3 -c 'import sglang, torch; print("sglang", sglang.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'
echo "$(date -Is) SIF DONE $(du -h "$OUT" | cut -f1)"
rm -rf "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"
