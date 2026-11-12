#!/bin/bash
#SBATCH -J dsv41-stage-local
#SBATCH -N 2
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 06:00:00
#SBATCH -o /home/users/masaru.nagaso/tmp/dsv41/logs/stage-local-%j.log
# Node-local staging for DeepSeek-V4.1-Flash-1017 (TensorFold recipe) on one ng-dgx-m pair: on EVERY node of the job,
# Engram's tables (deepseek-ai @1bae0e10 shards 47/48, ~203 GB) and both ranks' weight files (~106 GB each, so either
# node can be the head) under /var/tmp/umrs-tf. The rccs Lustre project quota is full, and local NVMe is faster.
# sbatch -p ng-dgx-m2 --export=ALL,IMG=localhost/tensorfold-dsv41:exl3-tp2-8a98ae503d85 dsv41_stage_local.sbatch
set -uo pipefail
: "${IMG:?}"
D=$HOME/tmp/dsv41
HF=/lvs0/rccs-dmp/masaru.nagaso/umrs_models
SNAP_SUB=hub/models--Mia-AiLab--DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw/snapshots/45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b
DATA_SUB=tensorfold-dsv41/45ee5d0f
VENV=$HOME/venvs/hf-dl-arm
log() { echo "$(date -Is) $*"; }
# the served folder on Lustre (links + the recipe's chat template; a few KB)
VIEW=$HF/$DATA_SUB/model
if [ ! -s "$VIEW/chat_template.jinja" ]; then
  rm -rf "$VIEW.new" && mkdir -p "$VIEW.new"
  for f in "$HF/$SNAP_SUB"/*; do ln -s "$(realpath -s --relative-to="$VIEW.new" "$f")" "$VIEW.new/"; done
  rm -f "$VIEW.new/chat_template.jinja"; cp "$D/recipe/model/chat_template.jinja" "$VIEW.new/chat_template.jinja"
  rm -rf "$VIEW" && mv "$VIEW.new" "$VIEW"; log "served folder ready"
fi
[ -s "$VIEW/config.json" ] || { log "served folder broken"; exit 1; }
if [ ! -x "$VENV/bin/hf" ]; then
  srun -N1 -n1 -w "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)" --overlap bash -c \
    "python3 -m venv $VENV && $VENV/bin/pip install -q 'huggingface_hub[hf_xet]' && $VENV/bin/python -c 'import hf_xet'" || { log "venv failed"; exit 1; }
fi
# shellcheck disable=SC2016
NODE_SCRIPT='
set -uo pipefail
L=/var/tmp/umrs-tf; E=$L/engram-1bae0e10; C=$L/rank-cache-45ee5d0f
mkdir -p "$E" "$C"
log() { echo "$(date -Is) $(hostname -s) $*"; }
log "free: $(df -h /var/tmp | tail -1)"
( for try in 1 2 3 4 5; do
    "$VENV/bin/hf" download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
      --local-dir "$E" --include model.safetensors.index.json --include config.json \
      --include model-00047-of-00048.safetensors --include model-00048-of-00048.safetensors >/dev/null 2>"$L/engram-dl.err" && break
    log "engram download try $try failed: $(tail -1 $L/engram-dl.err)"; sleep 30
  done
  s47=$(stat -c %s "$E/model-00047-of-00048.safetensors" 2>/dev/null); s48=$(stat -c %s "$E/model-00048-of-00048.safetensors" 2>/dev/null)
  if [ "$s47" = 101535150936 ] && [ "$s48" = 101537926640 ]; then log "ENGRAM OK"; else log "ENGRAM BAD sizes $s47 $s48"; fi ) &
X=$L/exl3-45ee5d0f
for try in 1 2 3; do
  "$VENV/bin/hf" download Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw --revision 45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b \
    --local-dir "$X" >/dev/null 2>"$L/exl3-dl.err" && break
  log "exl3 download try $try failed: $(tail -1 $L/exl3-dl.err)"; sleep 30
done
log "EXL3 local copy $(du -sh $X | cut -f1)"
R=/var/tmp/tfbuild-$USER; RR=$XDG_RUNTIME_DIR/tfbuild; mkdir -p "$R" "$RR"; rm -rf "$R/db.sql" "$R/libpod" "$R/overlay-containers"
P=(podman --root "$R" --runroot "$RR" --storage-driver overlay)
"${P[@]}" image exists "$IMG" || "${P[@]}" load -q -i "$ARCH" >/dev/null || { log "image load failed"; exit 1; }
for r in 0 1; do
  want=$("${P[@]}" run --rm --network none -e HOME=/tmp -v "$X:/ckpt:ro" --entrypoint bash "$IMG" -c "tf-dsv41-rank-cache /ckpt /tmp/x $r 2 --index-only | sha256sum | cut -c1-20")
  if [ -f "$C/rank${r}of2-$want.bin" ]; then log "rank $r weight file ready"; continue; fi
  rm -f "$C/rank${r}of2-"*.bin "$C/.rank${r}of2.partial"
  log "rank $r weight file"
  "${P[@]}" run --rm --network none -e HOME=/tmp -v "$X:/ckpt:ro" -v "$C:/out" --entrypoint tf-dsv41-rank-cache "$IMG" /ckpt /out "$r" 2 | tail -2
  [ -f "$C/rank${r}of2-$want.bin" ] && log "RANK $r OK $(du -h $C/rank${r}of2-$want.bin | cut -f1)" || log "RANK $r FAILED"
done
wait
log "done: $(du -sh $L | cut -f1)"
'
export IMG HF SNAP_SUB VENV ARCH=$HF/images/tensorfold-dsv41-${IMG##*:}-aarch64.tar
srun --ntasks-per-node=1 --overlap bash -c "$NODE_SCRIPT"
log "stage-local finished"
