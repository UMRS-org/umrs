#!/bin/bash
#SBATCH -J tf-dsv41-build
#SBATCH -p ng-dgx-m2
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -o /home/users/masaru.nagaso/tmp/dsv41/logs/build-%j.log
# Build the MiaAI recipe's TensorFold image (recipe c7f2b851, upstream zig/Dockerfile, sm_121) in a throwaway rootless
# podman store on node-local NVMe, then save it to the rccs image archive dir. Never touches the shared node store.
set -euo pipefail
RECIPE=$HOME/tmp/dsv41/recipe
ARCH_DIR=/lvs0/rccs-dmp/masaru.nagaso/umrs_models/images
ROOT=/var/tmp/tfbuild-$USER
RUN=$XDG_RUNTIME_DIR/tfbuild
P=(podman --root "$ROOT" --runroot "$RUN" --storage-driver overlay)
TF_REPO=https://github.com/bertholomus/TensorFold.git
TF_REF=fbbc97132cd004237d9887b8d3185896efaedc5c
INPUTS_REF=59402086381a42c74d53053443224eec2790b4c5
ZIG_VERSION=0.17.0
ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
ZIG_MIRRORS=https://zig.linus.dev/zig
BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
cd "$RECIPE"
[ "$(git rev-parse HEAD)" = c7f2b85100f0a7b099c92a4e21825e30d93c9c51 ] || { echo "recipe not at c7f2b851"; exit 1; }
HASH=$( { cat zig/patches/*.patch 2>/dev/null; cat zig/patches-inputs/*.patch 2>/dev/null; cat zig/Dockerfile;
          echo "$TF_REPO $TF_REF $INPUTS_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE"; } | sha256sum | cut -c1-12)
TAG=exl3-tp2-$HASH
IMG=localhost/tensorfold-dsv41:$TAG
echo "$(date -Is) host=$(hostname) image=$IMG"
mkdir -p "$ROOT" "$RUN"
CTX=$(mktemp -d /var/tmp/tfctx.XXXXXX)
mkdir -p "$CTX/zig"; cp -r zig/patches zig/patches-inputs "$CTX/zig/"; cp zig/Dockerfile "$CTX/Dockerfile"
"${P[@]}" build --network host -t "$IMG" -f "$CTX/Dockerfile" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg TF_REPO="$TF_REPO" --build-arg TF_REF="$TF_REF" \
  --build-arg INPUTS_REF="$INPUTS_REF" --build-arg ZIG_VERSION="$ZIG_VERSION" --build-arg ZIG_SHA256="$ZIG_SHA256" \
  --build-arg ZIG_MIRRORS="$ZIG_MIRRORS" --build-arg PATCHES_HASH="$HASH" "$CTX"
rm -rf "$CTX"
"${P[@]}" run --rm --network none --entrypoint tensorfold-native "$IMG" capabilities --json | head -c 2000; echo
OUT=$ARCH_DIR/tensorfold-dsv41-$TAG-aarch64.tar
"${P[@]}" save -o "$OUT.part" "$IMG" && mv "$OUT.part" "$OUT"
"${P[@]}" image inspect -f '{{.Id}}' "$IMG" > "$OUT.id"
sha256sum "$OUT" > "$OUT.sha256"
ln -sf "$(basename "$OUT")" "${OUT%-aarch64.tar}-arm64.tar"   # venue.sh rivault_arch_suffix names aarch64 archives -arm64
echo "$(date -Is) SAVED $OUT ($(du -h "$OUT" | cut -f1)) id=$(cat "$OUT.id")"
