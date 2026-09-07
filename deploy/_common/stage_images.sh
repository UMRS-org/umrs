#!/usr/bin/env bash
# =============================================================================
# stage_images.sh — build the site's egress-guard image and save it as an
# archive on the shared filesystem for compute nodes to load (ADR-045 D3)
# =============================================================================
# Compute nodes keep node-local podman stores and usually have no registry
# route for a locally built image, so `localhost/r4-mcp:latest` (the declared
# egress proxy, ADR-040 D4) can only reach them as a `podman save` archive that
# the wrapper's rivault_ensure_image loads on first use.
#
# Usage (login node, through the cluster shim):
#   cd deploy/<cluster> && ./stage_images.sh [options]
#     --image REF          image to build+save  (default: site network.egress_proxy_image)
#     --archive-dir DIR    where archives live   (default: site runtime.image_archive_dir)
#     --partition P        partition whose arch to build for (default: first scheduler.partition)
#     --arch A             target machine arch, e.g. aarch64 (default: partition profile arch,
#                          else the login node's)
#     --build-context DIR  (default: <repo>/.. — the ai4s-platform checkout root;
#                          the r4-mcp Dockerfile installs the unpublished
#                          r4-contracts sibling from there)
#     --dockerfile FILE    (default: <build-context>/mcp-servers/Dockerfile)
#     --if-missing         exit 0 without building when the archive already exists
#     --print-path         print the archive path and exit
#     --local              build on this node even if the target arch differs
#
# When the target arch differs from this node's, the build runs on one compute
# node of the partition through srun (podman there, as for serving jobs).
# Archives are written atomically (<path>.tmp then mv).
# =============================================================================
set -euo pipefail

# shellcheck disable=SC1091  # resolved at runtime through the cluster shim
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cluster-context.sh"

REPO="${UMRS_REPO:-$(cd "${UMRS_CLUSTER_DIR}/../.." && pwd)}"
VENUE_LIB="${REPO}/src/umrs/mini-rivault-llm/venue.sh"
# shellcheck source=/dev/null  # lives in the umrs checkout next to the wrapper
source "${VENUE_LIB}"

IMAGE=""
ARCHIVE_DIR=""
PARTITION=""
TARGET_ARCH=""
BUILD_CONTEXT=""
DOCKERFILE=""
IF_MISSING=0
PRINT_PATH=0
FORCE_LOCAL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --archive-dir) ARCHIVE_DIR="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --arch) TARGET_ARCH="$2"; shift 2 ;;
        --build-context) BUILD_CONTEXT="$2"; shift 2 ;;
        --dockerfile) DOCKERFILE="$2"; shift 2 ;;
        --if-missing) IF_MISSING=1; shift ;;
        --print-path) PRINT_PATH=1; shift ;;
        --local) FORCE_LOCAL=1; shift ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "stage_images: unknown option: $1" >&2; exit 2 ;;
    esac
done

IMAGE="${IMAGE:-$(umrs_site_value network.egress_proxy_image)}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$(umrs_site_value runtime.image_archive_dir)}"
if [[ -z "${PARTITION}" ]]; then
    PARTITION="$(umrs_site_value scheduler.partition)"
    PARTITION="${PARTITION%%,*}"
fi
if [[ -z "${TARGET_ARCH}" && -n "${PARTITION}" ]]; then
    TARGET_ARCH="$(umrs_site_value "partition_profile.${PARTITION}.arch")"
fi
TARGET_ARCH="${TARGET_ARCH:-$(uname -m)}"
BUILD_CONTEXT="${BUILD_CONTEXT:-$(cd "${REPO}/.." && pwd)}"
DOCKERFILE="${DOCKERFILE:-${BUILD_CONTEXT}/mcp-servers/Dockerfile}"

if [[ -z "${IMAGE}" ]]; then
    echo "stage_images: no image — pass --image or declare network.egress_proxy_image in ${UMRS_CLUSTER_DIR}/site.yaml" >&2
    exit 2
fi
if [[ -z "${ARCHIVE_DIR}" ]]; then
    echo "stage_images: no archive dir — pass --archive-dir or declare runtime.image_archive_dir in ${UMRS_CLUSTER_DIR}/site.yaml" >&2
    exit 2
fi

ARCHIVE="$(rivault_image_archive_path "${ARCHIVE_DIR}" "${IMAGE}" "$(rivault_arch_suffix "${TARGET_ARCH}")")"
if [[ "${PRINT_PATH}" -eq 1 ]]; then
    printf '%s\n' "${ARCHIVE}"
    exit 0
fi
if [[ "${IF_MISSING}" -eq 1 && -f "${ARCHIVE}" ]]; then
    echo "stage_images: archive present: ${ARCHIVE} ($(du -h "${ARCHIVE}" | cut -f1)) — nothing to do"
    exit 0
fi
if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "stage_images: Dockerfile not found: ${DOCKERFILE} (is ${REPO} the umrs submodule of an ai4s-platform checkout?)" >&2
    exit 2
fi

# The build runs from a script on the shared filesystem so srun can execute it
# on the compute node; the venue library gives it the same node-local podman
# configuration serving jobs use (the image therefore also lands in that
# node's store as a bonus).
STAGE_DIR="${UMRS_CLUSTER_DIR}/tmp/stage_images_$(date +%Y%m%d%H%M%S)"
mkdir -p "${STAGE_DIR}" "${ARCHIVE_DIR}"
BUILD_SCRIPT="${STAGE_DIR}/build.sh"
cat > "${BUILD_SCRIPT}" <<'BUILDEOF'
#!/bin/bash
set -euo pipefail
# shellcheck source=/dev/null
source "${VENUE_LIB}"
rivault_venue_prepare "stage-${SLURM_JOB_ID:-$$}" cgroupfs
echo "[stage] building ${IMAGE} on $(hostname) ($(uname -m)) from ${DOCKERFILE}"
podman build --pull=missing -t "${IMAGE}" -f "${DOCKERFILE}" "${BUILD_CONTEXT}"
podman save -o "${ARCHIVE}.tmp" "${IMAGE}"
mv -f "${ARCHIVE}.tmp" "${ARCHIVE}"
echo "[stage] archive written: ${ARCHIVE} ($(du -h "${ARCHIVE}" | cut -f1))"
BUILDEOF
chmod +x "${BUILD_SCRIPT}"
export VENUE_LIB IMAGE DOCKERFILE BUILD_CONTEXT ARCHIVE

echo "stage_images: image=${IMAGE} arch=${TARGET_ARCH} archive=${ARCHIVE}"
if [[ "${FORCE_LOCAL}" -eq 1 || "$(rivault_arch_suffix "${TARGET_ARCH}")" == "$(rivault_arch_suffix)" ]]; then
    echo "stage_images: building on this node ($(uname -m))"
    bash "${BUILD_SCRIPT}"
else
    if [[ -z "${PARTITION}" ]]; then
        echo "stage_images: target arch ${TARGET_ARCH} differs from $(uname -m) and no partition to build on — pass --partition" >&2
        exit 2
    fi
    echo "stage_images: building on a ${PARTITION} node via srun (target ${TARGET_ARCH}, login node is $(uname -m))"
    srun --partition="${PARTITION}" --nodes=1 --ntasks=1 \
         --time="${STAGE_IMAGES_TIME:-00:45:00}" --job-name=umrs-stage-images \
         bash "${BUILD_SCRIPT}"
fi
rm -rf "${STAGE_DIR}"
