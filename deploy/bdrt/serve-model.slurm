#!/bin/bash
# =============================================================================
# serve-model.slurm — compute-side deploy shim (identical in every cluster dir)
# =============================================================================
# sbatch submits THIS file (so its #SBATCH directives apply). It prints the
# node header the launcher parses, points the wrapper at this cluster's
# site.yaml and execs the canonical wrapper. Everything else — partition
# profiles, job-private podman configuration, private-network allocation,
# image staging, GPU override, cleanup — lives in
# src/umrs/mini-rivault-llm/serve-model.slurm + venue.sh (ADR-045), so no
# per-cluster copy can drift again (ADR-019). The previous 580-line shims
# rewrote the shared-$HOME storage.conf and wiped every CNI network config
# and container of the user at each start; both broke concurrent jobs.
#
# Inputs (exported by the launcher's `sbatch --export`):
#   RIVAULT_WRAPPER_DIR   directory of this shim on the shared filesystem
#                         (sbatch runs a spool copy, so $0 is useless here)
#   UMRS_SITE_CONFIG      optional; defaults to <RIVAULT_WRAPPER_DIR>/site.yaml
#   RIVAULT_DIR           optional; defaults to <repo>/src/umrs/mini-rivault-llm
# The launcher adds --partition/--time/--output/--error on the command line;
# command-line options override the directives below.
# =============================================================================

#SBATCH --job-name=umrs-rivault
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --signal=B:TERM@120

# Node header — rivault_launcher.extract_node_address() waits for this line.
echo "=============================="
echo "SLURM JOB ID: ${SLURM_JOB_ID:-}"
echo "Allocated nodes: ${SLURM_NODELIST:-}"
echo "Running on node: $(hostname)"
echo "Fully qualified name: $(hostname -f)"
echo "=============================="

WRAPPER_DIR="${RIVAULT_WRAPPER_DIR:-}"
if [[ -z "${WRAPPER_DIR}" ]]; then
    # Manual submission from the cluster directory itself.
    _self="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || printf '%s' "${BASH_SOURCE[0]}")"
    WRAPPER_DIR="$(cd "$(dirname "${_self}")" && pwd)"
fi
if [[ ! -f "${WRAPPER_DIR}/site.yaml" ]]; then
    echo "[shim] FATAL: no site.yaml in ${WRAPPER_DIR} — submit through the launcher or export RIVAULT_WRAPPER_DIR=<repo>/umrs/deploy/<cluster>"
    exit 1
fi
export UMRS_SITE_CONFIG="${UMRS_SITE_CONFIG:-${WRAPPER_DIR}/site.yaml}"
RIVAULT_DIR="${RIVAULT_DIR:-$(cd "${WRAPPER_DIR}/../../src/umrs/mini-rivault-llm" 2>/dev/null && pwd)}"
if [[ -z "${RIVAULT_DIR}" || ! -f "${RIVAULT_DIR}/serve-model.slurm" ]]; then
    echo "[shim] FATAL: canonical wrapper not found under ${WRAPPER_DIR}/../../src/umrs/mini-rivault-llm (set RIVAULT_DIR)"
    exit 1
fi

# The job runs as the real user but inherits the submitter's environment: the
# umrs-server container exports HOME=/root and an empty or foreign USER.
_real_home="$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6)"
if [[ -n "${_real_home}" && "${_real_home}" != "${HOME:-}" ]]; then
    echo "[shim] HOME ${HOME:-<unset>} -> ${_real_home}"
    export HOME="${_real_home}"
fi
USER="$(id -un)"
export USER

# LiteLLM master key: the launcher passes it; a manual submission falls back to
# the cluster's .env next to this shim.
if [[ -z "${LITELLM_MASTER_KEY:-}" && -f "${WRAPPER_DIR}/.env" ]]; then
    _key="$(awk -F= '$1 == "LITELLM_MASTER_KEY" {sub(/^[^=]*=/, ""); print; exit}' "${WRAPPER_DIR}/.env")"
    if [[ -n "${_key}" ]]; then
        export LITELLM_MASTER_KEY="${_key}"
    fi
    unset _key
fi

echo "[shim] site config: ${UMRS_SITE_CONFIG}"
echo "[shim] wrapper: ${RIVAULT_DIR}/serve-model.slurm"
cd "${RIVAULT_DIR}" || exit 1
# --norc --noprofile: a login rc must not reset PATH or podman env mid-job.
exec bash --norc --noprofile ./serve-model.slurm "$@"
