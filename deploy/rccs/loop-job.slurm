#!/bin/bash
# =============================================================================
# loop-job.slurm — compute-side shim for research_loop_job cluster venue
# (WP-05 round 2).
#
# Submits the canonical loop wrapper from the repo with the job-private
# venue machinery isolated per job (venue.sh), exactly like serve-model.slurm
# — same site.yaml / wrapper-dir contact, same shared-$HOME guardrails. \
# Prints the node headers the launcher's log parser waits on.
# =============================================================================

#SBATCH --job-name=umrs-loop-job
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --signal=B:TERM@120

echo "=============================="
echo "SLURM JOB ID: ${SLURM_JOB_ID:-}"
echo "Allocated nodes: ${SLURM_NODELIST:-}"
echo "Running on node: $(hostname)"
echo "Fully qualified name: $(hostname -f)"
echo "=============================="

WRAPPER_DIR="${RIVAULT_WRAPPER_DIR:-}"
if [[ -z "${WRAPPER_DIR}" ]]; then
    _self="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || printf '%s' "${BASH_SOURCE[0]}")"
    WRAPPER_DIR="$(cd "$(dirname "${_self}")" && pwd)"
fi
if [[ ! -f "${WRAPPER_DIR}/site.yaml" ]]; then
    echo "[loop-job] FATAL: no site.yaml in ${WRAPPER_DIR} — export RIVAULT_WRAPPER_DIR=<repo>/umrs/deploy/<cluster>"
    exit 1
fi
export UMRS_SITE_CONFIG="${UMRS_SITE_CONFIG:-${WRAPPER_DIR}/site.yaml}"
R4_DIR="${R4_DIR:-$(cd "${WRAPPER_DIR}/../../src/umrs" 2>/dev/null && pwd)}"
export R4_DIR
if [[ -z "${R4_DIR}" || ! -d "${R4_DIR}/knowledge" ]]; then
    echo "[loop-job] FATAL: umrs source tree not under ${R4_DIR} (set R4_DIR=<repo-root>/src/umrs)"
    exit 1
fi

# The job runs as the real user but inherits the submitter's environment.
_real_home="$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6)"
if [[ -n "${_real_home}" && "${_real_home}" != "${HOME:-}" ]]; then
    echo "[loop-job] HOME ${HOME:-<unset>} -> ${_real_home}"
    export HOME="${_real_home}"
fi
USER="$(id -un)"
export USER

echo "[loop-job] site config: ${UMRS_SITE_CONFIG}"
echo "[loop-job] umrs source: ${R4_DIR}"
cd "${R4_DIR}" || exit 1
exec bash --norc --noprofile "${R4_DIR}/knowledge/_cluster_loop_job.sh" "$@"
