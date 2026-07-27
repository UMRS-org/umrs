#!/usr/bin/env bash
# =============================================================================
# Shared cluster context resolver
# =============================================================================
# Sourced by every script in deploy/_common/. Establishes the cluster directory
# the caller is operating on and loads that cluster's parameters.
#
# The scripts in this directory are never invoked directly. Each cluster
# directory (deploy/rccs, deploy/ai4s, deploy/bdrt, deploy/octopus) contains a
# thin shim per script that exports UMRS_CLUSTER_DIR and execs the shared
# implementation. This keeps the operator- and Ansible-facing paths unchanged
# (`cd deploy/<cluster> && ./start.sh`) while the behaviour lives in one place.
#
# Rationale: these scripts previously existed as near-identical copies per
# cluster. The copies drifted — octopus pointed its SSH tunnel at bdrt's login
# node, octopus wrote a "bdrt" client profile, and only rccs ever received the
# r4-contracts install fix. See ADR-019.
# =============================================================================

# UMRS_CLUSTER_DIR is exported by the shim. Refuse to guess: running these
# scripts against the wrong directory would target the wrong cluster's .env,
# venv and PID files.
if [[ -z "${UMRS_CLUSTER_DIR:-}" ]]; then
    echo "ERROR: $(basename "${BASH_SOURCE[1]:-$0}") must be invoked through a cluster shim." >&2
    echo "  Run:  cd deploy/<cluster> && ./$(basename "${BASH_SOURCE[1]:-$0}")" >&2
    exit 1
fi

if [[ ! -d "$UMRS_CLUSTER_DIR" ]]; then
    echo "ERROR: UMRS_CLUSTER_DIR does not exist: ${UMRS_CLUSTER_DIR}" >&2
    exit 1
fi

UMRS_CLUSTER_DIR="$(cd "$UMRS_CLUSTER_DIR" && pwd)"
export UMRS_CLUSTER_DIR

# Per-cluster parameters. Every value has a default matching the historical
# rccs behaviour, so a cluster.env that omits a key keeps working.
if [[ -f "${UMRS_CLUSTER_DIR}/cluster.env" ]]; then
    # shellcheck disable=SC1091  # path is resolved at runtime
    source "${UMRS_CLUSTER_DIR}/cluster.env"
fi

# Cluster identity — defaults to the directory name.
UMRS_CLUSTER="${UMRS_CLUSTER:-$(basename "$UMRS_CLUSTER_DIR")}"

# Root under which model weights are staged on this cluster's filesystem.
# Only a fallback: site.yaml's storage.model_cache_dir wins when set.
UMRS_MODEL_ROOT="${UMRS_MODEL_ROOT:-/lvs0/dne1/rccs-dmp/${USER}/umrs_models}"

# SSH alias for this cluster's login node, used in operator hints and by
# ssh_tunnel.sh.
UMRS_SSH_ALIAS="${UMRS_SSH_ALIAS:-${UMRS_CLUSTER}}"

# Local port the Platform Gateway is published on when tunnelling.
UMRS_GATEWAY_PORT="${UMRS_GATEWAY_PORT:-8080}"

# Connection profile name written by setup_client.sh.
UMRS_CLIENT_PROFILE="${UMRS_CLIENT_PROFILE:-${UMRS_CLUSTER}}"

# Expected `name:` in this cluster's site.yaml. health_check_server.sh asserts
# on it, so a wrong value silently fails the site-config check rather than
# erroring loudly — which is exactly what happened on bdrt and octopus, whose
# copied health checks asserted "rccs-cloud".
UMRS_SITE_NAME="${UMRS_SITE_NAME:-${UMRS_CLUSTER}}"

# Slurm partitions probed by harness_probe.sh. Newline- or space-separated.
UMRS_PARTITIONS="${UMRS_PARTITIONS:-}"

export UMRS_CLUSTER UMRS_MODEL_ROOT UMRS_SSH_ALIAS UMRS_GATEWAY_PORT \
    UMRS_CLIENT_PROFILE UMRS_SITE_NAME UMRS_PARTITIONS
