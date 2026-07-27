#!/usr/bin/env bash
# Shim → ../_common/ssh_tunnel.sh
#
# Behaviour is shared across clusters; this cluster's values live in
# ./cluster.env. Operator and Ansible entry points are unchanged:
#   cd deploy/ai4s && ./ssh_tunnel.sh
#
# See ADR-019 (deployment tree consolidation).
set -euo pipefail
UMRS_CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UMRS_CLUSTER_DIR
exec "${UMRS_CLUSTER_DIR}/../_common/ssh_tunnel.sh" "$@"
