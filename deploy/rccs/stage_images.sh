#!/usr/bin/env bash
# Shim → ../_common/stage_images.sh
#
# Behaviour is shared across clusters; this cluster's values live in
# ./cluster.env and ./site.yaml. Operator and Ansible entry points:
#   cd deploy/<cluster> && ./stage_images.sh [--if-missing] [--partition P]
#
# See ADR-019 (deployment tree consolidation) and ADR-045 D3 (image staging).
set -euo pipefail
UMRS_CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UMRS_CLUSTER_DIR
exec "${UMRS_CLUSTER_DIR}/../_common/stage_images.sh" "$@"
