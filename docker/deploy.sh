#!/usr/bin/env bash
set -euo pipefail

# Simple KinD + Kustomize deploy for UMRS
# - Builds local Docker images
# - Loads images into KinD (if cluster exists)
# - Applies k8s resources via kustomize

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
REPOROOT="$(cd "$WORKDIR/.." && pwd)"
K8S_DIR="$WORKDIR/k8s"
CLUSTER_NAME="umrs-k8s"

info() { echo "[deploy] $*"; }

# Build images used by k8s manifests (tags must match)
info "Building images..."
docker build -t docker-umrs-server -f "$WORKDIR/Dockerfile" "$REPOROOT"
docker build -t docker-rivault-mock -f "$WORKDIR/Dockerfile.rivault" "$REPOROOT"

docker build -t docker-umrs-api-test -f "$WORKDIR/Dockerfile.umrs-api-test" "$REPOROOT" || true

# Load into KinD if cluster exists
if command -v kind >/dev/null 2>&1 && kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
  info "Loading images into KinD cluster ${CLUSTER_NAME}..."
  kind load docker-image docker-umrs-server --name "$CLUSTER_NAME"
  kind load docker-image docker-rivault-mock --name "$CLUSTER_NAME"
  kind load docker-image docker-umrs-api-test --name "$CLUSTER_NAME" || true
else
  info "KinD cluster ${CLUSTER_NAME} not found; skipping image load."
fi

# Apply kustomize
info "Applying kustomization from $K8S_DIR..."
kubectl apply -k "$K8S_DIR"

info "Done. Use: kubectl get pods -n umrs-system"
