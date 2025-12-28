#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME=${CLUSTER_NAME:-umrs-k8s}
REGISTRY_NAME=${REGISTRY_NAME:-kind-registry}
REGISTRY_PORT=${REGISTRY_PORT:-5001}
KIND_CONFIG=${KIND_CONFIG:-$(dirname "$0")/kind-config.yaml}

info() {
  echo "[kind-setup] $*"
}

if ! command -v kind >/dev/null 2>&1; then
  echo "kind is not installed. Please install kind before running this script." >&2
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is not installed. Please install kubectl before running this script." >&2
  exit 1
fi

if ! docker inspect -f '{{.State.Running}}' ${REGISTRY_NAME} >/dev/null 2>&1; then
  info "Starting local registry ${REGISTRY_NAME} on port ${REGISTRY_PORT}"
  docker run -d --restart=always -p "127.0.0.1:${REGISTRY_PORT}:5000" --name "${REGISTRY_NAME}" registry:2
else
  info "Local registry ${REGISTRY_NAME} already running"
fi

if ! kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  info "Creating kind cluster ${CLUSTER_NAME}"
  kind create cluster --name "${CLUSTER_NAME}" --config "${KIND_CONFIG}"
else
  info "Kind cluster ${CLUSTER_NAME} already exists"
fi

REGISTRY_CONTAINER_ID=$(docker inspect -f '{{.Id}}' ${REGISTRY_NAME})
KIND_NETWORK="kind"
if [ -n "${REGISTRY_CONTAINER_ID}" ]; then
  if ! docker network inspect ${KIND_NETWORK} | grep -q ${REGISTRY_NAME}; then
    info "Connecting registry to kind network"
    docker network connect ${KIND_NETWORK} ${REGISTRY_NAME} || true
  fi
fi

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: local-registry-hosting
  namespace: kube-public
data:
  localRegistryHosting.v1: |
    host: "localhost:${REGISTRY_PORT}"
    help: "Push images with 'docker push localhost:${REGISTRY_PORT}/IMAGE_NAME'"
EOF

info "Kind cluster ${CLUSTER_NAME} is ready to use."
