#!/usr/bin/env bash

# End-to-end RAG demo:
# 1. Run the standard UMRS API test to ingest data and build embeddings.
# 2. Call the UMRS RAG microservice to answer a question over the ingested data.
#
# Environment variables:
#   SKIP_RAG_TESTS: If set to "true", skip the RAG microservice tests (steps 2-3)
#   UMRS_RAG_HOST: Hostname for RAG service (default: umrs-rag)
#   UMRS_RAG_PORT: Port for RAG service (default: 8081)

set -euo pipefail

retry() {
  # retry MAX_ATTEMPTS SLEEP_SECS <command...>
  local max_attempts="$1"; shift
  local sleep_secs="$1"; shift
  local attempt=1
  while true; do
    if "$@"; then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "Command failed after $attempt attempts: $*" >&2
      return 1
    fi
    attempt=$((attempt+1))
    sleep "$sleep_secs"
  done
}

echo "===== Step 1: running UMRS API integration + ingestion ====="
chmod +x ./test_umrs_api.sh
./test_umrs_api.sh

# Check if RAG tests should be skipped
SKIP_RAG_TESTS=${SKIP_RAG_TESTS:-false}
if [ "$SKIP_RAG_TESTS" = "true" ]; then
  echo "===== Skipping RAG tests (SKIP_RAG_TESTS=true) ====="
  echo "===== UMRS API tests completed successfully ====="
  exit 0
fi

echo "===== Step 2: waiting for UMRS RAG microservice to be ready ====="
UMRS_RAG_HOST=${UMRS_RAG_HOST:-umrs-rag}
UMRS_RAG_PORT=${UMRS_RAG_PORT:-8081}

retry 60 2 curl -fsS "http://${UMRS_RAG_HOST}:${UMRS_RAG_PORT}/docs" >/dev/null

echo "===== Step 3: querying agentic RAG over user1 text-file bucket ====="

RAG_PAYLOAD=$(cat << 'EOF'
{
  "query": "What is the main topic of the uploaded text?",
  "top_k": 4,
  "max_context_chunks": 4,
  "enable_query_rewrite": true,
  "user": "user1",
  "xp_name": "Images and Texts",
  "bucket_alias": "bucket-5",
  "modality": "text file"
}
EOF
)

curl -fsS \
  -H "Content-Type: application/json" \
  -X POST \
  -d "${RAG_PAYLOAD}" \
  "http://${UMRS_RAG_HOST}:${UMRS_RAG_PORT}/rag/qa" | jq . || {
    echo "RAG query failed" >&2
    exit 1
  }

echo "===== End-to-end UMRS + RAG demo completed successfully ====="
