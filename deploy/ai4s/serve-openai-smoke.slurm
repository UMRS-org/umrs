#!/usr/bin/env bash
set -euo pipefail

# AI4S smoke endpoint wrapper.
# This validates the real client -> umrs-front -> AI4S Slurm -> compute-node
# Gateway path while the full Singularity/vLLM runtime is staged separately.

JOB_INFO_DIR="${JOB_INFO_DIR:-/work/hps0/home/ea0022/ai4s-platform/umrs/deploy/ai4s/jobs}"
RIVAULT_CONTROL_DIR="${RIVAULT_CONTROL_DIR:-${JOB_INFO_DIR}/control}"
LITELLM_PORT="${LITELLM_PORT:-${NGINX_LITELLM_DOCKER_PORT:-8001}}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-${MODEL_NAME:-ai4s-smoke}}"
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/work/hps0/home/ea0022/umrs_models/ai4s-smoke}"

mkdir -p "${JOB_INFO_DIR}" "${RIVAULT_CONTROL_DIR}"

RIVAULT_NODE="$(hostname -f 2>/dev/null || hostname)"
JOB_INFO_FILE="${JOB_INFO_DIR}/rivault-job-${SLURM_JOB_ID}.info"
STOP_REQUEST_FILE="${RIVAULT_CONTROL_DIR}/stop-${SLURM_JOB_ID}.request"
SERVER_SCRIPT="${TMPDIR:-/tmp}/ai4s-openai-smoke-${SLURM_JOB_ID}.py"
SERVER_PID=""

write_job_info() {
    local status="$1"
    cat > "${JOB_INFO_FILE}" <<EOF
RIVAULT_JOB_ID=${SLURM_JOB_ID}
RIVAULT_STATUS=${status}
RIVAULT_NODE=${RIVAULT_NODE}
RIVAULT_LITELLM_PORT=${LITELLM_PORT}
RIVAULT_VLLM_PORT=${VLLM_PORT}
RIVAULT_MODEL_PATH=${VLLM_MODEL_PATH}
RIVAULT_MODEL_NAME=${MODEL_NAME}
RIVAULT_START_TIME=$(date -Iseconds)
EOF
}

cleanup() {
    local rc=$?
    trap - EXIT TERM INT
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
    rm -f "${SERVER_SCRIPT}" "${STOP_REQUEST_FILE}" >/dev/null 2>&1 || true
    if [[ -f "${JOB_INFO_FILE}" ]]; then
        sed -i 's/RIVAULT_STATUS=.*/RIVAULT_STATUS=stopped/' "${JOB_INFO_FILE}" || true
        echo "RIVAULT_STOP_TIME=$(date -Iseconds)" >> "${JOB_INFO_FILE}" || true
    fi
    exit "${rc}"
}

trap cleanup EXIT TERM INT

if [[ ! -d "${VLLM_MODEL_PATH}" ]]; then
    mkdir -p "${VLLM_MODEL_PATH}"
fi

write_job_info starting

cat > "${SERVER_SCRIPT}" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_NAME = os.environ.get("MODEL_NAME", "ai4s-smoke")
PORT = int(os.environ.get("LITELLM_PORT", "8001"))


class Handler(BaseHTTPRequestHandler):
    server_version = "AI4SOpenAISmoke/1.0"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/health", "/health/liveliness", "/ready"}:
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "ai4s"}],
            })
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            request = {}

        if self.path == "/v1/chat/completions":
            requested_model = request.get("model") or MODEL_NAME
            self._send_json(200, {
                "id": f"chatcmpl-ai4s-smoke-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": requested_model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "gateway hpc ready"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
            })
            return

        self._send_json(404, {"error": {"message": "not found"}})


ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
PY

chmod +x "${SERVER_SCRIPT}"

echo "[ai4s-smoke] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[ai4s-smoke] node=${RIVAULT_NODE}"
echo "[ai4s-smoke] model=${MODEL_NAME}"
echo "[ai4s-smoke] port=${LITELLM_PORT}"

MODEL_NAME="${MODEL_NAME}" LITELLM_PORT="${LITELLM_PORT}" python3 "${SERVER_SCRIPT}" &
SERVER_PID=$!

write_job_info running

while kill -0 "${SERVER_PID}" >/dev/null 2>&1; do
    if [[ -f "${STOP_REQUEST_FILE}" ]]; then
        echo "[ai4s-smoke] graceful stop requested: ${STOP_REQUEST_FILE}"
        exit 0
    fi
    sleep 2
done

wait "${SERVER_PID}"