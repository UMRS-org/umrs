# Demo 29: LLM Completions (OpenAI-Compatible)

## Purpose

Demonstrates the UMRS OpenAI-compatible LLM API endpoints —
chat completions, text completions, embeddings, and model listing.

## API Endpoints Covered

| Method | Path                                | Description                |
|--------|-------------------------------------|----------------------------|
| GET    | `/umrs/llm/v1/models`               | List available models      |
| POST   | `/umrs/llm/v1/chat/completions`     | Chat completion            |
| POST   | `/umrs/llm/v1/completions`          | Text completion            |
| POST   | `/umrs/llm/v1/embeddings`           | Generate embeddings        |

## Prerequisites

- UMRS HTTP sidecar running (`UMRS_HTTP_URL` or default `http://localhost:8001`)
- `curl` and `jq` installed
- A valid project ID (default: `default-project`)
- At least one LLM model registered and endpoint provisioned

## Step-by-Step Commands

### Step 1: List available models

```bash
UMRS_URL="${UMRS_HTTP_URL:-http://localhost:8001}"

curl -sf "${UMRS_URL}/umrs/llm/v1/models" \
  -H "X-Project-ID: default-project" | jq .
```

**Expected output:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "rivault-gpt-oss-static",
      "object": "model",
      "created": 1719900000,
      "owned_by": "umrs"
    }
  ]
}
```

### Step 2: Generate a chat completion

```bash
curl -sf -X POST "${UMRS_URL}/umrs/llm/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-Project-ID: default-project" \
  -d '{
    "model": "rivault-gpt-oss-static",
    "messages": [
      {"role": "user", "content": "Hello, what is UMRS?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq .
```

**Expected output:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1719900000,
  "model": "rivault-gpt-oss-static",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "UMRS is a unified model runtime system..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 30,
    "total_tokens": 40
  }
}
```

### Step 3: Generate a text completion

```bash
curl -sf -X POST "${UMRS_URL}/umrs/llm/v1/completions" \
  -H "Content-Type: application/json" \
  -H "X-Project-ID: default-project" \
  -d '{
    "model": "rivault-gpt-oss-static",
    "prompt": "The capital of Japan is",
    "max_tokens": 5,
    "temperature": 0
  }' | jq .
```

### Step 4: Generate embeddings

```bash
curl -sf -X POST "${UMRS_URL}/umrs/llm/v1/embeddings" \
  -H "Content-Type: application/json" \
  -H "X-Project-ID: default-project" \
  -d '{
    "model": "text-embedding-3-small",
    "input": "Hello world"
  }' | jq .
```

## Notes

- The LLM proxy forwards requests to the active endpoint's LiteLLM instance.
- If no endpoint is active, requests will return `503 Service Unavailable`.
- Model names must match registered models in the UMRS model registry.
- Token usage is tracked for billing and quota enforcement.

## Cleanup

```bash
# No cleanup needed — completions are stateless.
echo "[demo29] LLM completion demo complete."
```
