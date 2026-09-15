#!/usr/bin/env python3
from openai import OpenAI
import httpx


def get_max_tokens_from_vllm(url: str, model_name: str, timeout: float = 2.0) -> int:
    """
    Query a vLLM OpenAI-compatible server to get the max token length
    supported by a given model.
    """
    models_url = url.rstrip("/") + "/v1/models"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(models_url)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to query models endpoint at {models_url}") from e

    payload = resp.json()

    print(payload)

    for model in payload.get("data", []):
        if model.get("id") == model_name:
            # vLLM variants (depending on version)
            if "max_model_len" in model:
                return int(model["max_model_len"])
            if "context_length" in model:
                return int(model["context_length"])
            if "max_position_embeddings" in model:
                return int(model["max_position_embeddings"])

            raise RuntimeError(f"Model {model_name} found, but no max token field exposed")

    raise RuntimeError(f"Model {model_name} not found on vLLM server")


print(get_max_tokens_from_vllm("http://localhost:8001", "intfloat/e5-base-v2"))

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",
)

texts = ["ojqdpodj"]

response = client.embeddings.create(
    model="intfloat/e5-base-v2",
    input=texts,
)

embeddings = [d.embedding for d in response.data]

print(f"batch size = {len(embeddings)}")
print(f"embedding dim = {len(embeddings[0])}")
