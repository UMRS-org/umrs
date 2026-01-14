from openai import OpenAI, OpenAIError
import httpx
from umrs.utils import flatten_text_chunks, add_embeddings_to_chunks_dict

_clients = {}

def get_client(url, model_name):
    global _client
    key = (url, model_name)

    client = _clients.get(key)
    if client is None:
        client = OpenAI(
            base_url=url,
            api_key="EMPTY",
        )
        _clients[key] = client

    return client

def is_server_alive(
    url: str,
    timeout: float = 2.0,
) -> bool:
    """
    Check whether a vLLM OpenAI-compatible server is alive.

    Args:
        url: Base URL, e.g. "http://localhost:8000/v1"
        timeout: timeout in seconds

    Returns:
        True if server responds correctly, False otherwise
    """
    endpoint = url.rstrip("/") + "/models"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(endpoint)
            return resp.status_code == 200
    except httpx.HTTPError:
        return False

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

    for model in payload.get("data", []):
        if model.get("id") == model_name:
            # vLLM variants (depending on version)
            if "max_model_len" in model:
                return int(model["max_model_len"])
            if "context_length" in model:
                return int(model["context_length"])
            if "max_position_embeddings" in model:
                return int(model["max_position_embeddings"])

            raise RuntimeError(
                f"Model {model_name} found, but no max token field exposed"
            )

    raise RuntimeError(f"Model {model_name} not found on vLLM server")

def get_max_tokens_from_vllm(url: str, model_name: str, timeout: float = 2.0) -> int:
    """
    Query a vLLM OpenAI-compatible server to get the max token length
    supported by a given model.
    """
    models_url = url.rstrip("/") + "/models"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(models_url)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to query models endpoint at {models_url}") from e

    payload = resp.json()

    for model in payload.get("data", []):
        if model.get("id") == model_name:
            # vLLM variants (depending on version)
            if "max_model_len" in model:
                return int(model["max_model_len"])
            if "context_length" in model:
                return int(model["context_length"])
            if "max_position_embeddings" in model:
                return int(model["max_position_embeddings"])

            raise RuntimeError(
                f"Model {model_name} found, but no max token field exposed"
            )

    raise RuntimeError(f"Model {model_name} not found on vLLM server")

def make_openai_compute_embeddings(url, model_name):
    def _compute_embeddings_openai_api(self, texts):
        client = get_client(url, model_name)

        try:
            flat_texts, chunks_dict = flatten_text_chunks(texts, self.nlp, tokenizer=self.tokenizer, max_tokens=self.max_tokens)
            response = client.embeddings.create(
                model=model_name,
                input=flat_texts,
            )
        except OpenAIError as e:
            raise RuntimeError(
                f"OpenAI API error while computing embeddings "
                f"(model={model_name}, n_texts={len(texts)})"
            ) from e
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"HTTP error while contacting vLLM server at {url}"
            ) from e
        
        if not response.data:
            raise RuntimeError("Empty embedding response from server")

        if len(response.data) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(response.data)}, "
                f"expected {len(texts)}"
            )
        
       
        embeddings = [d.embedding for d in response.data]
        result = add_embeddings_to_chunks_dict(embeddings, chunks_dict)
        return result
    
    return _compute_embeddings_openai_api