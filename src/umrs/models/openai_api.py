from openai import OpenAI, OpenAIError
from typing import Optional
import hashlib
import httpx
from umrs.shared.utils import flatten_text_chunks, add_embeddings_to_chunks_dict, add_document_prefix

_clients = {}

def get_client(url, model_name, api_key=None):
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest() if api_key else None
    key = (url, model_name, api_key_hash)

    client = _clients.get(key)
    if client is None:
        client = OpenAI(
            base_url=url,
            api_key=api_key or "EMPTY",
        )
        _clients[key] = client

    return client


def is_server_alive(
    url: str,
    api_key: Optional[str] = None,
    timeout: float = 2.0,
) -> bool:
    """
    Check whether an OpenAI-compatible (vLLM or OpenAI) server is alive.

    Args:
        url: Base URL, e.g. "http://localhost:8000/v1"
        api_key: Optional API key (None for open/local vLLM servers)
        timeout: timeout in seconds

    Returns:
        True if server responds correctly, False otherwise
    """
    endpoint = url.rstrip("/") + "/models"

    headers = {}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(endpoint, headers=headers)
            return resp.status_code == 200
    except httpx.HTTPError:
        return False

def get_max_tokens_from_vllm(url: str, model_name: str, timeout: float = 2.0, api_key: Optional[str] = None) -> Optional[int]:
    """
    Query a vLLM OpenAI-compatible server to get the max token length
    supported by a given model.
    """
    models_url = url.rstrip("/") + "/models"

    headers = {}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(models_url, headers=headers)
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

            # LiteLLM proxy responses lack vLLM-specific fields;
            # return None so the caller can fall back to tokenizer metadata.
            return None

    raise RuntimeError(f"Model {model_name} not found on vLLM server")

def make_openai_compute_embeddings(url, model_name, api_key=None):
    """
    Factory that builds a `compute_embeddings` method backed by an
    OpenAI-compatible embedding API (e.g. vLLM).

    This function returns a callable that:
      - Is bound to an EmbeddingModel instance (`self`)
      - Automatically handles long inputs by splitting texts into
        sentence-level chunks using spaCy and a HuggingFace tokenizer,
        ensuring no chunk exceeds the model's max token limit
      - Sends the flattened list of chunks to the remote embedding server
      - Re-associates the returned embeddings with their original texts
        and chunk metadata

    Design rationale:
      - Keeps the EmbeddingModel API uniform: callers simply pass a list
        of texts, regardless of whether embeddings are computed locally
        or via a remote OpenAI-compatible service
      - Avoids leaking transport- or API-specific parameters (URL, model
        name) into the rest of the codebase
      - Makes sentence-level embeddings a first-class concept while still
        safely handling pathological long sentences

    Parameters:
      - url: Base URL of the OpenAI-compatible server (e.g. vLLM)
      - model_name: Name of the embedding model exposed by the server

    Returns:
      - A method compatible with `EmbeddingModel.compute_embeddings`
        that returns embeddings aligned with the original input structure
    """
    def _compute_embeddings_openai_api(self, texts):
        client = get_client(url, model_name, api_key=api_key)

        try:
            flat_texts, chunks_dict = flatten_text_chunks(texts, self.nlp, tokenizer=self.tokenizer, max_tokens=self.max_tokens)
            flat_texts = add_document_prefix(flat_texts, model_name)
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

        if len(response.data) != len(flat_texts):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(response.data)}, "
                f"expected {len(flat_texts)}"
            )


        embeddings = [d.embedding for d in response.data]
        result = add_embeddings_to_chunks_dict(embeddings, chunks_dict)
        return result

    return _compute_embeddings_openai_api
