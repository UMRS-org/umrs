#!/usr/bin/python3

import argparse
import os
import sys
from functools import lru_cache
from typing import Any, Dict, Tuple
from urllib.parse import quote

import tarfile
import json
import tempfile
import requests
import zmq
import numpy as np

# Ensure src/ is on sys.path when running as a standalone script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from umrs.s3 import load_data_from_s3, connect_to_s3
from umrs.models_loading import load_model
from umrs.utils import zero_out_embeddings_for_empty_objects
from umrs.cache_paths import cache_base_for_kind, find_model_cache


class WorkerConfigError(RuntimeError):
    """Raised when required worker configuration is missing."""


def _normalize_endpoint(raw_endpoint: str) -> str:
    endpoint = raw_endpoint.rstrip("/")
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        raise WorkerConfigError(
            f"RIVAULT endpoint must include http/https scheme (got '{raw_endpoint}')"
        )
    return endpoint


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert tensors/arrays to JSON-serializable structures.

    This is used for textual modalities where the worker replies with a
    JSON payload containing nested chunk dictionaries. It ensures that
    any torch.Tensor or numpy.ndarray instances are converted to plain
    Python lists/scalars before ``socket.send_json`` is called.
    """

    # Fast path for primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Containers: recurse
    if isinstance(obj, dict):
        return {key: _to_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(value) for value in obj]

    # Numpy arrays, torch tensors and similar objects implementing ``tolist``
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return _to_json_safe(tolist())

    # Fallback: try ``item()`` for 0-D tensors/arrays
    item = getattr(obj, "item", None)
    if callable(item):
        return item()

    # As a last resort, stringify the object so that the worker never
    # crashes the whole job on an unexpected type.
    return str(obj)


CATALOG_PATH = os.environ.get(
    "RIVAULT_MODELS_FILE",
    os.path.join(CURRENT_DIR, "rivault_models.json"),
)


@lru_cache(maxsize=1)
def _load_local_catalog(path: str | None = None) -> Dict[str, Dict[str, Any]]:
    """Load the local RiVault-style model catalog.

    This is used when the worker runs without a RiVault endpoint but still
    needs full model configuration (including vector size and type).
    """

    catalog_path = path or CATALOG_PATH
    if not os.path.exists(catalog_path):
        raise WorkerConfigError(
            f"Local RiVault catalog file not found: {catalog_path}"
        )

    try:
        with open(catalog_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        raise WorkerConfigError(
            f"Failed to load local RiVault catalog '{catalog_path}': {exc}"
        ) from exc

    models = raw.get("models", {})
    if not isinstance(models, dict):
        raise WorkerConfigError(
            "Invalid local RiVault catalog structure: 'models' must be a dict"
        )

    return models


def _load_local_model_cfg(model_name: str) -> Dict[str, Any]:
    """Return a full model config for local-loading scenarios.

    The returned dict is compatible with ``umrs.models_loading.load_model``
    (contains at least ``model_name``, ``vector_size`` and ``model_type``).
    """

    models = _load_local_catalog()
    cfg = models.get(model_name)
    if cfg is None:
        raise WorkerConfigError(
            f"Model config for '{model_name}' not found in local catalog"
        )
    return cfg


@lru_cache(maxsize=16)
def _fetch_model_cfg(endpoint: str, model_name: str) -> Dict[str, Any]:
    safe_name = quote(model_name, safe="")
    url = f"{endpoint}/api/models/{safe_name}"
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        # Missing config on RiVault: caller should fall back to local/HF model
        raise WorkerConfigError(f"Model config for '{model_name}' not found on RiVault ({url})")
    response.raise_for_status()
    payload = response.json()
    config = payload.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"RiVault returned invalid config payload for model '{model_name}'")
    return config


def _extract_weights_archive(tar_path: str) -> None:
    """Extract a RiVault tarball into the appropriate cache roots."""

    def _safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, base_path: str, relative_target: str) -> None:
        base_abs = os.path.abspath(base_path)
        target_abs = os.path.abspath(os.path.join(base_abs, relative_target))
        if not (target_abs == base_abs or target_abs.startswith(base_abs + os.sep)):
            return
        os.makedirs(base_abs, exist_ok=True)
        original_name = member.name
        try:
            member.name = relative_target
            tar.extract(member, path=base_abs)
        finally:
            member.name = original_name

    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            parts = member.name.split("/", 1)
            if len(parts) < 2:
                continue
            cache_kind, remainder = parts
            if not remainder:
                continue
            try:
                base_path = cache_base_for_kind(cache_kind)
            except ValueError:
                continue
            try:
                _safe_extract_member(tar, member, base_path, remainder)
            except Exception as exc:
                print(f"[worker] Failed to extract '{member.name}' into cache '{cache_kind}': {exc}")


def _ensure_weights_from_rivault(endpoint: str, model_name: str) -> None:
    """Ensure model weights are present in the worker's local cache."""

    if find_model_cache(model_name):
        return

    safe_name = quote(model_name, safe="")
    url = f"{endpoint}/api/models/{safe_name}/weights"

    try:
        resp = requests.get(url, stream=True, timeout=600)
    except Exception:
        return

    if resp.status_code != 200:
        return

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    break
                tmp_file.write(chunk)

        if tmp_path:
            _extract_weights_archive(tmp_path)
    except Exception as exc:
        print(f"[worker] Failed to download or extract weights for '{model_name}': {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _load_model_from_rivault(
    cache: Dict[Tuple[str, str, str | None], Any],
    cache_key: Tuple[str, str, str | None],
    endpoint: str,
    modality_name: str,
    model_name: str,
) -> Any:
    if cache_key not in cache:
        print(
            f"[worker] Fetching model '{model_name}' config from RiVault at {endpoint}"
        )
        # Try to fetch weights first so that the underlying loaders can pick
        # them up from the local cache instead of downloading from HuggingFace.
        _ensure_weights_from_rivault(endpoint, model_name)
        model_cfg = _fetch_model_cfg(endpoint, model_name)
        print(
            f"[worker] Loading model '{model_name}' for modality '{modality_name}'"
        )
        cache[cache_key] = load_model(model_cfg, modality_name)
    return cache[cache_key]


def _compute_embeddings(
    model: Any,
    modality_name: str,
    s3_connection: Any,
    path_to_bucket: str,
    s3_keys: list[str],
) -> Tuple[list[Any], list[Any], list[Any]]:
    if modality_name in ["tomography"]:
        embeddings, checksums, metadata = model.compute_embeddings(
            s3_connection,
            s3_keys,
            path_to_bucket,
        )
        return embeddings, checksums, metadata

    objects, metadata, checksums, sizes, has_zero = load_data_from_s3(
        s3_connection,
        path_to_bucket,
        s3_keys,
        modality_name,
    )
    embeddings = model.compute_embeddings(objects)
    if has_zero:
        zero_out_embeddings_for_empty_objects(embeddings, sizes, modality_name)
    return embeddings, checksums, metadata


def _get_or_load_model(
    model_cache: Dict[Tuple[str, str, str | None], Any],
    cache_key: Tuple[str, str, str | None],
    rivault_endpoint: str | None,
    modality_name: str,
    model_name: str,
) -> Any:
    """Return a cached model instance, loading it on first use.

    This keeps the main request loop focused on I/O and orchestration while
    encapsulating the model-selection and cache-discovery behaviour.
    """

    if cache_key in model_cache:
        return model_cache[cache_key]

    if rivault_endpoint is not None:
        model = _load_model_from_rivault(
            model_cache,
            cache_key,
            rivault_endpoint,
            modality_name,
            model_name,
        )
        return model

    print(
        f"[worker] Loading model '{model_name}' locally for modality '{modality_name}'"
    )
    model_cfg = _load_local_model_cfg(model_name)
    model_type = model_cfg.get("model_type")

    # For HF-based models we require that the server has already downloaded
    # weights into a shared cache that the worker can see. This avoids the
    # worker attempting to reach out to the internet on its own.
    if model_type in ("SentenceTransformer", "TransformerEncoder"):
        cache_entry = find_model_cache(model_name)
        if cache_entry is None:
            raise WorkerConfigError(
                f"Model cache for '{model_name}' not found on worker. "
                "Make sure the UMRS server has downloaded the model "
                "and that the cache directory is shared between server "
                "and worker nodes."
            )
    else:
        # Non-HF models (e.g. Resnet-50, MAE) typically rely on an explicit
        # checkpoint path rather than a Hugging Face cache. When a checkpoint
        # is configured but missing, log a warning and fall back to the model
        # loader's default behaviour (which may download weights).
        checkpoint_path = model_cfg.get("checkpoint_path")
        if checkpoint_path and not os.path.exists(checkpoint_path):
            print(
                f"[worker] Warning: checkpoint '{checkpoint_path}' for model "
                f"'{model_name}' not found; falling back to default weights."
            )

    model_cache[cache_key] = load_model(model_cfg, modality_name)
    return model_cache[cache_key]


def _send_textual_reply(
    socket: zmq.Socket,
    job_id: Any,
    modality_name: str,
    embeddings: Any,
    checksums: Any,
    metadata: Any,
) -> None:
    """Send a JSON reply for textual modalities.

    Text and text-file modes use nested chunk dictionaries that are not easily
    flattened into a 2D float array. We therefore keep the original JSON-based
    protocol and ensure the payload is fully JSON-serializable.
    """

    embeddings = _to_json_safe(embeddings)
    checksums = _to_json_safe(checksums)
    metadata = _to_json_safe(metadata)

    reply = {
        "job_id": job_id,
        "status": "ok",
        "modality_name": modality_name,
        "embeddings": embeddings,
        "checksums": checksums,
        "metadata": metadata,
    }
    socket.send_json(reply)


def _send_numeric_reply(
    socket: zmq.Socket,
    job_id: Any,
    modality_name: str,
    embeddings: Any,
    checksums: Any,
    metadata: Any,
) -> None:
    """Send a multipart reply for numeric embedding arrays."""

    embeddings_arr = np.asarray(embeddings)
    if embeddings_arr.ndim != 2:
        raise RuntimeError(
            f"Worker expected 2D embeddings array, got shape {embeddings_arr.shape}"
        )
    embeddings_arr = np.ascontiguousarray(embeddings_arr, dtype=np.float32)

    header = {
        "job_id": job_id,
        "status": "ok",
        "modality_name": modality_name,
        "shape": embeddings_arr.shape,
        "dtype": str(embeddings_arr.dtype),
    }

    header_frame = json.dumps(header).encode("utf-8")
    embeddings_frame = embeddings_arr.tobytes()
    checksums_frame = json.dumps(checksums).encode("utf-8")
    metadata_frame = json.dumps(metadata).encode("utf-8")

    socket.send_multipart(
        [header_frame, embeddings_frame, checksums_frame, metadata_frame]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("UMRS Embedding Worker")
    parser.add_argument(
        "--bind",
        default=os.getenv("EMBEDDING_WORKER_BIND", "tcp://0.0.0.0:6000"),
        help="ZeroMQ bind address for the worker REP socket",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("EMBEDDING_DEVICE", None),
        help="Optional device hint (e.g. 'cuda' or 'cpu'), passed to model loader if applicable",
    )
    parser.add_argument(
        "--s3-config",
        default=os.getenv("EMBEDDING_S3_CONFIG", os.path.expanduser("~/.s3cfg")),
        help="Path to S3 configuration file used by the worker",
    )
    parser.add_argument(
        "--rivault-endpoint",
        default=os.getenv("RIVAULT_ENDPOINT"),
        help="Base URL for the RiVault OpenAPI service (e.g. http://rivault:8000)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.rivault_endpoint:
        rivault_endpoint: str | None = _normalize_endpoint(args.rivault_endpoint)
        print(f"[worker] RiVault enabled at {rivault_endpoint}")
    else:
        rivault_endpoint = None
        print("[worker] RiVault disabled (RIVAULT_ENDPOINT not set)")

    print(f"[worker] Starting UMRS embedding worker on {args.bind}")

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(args.bind)

    s3_cfg_path = os.path.expanduser(args.s3_config)
    print(f"[worker] Using S3 config: {s3_cfg_path}")
    s3_connection, _ = connect_to_s3(s3_cfg_path)

    model_cache: Dict[Tuple[str, str, str | None], Any] = {}

    while True:
        request = socket.recv_json()
        job_id = request.get("job_id")

        try:
            command = request.get("command")
            if command != "compute_embeddings":
                raise ValueError(f"Unsupported command: {command}")

            path_to_bucket = request["path_to_bucket"]
            s3_keys = request["s3_keys"]
            modality_name = request["modality_name"]
            model_name = request.get("model_name")

            if not model_name:
                raise ValueError("Missing 'model_name' in request")

            cache_key = (modality_name, model_name, args.device)
            model = _get_or_load_model(
                model_cache,
                cache_key,
                rivault_endpoint,
                modality_name,
                model_name,
            )

            embeddings, checksums, metadata = _compute_embeddings(
                model,
                modality_name,
                s3_connection,
                path_to_bucket,
                s3_keys,
            )

            if modality_name in ("text", "text file"):
                _send_textual_reply(
                    socket,
                    job_id,
                    modality_name,
                    embeddings,
                    checksums,
                    metadata,
                )
                continue

            _send_numeric_reply(
                socket,
                job_id,
                modality_name,
                embeddings,
                checksums,
                metadata,
            )
            continue
        except Exception as exc:
            print(f"[worker] Error while processing job {job_id}: {exc}")
            reply = {
                "job_id": job_id,
                "status": "error",
                "error": str(exc),
            }
            # Errors are small and JSON-serializable, so we keep using JSON
            # here and send the reply immediately from the exception handler.
            socket.send_json(reply)




if __name__ == "__main__":
    main()
