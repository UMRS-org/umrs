"""BentoML compute_embeddings adapters for UMRS vision services.

Bridges the ``bentoml.AsyncHTTPClient`` task API into the synchronous
compute_embeddings interface expected by EmbeddingModel / embeddings.py.

The generic factory ``make_bentoml_vision_compute`` creates a closure for
**any** model type supported by the generic ``/infer`` endpoint.  Internally
it creates a short-lived ``AsyncHTTPClient`` per batch via ``asyncio.run()``.

R-16 (Phase 2k): the adapter caches one long-lived ``AsyncHTTPClient`` per
``(service_url, token)`` pair, opened lazily on first use and reused across
batches. Call :func:`close_bentoml_clients` from process-shutdown hooks to
release file descriptors cleanly. Set ``BENTOML_CLIENT_REUSE=0`` to fall
back to the legacy per-batch client (useful when debugging hung tasks).
"""

import asyncio
import logging
import os
import threading

from botocore.exceptions import ClientError

from umrs.shared.storage.s3_presign import (
    generate_presigned_get,
    generate_presigned_put,
)
from umrs.vision_service.bentoml_client import async_submit_and_wait
from umrs.vision_service.output_specs import (
    describe_output_artifacts,
    resolve_hmae_task,
)

logger = logging.getLogger(__name__)


# R-16: per-event-loop AsyncHTTPClient cache. ``asyncio.run`` creates a fresh
# event loop per batch, and BentoML's HTTP client is bound to the loop in
# which it was opened. We therefore key the cache by ``(loop_id, service_url,
# token)`` so a stale entry from a previous loop is never reused.
_CLIENT_CACHE: dict[tuple[int, str, str | None], "bentoml.AsyncHTTPClient"] = {}
# Use a threading.Lock (not asyncio.Lock) because each asyncio.run() creates a
# fresh event loop — a module-level asyncio.Lock would bind to the first loop
# and raise "bound to a different event loop" on subsequent asyncio.run() calls.
_CLIENT_CACHE_LOCK = threading.Lock()


def _client_reuse_enabled() -> bool:
    return os.environ.get("BENTOML_CLIENT_REUSE", "1").lower() in {
        "1", "true", "yes", "on",
    }


async def _get_or_open_client(service_url: str, token: str | None):
    """Return an ``AsyncHTTPClient`` cached for the *current* event loop.

    When the loop differs from the cache key (typical under per-batch
    ``asyncio.run`` callers), a new client is opened. The previous client
    bound to a now-closed loop is garbage-collected; its underlying socket
    was already torn down with the loop.
    """
    loop_id = id(asyncio.get_running_loop())
    key = (loop_id, service_url, token)
    cached = _CLIENT_CACHE.get(key)
    if cached is not None:
        return cached
    with _CLIENT_CACHE_LOCK:
        cached = _CLIENT_CACHE.get(key)
        if cached is not None:
            return cached
        import bentoml

        client = bentoml.AsyncHTTPClient(service_url, token=token)
        await client.__aenter__()
        _CLIENT_CACHE[key] = client
        logger.debug(
            "Opened cached BentoML client for %s on loop %s",
            service_url, loop_id,
        )
        return client


async def close_bentoml_clients() -> None:
    """Close every cached BentoML client bound to the current loop."""
    loop_id = id(asyncio.get_running_loop())
    with _CLIENT_CACHE_LOCK:
        for key in list(_CLIENT_CACHE.keys()):
            if key[0] != loop_id:
                continue
            client = _CLIENT_CACHE.pop(key)
            try:
                await client.__aexit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                logger.warning("Error closing BentoML client %s: %s", key[1], exc)


def _copy_to_output_bucket(s3, s3_uri, output_bucket):
    """Copy an object from its current bucket to the output bucket.

    Args:
        s3: boto3 S3 client.
        s3_uri: Full S3 URI like ``s3://bucket/key``.
        output_bucket: Target output bucket name.

    Returns:
        New S3 URI in the output bucket, or the original if copy is skipped.
    """
    if not output_bucket or not s3_uri:
        return s3_uri
    stripped = s3_uri.replace("s3://", "", 1)
    src_bucket, key = stripped.split("/", 1)
    if src_bucket == output_bucket:
        return s3_uri
    copy_source = {"Bucket": src_bucket, "Key": key}
    s3.copy_object(CopySource=copy_source, Bucket=output_bucket, Key=key)
    new_uri = f"s3://{output_bucket}/{key}"
    logger.info("Copied %s -> %s", s3_uri, new_uri)
    return new_uri


def _require_presigned_urls() -> bool:
    return os.environ.get("VISION_REQUIRE_PRESIGNED_URLS", "").lower() in {
        "1", "true", "yes", "on",
    }


def _presigned_ttl_sec() -> int:
    raw = os.environ.get("VISION_PRESIGNED_URL_TTL_SEC", "3600")
    try:
        ttl = int(raw)
    except ValueError:
        logger.warning(
            "Invalid VISION_PRESIGNED_URL_TTL_SEC=%s; falling back to 3600",
            raw,
        )
        return 3600
    return max(ttl, 1)


def _vision_presign_endpoint():
    endpoint = os.environ.get("VISION_S3_PRESIGN_ENDPOINT", "").strip()
    return endpoint or None


def _jens_task_for_model(model_type: str, config: dict) -> str:
    if model_type == "sam3":
        return str(config.get("prompt") or config.get("task") or "object")
    if model_type == "hmae":
        return {
            "reconstruction": "reconstruct",
            "mae_reconstruction": "mae",
            "feature_extraction": "features",
        }.get(resolve_hmae_task(config), "reconstruct")
    return str(config.get("task") or config.get("prompt") or "")


def _jens_output_artifact(model_type: str, s3_key: str, config: dict) -> tuple[str, str]:
    predicted = describe_output_artifacts(model_type, s3_key, config)
    for out_key in predicted:
        if out_key.endswith(".pkl"):
            return out_key, "application/octet-stream"
    if predicted:
        out_key = next(iter(predicted))
        return out_key, "application/octet-stream"
    base = s3_key.rsplit(".", 1)[0] if "." in s3_key else s3_key
    return f"{base}_{model_type}_results.pkl", "application/octet-stream"


def make_jens_vision_compute(service_url: str, model_type, model_config,
                             task=None, token: str | None = None,
                             poll_interval=2.0, timeout=1200.0):
    """Create a compute_embeddings closure for internal_genAI VisionFrontend.

    The Jens stack exposes ``vision_task(image, model, task, out)`` rather
    than UMRS's generic ``infer`` task.  UMRS still owns S3 and presigning:
    the internal_genAI container receives a presigned input URL and uploads one
    pickle result artifact through a presigned PUT URL.
    """
    effective_config = dict(model_config)
    if task:
        effective_config["task"] = task
    effective_token = (
        token
        if token is not None
        else os.environ.get("JENS_VISION_TOKEN") or os.environ.get("LITELLM_MASTER_KEY")
    )

    def _compute(self, s3, s3_keys, path_to_bucket, output_bucket, skip_existing=True):

        async def _async_batch():
            results = []
            checksums = []
            metadata_list = []
            saved_keys = []

            if _client_reuse_enabled():
                client = await _get_or_open_client(service_url, effective_token)
                _client_ctx = None
            else:
                import bentoml

                _client_ctx = bentoml.AsyncHTTPClient(service_url, token=effective_token)
                client = await _client_ctx.__aenter__()

            try:
                for idx, s3_key in enumerate(s3_keys):
                    logger.info("[%d/%d] Jens BentoML %s: %s",
                                idx + 1, len(s3_keys), model_type, s3_key)

                    obj = s3.head_object(Bucket=path_to_bucket, Key=s3_key)
                    checksums.append(obj["ETag"].strip('"'))
                    metadata_list.append(obj.get("Metadata", {}))

                    target_bucket = output_bucket or path_to_bucket
                    out_key, content_type = _jens_output_artifact(
                        model_type,
                        s3_key,
                        effective_config,
                    )

                    if skip_existing and output_bucket:
                        try:
                            s3.head_object(Bucket=target_bucket, Key=out_key)
                        except ClientError as exc:
                            error_code = exc.response.get("Error", {}).get("Code")
                            if error_code not in {"404", "NoSuchKey", "NotFound"}:
                                raise
                        else:
                            output_s3_uri = f"s3://{target_bucket}/{out_key}"
                            saved_keys.append(output_s3_uri)
                            results.append({"status": "skipped", "output_s3_key": output_s3_uri})
                            continue

                    presign_endpoint = _vision_presign_endpoint()
                    presigned_input_url = generate_presigned_get(
                        s3,
                        path_to_bucket,
                        s3_key,
                        ttl=_presigned_ttl_sec(),
                        endpoint_override=presign_endpoint,
                    )
                    if not presigned_input_url:
                        raise RuntimeError(f"Could not presign input s3://{path_to_bucket}/{s3_key}")

                    presigned_output_url = generate_presigned_put(
                        s3,
                        target_bucket,
                        out_key,
                        content_type=content_type,
                        ttl=_presigned_ttl_sec(),
                        endpoint_override=presign_endpoint,
                    )
                    if not presigned_output_url:
                        raise RuntimeError(f"Could not presign output s3://{target_bucket}/{out_key}")

                    jens_task = _jens_task_for_model(str(model_type), effective_config)
                    result = await async_submit_and_wait(
                        client,
                        "vision_task",
                        poll_interval=poll_interval,
                        timeout=timeout,
                        image=presigned_input_url,
                        model=model_type,
                        task=jens_task,
                        out=presigned_output_url,
                    )

                    if isinstance(result, dict) and result.get("status") not in {None, "success"}:
                        raise RuntimeError(
                            f"Jens BentoML {model_type} task failed for {s3_key}: "
                            f"{result.get('status')}"
                        )

                    output_s3_uri = f"s3://{target_bucket}/{out_key}"
                    if isinstance(result, dict):
                        result.setdefault("source_image", f"s3://{path_to_bucket}/{s3_key}")
                        result.setdefault("task", jens_task)
                        result.setdefault("output_s3_key", output_s3_uri)
                        result.setdefault("results_pkl_s3_key", output_s3_uri)
                    results.append(result)
                    saved_keys.append(output_s3_uri)
            finally:
                if _client_ctx is not None:
                    await _client_ctx.__aexit__(None, None, None)

            return results, checksums, metadata_list, saved_keys

        return asyncio.run(_async_batch())

    return _compute


def make_bentoml_vision_compute(service_url: str, model_type, model_config,
                                task=None, token: str | None = None,
                                poll_interval=2.0, timeout=1200.0):
    """Create a compute_embeddings closure for any vision model via BentoML.

    Submits tasks to the ``/infer`` endpoint using an ``AsyncHTTPClient``
    created per batch call (via ``asyncio.run()``).

    Args:
        service_url: Base URL of the BentoML vision service
                     (e.g. ``"http://localhost:3001"``).
        model_type: Adapter key (e.g. ``"sam3"``, ``"hmae"``).
        model_config: Adapter-specific config dict forwarded to the service.
        task: UMRS task name (e.g. ``"reconstruction"``).
        token: Optional auth token for the service.
        poll_interval: Seconds between status polls.
        timeout: Maximum seconds to wait per task.

    Returns:
        A method suitable for binding to ``EmbeddingModel.compute_embeddings``.
    """
    # Inject task into model_config so it reaches the server-side adapter.
    effective_config = dict(model_config)
    if task:
        effective_config["task"] = task

    def _compute(self, s3, s3_keys, path_to_bucket, output_bucket, skip_existing=True):

        async def _async_batch():
            results = []
            checksums = []
            metadata_list = []
            saved_keys = []

            # R-16: prefer the long-lived cached client; fall back to a
            # per-batch context-managed client when the cache is disabled.
            if _client_reuse_enabled():
                client = await _get_or_open_client(service_url, token)
                _client_ctx = None
            else:
                import bentoml

                _client_ctx = bentoml.AsyncHTTPClient(service_url, token=token)
                client = await _client_ctx.__aenter__()

            try:
                for idx, s3_key in enumerate(s3_keys):
                    logger.info("[%d/%d] BentoML %s: %s",
                                idx + 1, len(s3_keys), model_type, s3_key)
                    logger.debug("Vision adapter buckets: input=%s output=%s",
                                 path_to_bucket, output_bucket)

                    obj = s3.head_object(Bucket=path_to_bucket, Key=s3_key)
                    checksums.append(obj["ETag"].strip('"'))
                    metadata_list.append(obj.get("Metadata", {}))

                    predicted = describe_output_artifacts(
                        model_type,
                        s3_key,
                        effective_config,
                    )
                    if _require_presigned_urls() and not predicted:
                        raise RuntimeError(
                            "Strict presigned vision mode requires declared output "
                            f"artifacts for model_type '{model_type}'"
                        )

                    # Skip if output already exists in the output bucket
                    if skip_existing and output_bucket and predicted:
                        all_exist = True
                        first_key = None
                        for out_key in predicted:
                            if first_key is None:
                                first_key = out_key
                            try:
                                s3.head_object(Bucket=output_bucket, Key=out_key)
                            except ClientError as exc:
                                error_code = exc.response.get("Error", {}).get("Code")
                                if error_code not in {"404", "NoSuchKey", "NotFound"}:
                                    raise
                                all_exist = False
                                break
                        if all_exist:
                            logger.info(
                                "[%d/%d] Skipping %s; output already exists in %s",
                                idx + 1,
                                len(s3_keys),
                                s3_key,
                                output_bucket,
                            )
                            output_s3_uri = f"s3://{output_bucket}/{first_key}"
                            saved_keys.append(output_s3_uri)
                            results.append({"status": "skipped", "output_s3_key": output_s3_uri})
                            continue

                    image_url = f"s3://{path_to_bucket}/{s3_key}"
                    target_bucket = output_bucket or path_to_bucket
                    presign_endpoint = _vision_presign_endpoint()
                    presigned_input_url = generate_presigned_get(
                        s3,
                        path_to_bucket,
                        s3_key,
                        ttl=_presigned_ttl_sec(),
                        endpoint_override=presign_endpoint,
                    )
                    presigned_output_urls = {
                        out_key: generate_presigned_put(
                            s3,
                            target_bucket,
                            out_key,
                            content_type=content_type,
                            ttl=_presigned_ttl_sec(),
                            endpoint_override=presign_endpoint,
                        )
                        for out_key, content_type in predicted.items()
                    }

                    logger.debug(
                        "Submitting BentoML vision task: model_type=%s image_url=%s "
                        "presigned_input=%s presigned_output_keys=%s endpoint=%s",
                        model_type, image_url,
                        bool(presigned_input_url),
                        list(presigned_output_urls.keys()) if presigned_output_urls else "(empty)",
                        presign_endpoint or "(default)",
                    )

                    result = await async_submit_and_wait(
                        client,
                        "infer",
                        poll_interval=poll_interval,
                        timeout=timeout,
                        model_type=model_type,
                        inference_config=effective_config,
                        image_url=image_url,
                        output_bucket=target_bucket,
                        presigned_input_url=presigned_input_url or "",
                        presigned_output_urls=presigned_output_urls,
                    )

                    logger.debug(
                        "BentoML vision result keys=%s",
                        list(result.keys()) if isinstance(result, dict) else type(result),
                    )

                    if "error" in result:
                        raise RuntimeError(
                            f"BentoML {model_type} task failed for {s3_key}: "
                            f"{result['error']}"
                        )

                    results.append(result)

                    output_s3_key = (result.get("output_s3_key")
                                     or result.get("overlay_s3_key"))
                    output_s3_key = _copy_to_output_bucket(s3, output_s3_key,
                                                           output_bucket)
                    saved_keys.append(output_s3_key)
            finally:
                if _client_ctx is not None:
                    await _client_ctx.__aexit__(None, None, None)

            return results, checksums, metadata_list, saved_keys

        return asyncio.run(_async_batch())

    return _compute
