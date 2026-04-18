"""BentoML compute_embeddings adapters for UMRS vision services.

Bridges the ``bentoml.AsyncHTTPClient`` task API into the synchronous
compute_embeddings interface expected by EmbeddingModel / embeddings.py.

The generic factory ``make_bentoml_vision_compute`` creates a closure for
**any** model type supported by the generic ``/infer`` endpoint.  Internally
it creates a short-lived ``AsyncHTTPClient`` per batch via ``asyncio.run()``.
"""

import asyncio
import logging
import os
import pickle

import bentoml

from umrs.shared.storage.s3_presign import (
    generate_presigned_get,
    generate_presigned_put,
)
from umrs.vision_service.bentoml_client import async_submit_and_wait
from umrs.vision_service.output_specs import describe_output_artifacts

logger = logging.getLogger(__name__)


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

            async with bentoml.AsyncHTTPClient(service_url, token=token) as client:
                for idx, s3_key in enumerate(s3_keys):
                    logger.info("[%d/%d] BentoML %s: %s",
                                idx + 1, len(s3_keys), model_type, s3_key)
                    logger.warning("[DEBUG-ADAPTER] path_to_bucket=%s output_bucket=%s",
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
                    if skip_existing and output_bucket:
                        if predicted:
                            all_exist = True
                            first_key = None
                            for out_key in predicted:
                                if first_key is None:
                                    first_key = out_key
                                try:
                                    s3.head_object(Bucket=output_bucket, Key=out_key)
                                except Exception:
                                    all_exist = False
                                    break
                            if all_exist:
                                logger.info("[%d/%d] Skipping %s — output already exists in %s",
                                            idx + 1, len(s3_keys), s3_key, output_bucket)
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

                    logger.warning(
                        "[DEBUG-ADAPTER] Submitting task: model_type=%s image_url=%s "
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

                    logger.warning("[DEBUG-ADAPTER] Result keys=%s",
                                   list(result.keys()) if isinstance(result, dict) else type(result))

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

            return results, checksums, metadata_list, saved_keys

        return asyncio.run(_async_batch())

    return _compute