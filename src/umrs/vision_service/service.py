"""BentoML Generic Vision Service for UMRS.

Provides a single ``/infer`` async task-queue endpoint that dispatches to
pluggable model adapters (SAM3, HMAE, or any future vision model).

BentoML auto-generates from ``@bentoml.task``:
    POST /infer/submit  — submit a task, returns task_id immediately
    GET  /infer/status  — check task status by task_id
    GET  /infer/get     — retrieve task result by task_id
    POST /infer/cancel  — cancel a pending task
    PUT  /infer/retry   — retry a failed task

Request payload for ``/infer/submit``::

    {
        "model_type":    "sam3" | "hmae" | "<future>",
        "inference_config":  { ... adapter-specific params ... },
        "image_url":     "s3://bucket/key.png",
        "output_bucket": "results"          # optional
    }

Environment variables:
        S3_ENDPOINT_URL      — MinIO/S3 endpoint for optional direct fallback
        S3_ACCESS_KEY        — S3 access key for optional direct fallback
        S3_SECRET_KEY        — S3 secret key for optional direct fallback
        VISION_REQUIRE_PRESIGNED_URLS
                                                — Require presigned GET for input and presigned PUT
                                                    for all declared outputs. Direct S3 fallback is
                                                    disabled when true.
    MAX_VISION_MODELS    — LRU model cache capacity (default 3)
    VISION_ADAPTERS_DIR  — colon-separated directories of custom adapter .py
                           files (default /adapters).  Each directory may also
                           contain a requirements.txt for auto-installation.
    VISION_ADAPTERS      — comma-separated ``module:ClassName`` entries for
                           custom adapters on the Python path
"""

import io
import logging
import os
import urllib.request

import bentoml
from PIL import Image

from umrs.vision_service.adapters import ADAPTER_REGISTRY
from umrs.vision_service.model_cache import ModelCache

logger = logging.getLogger(__name__)


def _require_presigned_urls() -> bool:
    return os.environ.get("VISION_REQUIRE_PRESIGNED_URLS", "").lower() in {
        "1", "true", "yes", "on",
    }


def _get_s3_client():
    """Create a boto3 S3 client from environment variables."""
    if _require_presigned_urls():
        raise RuntimeError(
            "Direct S3 access is disabled when VISION_REQUIRE_PRESIGNED_URLS=true"
        )

    import boto3

    endpoint_url = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
    access_key = os.environ.get("S3_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("S3_SECRET_KEY", "minioadmin")

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _download_image_from_s3(s3, image_url):
    """Parse an S3 URL (s3://bucket/key) and download as PIL Image."""
    if image_url.startswith("s3://"):
        path = image_url[5:]
    else:
        path = image_url

    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 image URL: {image_url}. Expected format: s3://bucket/key")
    bucket, key = parts

    obj = s3.get_object(Bucket=bucket, Key=key)
    image_bytes = obj["Body"].read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB"), bucket, key


def _download_image_from_presigned_url(presigned_url, image_url):
    """Download an image via presigned GET URL (no S3 credentials needed).

    Returns the same 3-tuple as ``_download_image_from_s3``:
    ``(PIL.Image, bucket, key)`` — bucket/key are parsed from ``image_url``
    so that downstream code can still construct output keys.
    """
    if image_url.startswith("s3://"):
        path = image_url[5:]
    else:
        path = image_url
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 image URL: {image_url}")
    bucket, key = parts

    req = urllib.request.Request(presigned_url, method="GET")
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        image_bytes = resp.read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB"), bucket, key


@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 600},
)
class VisionModelService:
    """Generic vision model service with pluggable adapters.

    Models are loaded lazily on first request and kept in an LRU cache.
    Multiple models can be resident concurrently (up to MAX_VISION_MODELS).
    """

    def __init__(self):
        self._s3 = None
        self._cache = ModelCache()

    def _ensure_s3(self):
        if self._s3 is None:
            self._s3 = _get_s3_client()
        return self._s3

    @bentoml.task
    def infer(
        self,
        model_type: str,
        inference_config: dict,
        image_url: str,
        output_bucket: str = "",
        presigned_input_url: str = "",
        presigned_output_urls: dict | None = None,
    ) -> dict:
        """Run vision-model inference on a single image.

        Args:
            model_type: Adapter key registered in ADAPTER_REGISTRY
                        (e.g. ``"sam3"``, ``"hmae"``).
            inference_config: Adapter-specific configuration (checkpoint paths,
                              task, prompt, thresholds, …).
            image_url: S3 URL of the input image (``s3://bucket/key``).
            output_bucket: Bucket for output artifacts.  Defaults to the
                           source bucket when empty.
            presigned_input_url: Presigned GET URL for the input image.
                                 When provided, the service downloads via
                                 HTTP instead of using direct S3 credentials.
            presigned_output_urls: Dict mapping output_key -> presigned PUT URL.
                                   When provided, adapters upload results via
                                   HTTP PUT instead of ``s3.put_object()``.

        Returns:
            dict with adapter-specific results and output S3 keys.
        """
        logger.warning(
            "[DEBUG-INFER] model_type=%s image_url=%s output_bucket=%s "
            "presigned_input=%s presigned_output_keys=%s",
            model_type, image_url, output_bucket,
            bool(presigned_input_url),
            list(presigned_output_urls.keys()) if presigned_output_urls else "(empty)",
        )

        adapter = ADAPTER_REGISTRY.get(model_type)
        if adapter is None:
            logger.error("[DEBUG-INFER] Unknown model_type '%s'. Available: %s",
                         model_type, sorted(ADAPTER_REGISTRY))
            return {
                "error": (
                    f"Unknown model_type '{model_type}'. "
                    f"Available: {sorted(ADAPTER_REGISTRY)}"
                ),
            }

        try:
            if _require_presigned_urls() and not presigned_input_url:
                return {
                    "error": (
                        "presigned_input_url is required when "
                        "VISION_REQUIRE_PRESIGNED_URLS=true"
                    ),
                    "source_image": image_url,
                }

            # Download input image: prefer presigned URL, fall back to direct S3
            if presigned_input_url:
                logger.warning("[DEBUG-INFER] Downloading via presigned URL")
                image, bucket, key = _download_image_from_presigned_url(
                    presigned_input_url, image_url,
                )
                s3 = None if (_require_presigned_urls() or presigned_output_urls) else self._ensure_s3()
            else:
                logger.warning("[DEBUG-INFER] Downloading via direct S3: %s", image_url)
                s3 = self._ensure_s3()
                logger.warning("[DEBUG-INFER] S3 endpoint: %s",
                               os.environ.get('S3_ENDPOINT_URL', '(not set)'))
                image, bucket, key = _download_image_from_s3(s3, image_url)

            expected_outputs = adapter.describe_outputs(key, inference_config)
            if _require_presigned_urls():
                missing = sorted(
                    out_key
                    for out_key in expected_outputs
                    if out_key not in (presigned_output_urls or {})
                )
                if missing:
                    return {
                        "error": (
                            "Missing presigned output URLs for "
                            + ", ".join(missing)
                        ),
                        "source_image": image_url,
                    }

            logger.warning("[DEBUG-INFER] Image downloaded: bucket=%s key=%s size=%s",
                           bucket, key, image.size)

            model = self._cache.get_or_load(adapter, inference_config)
            logger.warning("[DEBUG-INFER] Model loaded, dispatching to adapter %s",
                           type(adapter).__name__)

            result = adapter.infer(
                model, image, bucket, key,
                inference_config, s3, output_bucket or bucket,
                presigned_output_urls=presigned_output_urls or {},
            )
            logger.warning("[DEBUG-INFER] Adapter returned: %s",
                           {k: v for k, v in result.items() if k != 'error'} if isinstance(result, dict) else result)
            return result
        except Exception as exc:
            logger.exception("[DEBUG-INFER] infer FAILED (model_type=%s): %s",
                             model_type, exc)
            return {"error": str(exc), "source_image": image_url}
