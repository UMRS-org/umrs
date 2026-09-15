"""BentoML Generic Vision Service for UMRS.

Provides a single ``/infer`` async task-queue endpoint that dispatches to
pluggable model adapters (SAM3, HMAE, or any future vision model).

BentoML auto-generates from ``@bentoml.task``:
    POST /infer/submit  - submit a task, returns task_id immediately
    GET  /infer/status  - check task status by task_id
    GET  /infer/get     - retrieve task result by task_id
    POST /infer/cancel  - cancel a pending task
    PUT  /infer/retry   - retry a failed task

Request payload for ``/infer/submit``::

    {
        "model_type":    "sam3" | "hmae" | "<future>",
        "inference_config":  { ... adapter-specific params ... },
        "image_url":     "s3://bucket/key.png",
        "output_bucket": "results"          # optional
    }

Environment variables:
        S3_ENDPOINT_URL      - MinIO/S3 endpoint for optional direct fallback
        S3_ACCESS_KEY        - S3 access key for optional direct fallback
        S3_SECRET_KEY        - S3 secret key for optional direct fallback
        VISION_REQUIRE_PRESIGNED_URLS
                                                - Require presigned GET for input and presigned PUT
                                                    for all declared outputs. Direct S3 fallback is
                                                    disabled when true.
    MAX_VISION_MODELS    - LRU model cache capacity (default 3)
    VISION_ADAPTERS_DIR  - colon-separated directories of custom adapter .py
                           files (default /adapters).  Each directory may also
                           contain a requirements.txt for auto-installation.
    VISION_ADAPTERS      - comma-separated ``module:ClassName`` entries for
                           custom adapters on the Python path
"""

import io
import logging
import os
import pickle
import time
import urllib.error
import urllib.request

import bentoml
import cv2  # noqa: F401 - preload before torch/sam3 to prevent libpng conflicts
from PIL import Image

from umrs.vision_service.adapters import ADAPTER_REGISTRY
from umrs.vision_service.model_cache import ModelCache

logger = logging.getLogger(__name__)


def _require_presigned_urls() -> bool:
    return os.environ.get("VISION_REQUIRE_PRESIGNED_URLS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
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
    path = image_url[5:] if image_url.startswith("s3://") else image_url
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid S3 image URL: {image_url}. Expected format: s3://bucket/key"
        )
    bucket, key = parts

    obj = s3.get_object(Bucket=bucket, Key=key)
    image_bytes = obj["Body"].read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB"), bucket, key, image_bytes


def _download_image_from_presigned_url(presigned_url, image_url):
    """Download an image via presigned GET URL (no S3 credentials needed).

    Retries up to 3 times with exponential backoff for transient failures.

    Returns the same 3-tuple as ``_download_image_from_s3``:
    ``(PIL.Image, bucket, key)`` - bucket/key are parsed from ``image_url``
    so that downstream code can still construct output keys.
    """
    path = image_url[5:] if image_url.startswith("s3://") else image_url
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 image URL: {image_url}")
    bucket, key = parts

    max_retries = 3
    last_exc = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(presigned_url, method="GET")
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
                image_bytes = resp.read()
            return (
                Image.open(io.BytesIO(image_bytes)).convert("RGB"),
                bucket,
                key,
                image_bytes,
            )
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                backoff = 2**attempt
                logger.warning(
                    "Presigned URL download attempt %d/%d failed: %s. "
                    "Retrying in %ds...",
                    attempt + 1,
                    max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
            else:
                logger.error(
                    "Presigned URL download failed after %d attempts: %s",
                    max_retries,
                    exc,
                )
    raise last_exc  # type: ignore[misc]


@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 3600, "max_concurrency": 64},
)
class VisionModelService:
    """Generic vision model service with pluggable adapters.

    Models are loaded lazily on first request and kept in an LRU cache.
    Multiple models can be resident concurrently (up to MAX_VISION_MODELS).

    Endpoints:
        POST /infer/submit  - async task inference
        GET  /models        - list available adapters and capabilities
        GET  /health        - readiness check
    """

    def __init__(self):
        self._s3 = None
        self._cache = ModelCache()

    def _ensure_s3(self):
        if self._s3 is None:
            self._s3 = _get_s3_client()
        return self._s3

    @bentoml.api
    def models(self) -> dict:
        """List available vision model adapters and their capabilities."""
        result = {}
        for name, adapter in ADAPTER_REGISTRY.items():
            caps = None
            if hasattr(adapter, "capabilities"):
                caps = adapter.capabilities()
            result[name] = caps or {
                "name": name,
                "description": "No capabilities declared",
            }
        return {"adapters": result, "count": len(result)}

    @bentoml.api
    def health(self) -> dict:
        """Readiness check returning OK if service accepts requests."""
        return {
            "status": "healthy",
            "adapters_loaded": len(ADAPTER_REGISTRY),
            "adapter_names": sorted(ADAPTER_REGISTRY.keys()),
            "cache_capacity": self._cache.capacity,
        }

    @bentoml.task
    def infer(
        self,
        model_type: str,
        inference_config: dict,
        image_url: str,
        output_bucket: str = "",
        presigned_input_url: str = "",
        presigned_output_urls: dict | None = None,
    ) -> bytes:
        """Run vision-model inference on a single image.

        Args:
            model_type: Adapter key registered in ADAPTER_REGISTRY
                        (e.g. ``"sam3"``, ``"hmae"``).
            inference_config: Adapter-specific configuration (checkpoint paths,
                              task, prompt, thresholds, etc.).
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
            Pickle-serialised bytes containing the adapter result dict.
            The dict may include numpy arrays and torch tensors.
            Clients must ``pickle.loads()`` the return value.
        """
        logger.info(
            "Vision infer request: model_type=%s image_url=%s output_bucket=%s "
            "presigned_input=%s presigned_output_keys=%s",
            model_type,
            image_url,
            output_bucket,
            bool(presigned_input_url),
            list(presigned_output_urls.keys()) if presigned_output_urls else "(empty)",
        )

        adapter = ADAPTER_REGISTRY.get(model_type)
        if adapter is None:
            logger.error(
                "Unknown vision model_type '%s'. Available: %s",
                model_type,
                sorted(ADAPTER_REGISTRY),
            )
            return pickle.dumps(
                {
                    "error": (
                        f"Unknown model_type '{model_type}'. "
                        f"Available: {sorted(ADAPTER_REGISTRY)}"
                    ),
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        try:
            if _require_presigned_urls() and not presigned_input_url:
                return pickle.dumps(
                    {
                        "error": (
                            "presigned_input_url is required when "
                            "VISION_REQUIRE_PRESIGNED_URLS=true"
                        ),
                        "source_image": image_url,
                    },
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            # Download input image: prefer presigned URL, fall back to direct S3
            if presigned_input_url:
                logger.debug("Downloading vision input via presigned URL")
                image, bucket, key, raw_image_bytes = (
                    _download_image_from_presigned_url(
                        presigned_input_url,
                        image_url,
                    )
                )
                s3 = (
                    None
                    if (_require_presigned_urls() or presigned_output_urls)
                    else self._ensure_s3()
                )
            else:
                logger.debug("Downloading vision input via direct S3: %s", image_url)
                s3 = self._ensure_s3()
                logger.debug(
                    "Vision direct S3 endpoint: %s",
                    os.environ.get("S3_ENDPOINT_URL", "(not set)"),
                )
                image, bucket, key, raw_image_bytes = _download_image_from_s3(
                    s3, image_url
                )

            expected_outputs = adapter.describe_outputs(key, inference_config)
            if _require_presigned_urls():
                missing = sorted(
                    out_key
                    for out_key in expected_outputs
                    if out_key not in (presigned_output_urls or {})
                )
                if missing:
                    return pickle.dumps(
                        {
                            "error": (
                                "Missing presigned output URLs for "
                                + ", ".join(missing)
                            ),
                            "source_image": image_url,
                        },
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

            logger.debug(
                "Vision input downloaded: bucket=%s key=%s size=%s",
                bucket,
                key,
                image.size,
            )

            model = self._cache.get_or_load(adapter, inference_config)
            logger.debug(
                "Dispatching vision request to adapter %s", type(adapter).__name__
            )

            result = adapter.infer(
                model,
                image,
                bucket,
                key,
                inference_config,
                s3,
                output_bucket or bucket,
                presigned_output_urls=presigned_output_urls or {},
                raw_image_bytes=raw_image_bytes,
            )
            logger.debug(
                "Vision adapter returned keys: %s",
                list(result.keys()) if isinstance(result, dict) else type(result),
            )
            return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.exception(
                "Vision inference failed (model_type=%s): %s", model_type, exc
            )
            return pickle.dumps(
                {"error": str(exc), "source_image": image_url},
                protocol=pickle.HIGHEST_PROTOCOL,
            )
