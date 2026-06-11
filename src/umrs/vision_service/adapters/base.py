"""Abstract base class for vision model adapters.

To add a new vision model to the BentoML service:
  1. Subclass ``VisionModelAdapter``
  2. Implement ``load()``, ``infer()``, ``cache_key()``, ``describe_outputs()``
  3. Optionally override ``capabilities()`` and ``infer_batch()``
  4. Place the .py file in a VISION_ADAPTERS_DIR directory or register via
     VISION_ADAPTERS env var
"""

from __future__ import annotations

import abc
from typing import Any

from PIL import Image


class VisionModelAdapter(abc.ABC):
    """Interface that every vision model adapter must implement.

    v2 additions (backward-compatible):
    - ``capabilities()``: declare model metadata for orchestrator
    - ``infer_batch()``: process multiple images in one call
    - ``estimate_gpu_memory_mb()``: hint resource requirements
    - ``warmup()``: self-test after model load
    """

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def describe_outputs(self, key: str, config: dict[str, Any]) -> dict[str, str]:
        """Describe deterministic output artifacts for a given source object.

        Returns:
            Dict mapping output S3 key -> content type.

        Notes:
            This contract is used by the UMRS server to pre-generate
            presigned PUT URLs before the Bento task runs.
        """

    @abc.abstractmethod
    def load(self, config: dict[str, Any]) -> Any:
        """Load model weights into GPU/CPU memory.

        Args:
            config: Model-specific configuration dict supplied by the caller.
                    Typically includes ``checkpoint`` path and model hyper-params.

        Returns:
            An opaque model object (or tuple) stored in the model cache.
        """

    @abc.abstractmethod
    def infer(
        self,
        model: Any,
        image: Image.Image,
        bucket: str,
        key: str,
        config: dict[str, Any],
        s3,
        output_bucket: str,
        presigned_output_urls: dict[str, str] | None = None,
        raw_image_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        """Run inference on a single image.

        Args:
            model: The object returned by ``load()``.
            image: PIL RGB image already downloaded from S3.
            bucket: Source S3 bucket.
            key: Source S3 object key.
            config: Same config dict as ``load()``; may contain per-request
                    parameters (e.g. ``mode``, ``prompt``).
            s3: boto3 S3 client for uploading results. May be ``None`` when
                presigned output URLs are provided.
            output_bucket: Bucket to write output artifacts to.
            presigned_output_urls: Optional dict mapping output_key ->
                    presigned PUT URL.  When provided, adapters should upload
                    results via HTTP PUT instead of ``s3.put_object()``.
            raw_image_bytes: Original undecoded image bytes from S3.
                    Adapters that need the raw file (e.g. for OpenCV
                    grayscale decoding) should prefer these over
                    re-serialising the PIL image.

        Returns:
            Result dict.  May contain non-JSON-serialisable objects
            (numpy arrays, torch tensors); the service layer handles
            pickle serialization before returning to the client.
        """

    @abc.abstractmethod
    def cache_key(self, config: dict[str, Any]) -> str:
        """Return a stable string key that uniquely identifies the loaded model.

        Two configs that would produce the same in-memory model **must**
        return the same cache key.
        """

    # ------------------------------------------------------------------
    # Optional v2 methods (with default implementations)
    # ------------------------------------------------------------------

    def capabilities(self) -> dict[str, Any] | None:
        """Declare model capabilities for orchestrator resource planning.

        Override to return a dict compatible with ``ModelCapabilities.to_dict()``.
        When ``None`` (default), the orchestrator uses conservative defaults.

        Example::

            return {
                "name": "sam3",
                "supported_tasks": ["image-segmentation"],
                "supported_inputs": ["image_with_text", "image_with_points"],
                "gpu_memory_mb": 6000,
                "supports_batch": False,
                "model_source": "facebook/sam3",
            }
        """
        return None

    def infer_batch(
        self,
        model: Any,
        images: list[Image.Image],
        keys: list[str],
        bucket: str,
        config: dict[str, Any],
        s3,
        output_bucket: str,
        presigned_output_urls_list: list[dict[str, str]] | None = None,
        raw_image_bytes_list: list[bytes] | None = None,
    ) -> list[dict[str, Any]]:
        """Run inference on a batch of images.

        Default implementation processes images sequentially via ``infer()``.
        Override for models that benefit from batched GPU execution.

        Args:
            model: The object returned by ``load()``.
            images: List of PIL RGB images.
            keys: List of S3 object keys (one per image).
            bucket: Source S3 bucket (same for all images in batch).
            config: Adapter-specific configuration.
            s3: boto3 S3 client (may be None with presigned URLs).
            output_bucket: Target bucket for output artifacts.
            presigned_output_urls_list: List of presigned URL dicts per image.
            raw_image_bytes_list: List of raw image bytes per image.

        Returns:
            List of result dicts, one per input image.
        """
        results = []
        for i, (image, key) in enumerate(zip(images, keys, strict=True)):
            presigned = (
                presigned_output_urls_list[i] if presigned_output_urls_list else None
            )
            raw_bytes = raw_image_bytes_list[i] if raw_image_bytes_list else None
            result = self.infer(
                model,
                image,
                bucket,
                key,
                config,
                s3,
                output_bucket,
                presigned_output_urls=presigned,
                raw_image_bytes=raw_bytes,
            )
            results.append(result)
        return results

    def estimate_gpu_memory_mb(self, _config: dict[str, Any]) -> int:
        """Estimate GPU memory in MB needed for this model + config.

        Used by the orchestrator for Slurm resource allocation.
        Default returns 4096 MB (4 GB) as a safe conservative estimate.
        """
        return 4096

    def warmup(self, _model: Any, _config: dict[str, Any]) -> bool:
        """Run a self-test after loading to verify the model works.

        Called by ModelCache after ``load()`` succeeds. Return True if
        warmup passes, False otherwise. Default always returns True.

        Override to run a small inference pass (e.g. on a synthetic image)
        to catch initialization issues early.
        """
        return True
