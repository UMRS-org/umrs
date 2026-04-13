"""Abstract base class for vision model adapters.

To add a new vision model to the BentoML service:
  1. Subclass ``VisionModelAdapter``
  2. Implement ``load()``, ``infer()``, ``cache_key()``
  3. Register in ``adapters/__init__.py`` ADAPTER_REGISTRY
"""

from __future__ import annotations

import abc
from typing import Any, Dict

from PIL import Image


class VisionModelAdapter(abc.ABC):
    """Interface that every vision model adapter must implement."""

    @abc.abstractmethod
    def describe_outputs(self, key: str, config: Dict[str, Any]) -> Dict[str, str]:
        """Describe deterministic output artifacts for a given source object.

        Returns:
            Dict mapping output S3 key -> content type.

        Notes:
            This contract is used by the UMRS server to pre-generate
            presigned PUT URLs before the Bento task runs.
        """

    @abc.abstractmethod
    def load(self, config: Dict[str, Any]) -> Any:
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
        config: Dict[str, Any],
        s3,
        output_bucket: str,
        presigned_output_urls: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        """Run inference on a single image.

        Args:
            model: The object returned by ``load()``.
            image: PIL RGB image already downloaded from S3.
            bucket: Source S3 bucket.
            key: Source S3 object key.
            config: Same config dict as ``load()`` — may contain per-request
                    parameters (e.g. ``mode``, ``prompt``).
            s3: boto3 S3 client for uploading results. May be ``None`` when
                presigned output URLs are provided.
            output_bucket: Bucket to write output artifacts to.
            presigned_output_urls: Optional dict mapping output_key ->
                    presigned PUT URL.  When provided, adapters should upload
                    results via HTTP PUT instead of ``s3.put_object()``.

        Returns:
            JSON-serialisable result dict.
        """

    @abc.abstractmethod
    def cache_key(self, config: Dict[str, Any]) -> str:
        """Return a stable string key that uniquely identifies the loaded model.

        Two configs that would produce the same in-memory model **must**
        return the same cache key.
        """
