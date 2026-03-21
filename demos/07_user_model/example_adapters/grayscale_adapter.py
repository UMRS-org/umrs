"""Example custom VisionModelAdapter — grayscale conversion.

Demonstrates how to write a user-provided adapter for the BentoML
vision service.  This adapter converts images to grayscale and
uploads the result.

Usage:
  1. Place this file (and optional requirements.txt) in a directory
  2. Set USER_ADAPTERS_DIR to that directory before starting the service
  3. Submit inference with model_type="grayscale_adapter"
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict

from PIL import Image

from umrs.vision_service.adapters.base import VisionModelAdapter

logger = logging.getLogger(__name__)


class GrayscaleAdapter(VisionModelAdapter):
    """Convert images to grayscale — minimal example adapter."""

    def cache_key(self, config: Dict[str, Any]) -> str:
        return "grayscale:default"

    def load(self, config: Dict[str, Any]) -> Any:
        # No model weights to load — this is a pure image-processing adapter
        logger.info("GrayscaleAdapter loaded (no model weights needed)")
        return {"name": "grayscale"}

    def infer(
        self,
        model: Any,
        image: Image.Image,
        bucket: str,
        key: str,
        config: Dict[str, Any],
        s3,
        output_bucket: str,
    ) -> Dict[str, Any]:
        # Convert to grayscale
        gray = image.convert("L")

        # Upload result
        base = key.rsplit(".", 1)[0] if "." in key else key
        out_key = f"{base}_grayscale.png"
        dst_bucket = output_bucket or bucket

        buf = io.BytesIO()
        gray.save(buf, format="PNG")
        buf.seek(0)
        s3.put_object(
            Bucket=dst_bucket,
            Key=out_key,
            Body=buf.getvalue(),
            ContentType="image/png",
        )

        logger.info("Grayscale conversion done for s3://%s/%s", bucket, key)
        return {
            "source_image": f"s3://{bucket}/{key}",
            "output_s3_key": f"s3://{dst_bucket}/{out_key}",
            "mode": "grayscale",
        }
