"""SAM3 segmentation adapter for the generic vision service."""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict

import torch
from PIL import Image

from .base import VisionModelAdapter

logger = logging.getLogger(__name__)


class SAM3Adapter(VisionModelAdapter):
    """Load and run SAM3 text-prompted segmentation."""

    def cache_key(self, config: Dict[str, Any]) -> str:
        checkpoint = config.get("checkpoint",
                                os.environ.get("SAM3_MODEL_PATH", "facebook/sam3"))
        return f"sam3:{checkpoint}"

    def load(self, config: Dict[str, Any]) -> Any:
        from umrs.models.sam3_segmentation import load_sam3_model

        checkpoint = config.get("checkpoint",
                                os.environ.get("SAM3_MODEL_PATH", "facebook/sam3"))
        model, processor, device = load_sam3_model(checkpoint)
        return {"model": model, "processor": processor, "device": device}

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
        from umrs.models.sam3_segmentation import (
            segment_image_with_text,
            overlay_masks_on_image,
        )

        prompt = config.get("prompt", "object")
        threshold = config.get("threshold", 0.5)
        mask_threshold = config.get("mask_threshold", 0.5)
        device = model["device"]

        result = segment_image_with_text(
            model["model"], model["processor"], image, prompt,
            device, threshold, mask_threshold,
        )

        response: Dict[str, Any] = {
            "num_objects": result["num_objects"],
            "boxes": result["boxes"],
            "scores": result["scores"],
            "source_image": f"s3://{bucket}/{key}",
            "prompt": prompt,
        }

        # Upload overlay visualisation
        if result["num_objects"] > 0:
            overlay = overlay_masks_on_image(image, result["masks"])
            base = key.rsplit(".", 1)[0] if "." in key else key
            overlay_key = f"{base}_sam3_seg.png"
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            buf.seek(0)
            dst_bucket = output_bucket or bucket
            s3.put_object(Bucket=dst_bucket, Key=overlay_key,
                          Body=buf.getvalue(), ContentType="image/png")
            response["overlay_s3_key"] = f"s3://{dst_bucket}/{overlay_key}"

        logger.info("SAM3 segmentation: found %d objects for prompt '%s'",
                     result["num_objects"], prompt)
        return response
