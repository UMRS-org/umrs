"""SAM3 segmentation adapter for the generic vision service."""

from __future__ import annotations

import io
import logging
import os
import urllib.request
from typing import Any, Dict

from PIL import Image

from umrs.vision_service.adapters.base import VisionModelAdapter
from umrs.vision_service.output_specs import describe_output_artifacts

logger = logging.getLogger(__name__)


def _require_presigned_urls() -> bool:
    return os.environ.get("VISION_REQUIRE_PRESIGNED_URLS", "").lower() in {
        "1", "true", "yes", "on",
    }


class SAM3Adapter(VisionModelAdapter):
    """Load and run SAM3 text-prompted segmentation."""

    def describe_outputs(self, key: str, config: Dict[str, Any]) -> Dict[str, str]:
        return describe_output_artifacts("sam3", key, config)

    def cache_key(self, config: Dict[str, Any]) -> str:
        checkpoint = config.get("checkpoint",
                                os.environ.get("SAM3_MODEL_PATH", "facebook/sam3"))
        return f"sam3:{checkpoint}"

    def load(self, config: Dict[str, Any]) -> Any:
        from umrs.inference.sam3_segmentation import load_sam3_model

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
        presigned_output_urls: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        from umrs.inference.sam3_segmentation import (
            segment_image_with_text,
            overlay_masks_on_image,
        )

        prompt = config.get("prompt", "object")
        threshold = config.get("threshold", 0.5)
        mask_threshold = config.get("mask_threshold", 0.5)
        device = model["device"]

        logger.warning("[DEBUG-SAM3] infer: bucket=%s key=%s prompt=%s device=%s "
                       "output_bucket=%s presigned_output_urls=%s",
                       bucket, key, prompt, device, output_bucket,
                       list(presigned_output_urls.keys()) if presigned_output_urls else "(empty)")

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
            overlay_key = next(iter(self.describe_outputs(key, config)))
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            buf.seek(0)
            dst_bucket = output_bucket or bucket
            png_data = buf.getvalue()

            logger.warning("[DEBUG-SAM3] Uploading overlay: key=%s dst_bucket=%s "
                           "s3=%s presigned=%s",
                           overlay_key, dst_bucket, s3 is not None,
                           overlay_key in (presigned_output_urls or {}))

            # Prefer presigned PUT URL when available
            presigned_url = (presigned_output_urls or {}).get(overlay_key)
            if presigned_url:
                req = urllib.request.Request(
                    presigned_url,
                    data=png_data,
                    method="PUT",
                    headers={"Content-Type": "image/png"},
                )
                urllib.request.urlopen(req, timeout=120)  # noqa: S310
            else:
                if _require_presigned_urls() or s3 is None:
                    raise RuntimeError(
                        f"Missing presigned PUT URL for {overlay_key}"
                    )
                s3.put_object(Bucket=dst_bucket, Key=overlay_key,
                              Body=png_data, ContentType="image/png")
            response["overlay_s3_key"] = f"s3://{dst_bucket}/{overlay_key}"

        logger.info("SAM3 segmentation: found %d objects for prompt '%s'",
                     result["num_objects"], prompt)
        return response
