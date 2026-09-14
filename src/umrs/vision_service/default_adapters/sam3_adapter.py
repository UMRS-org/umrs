"""SAM3 segmentation adapter for the generic vision service."""

from __future__ import annotations

import io
import logging
import os
import pickle
import urllib.request
from typing import Any

from PIL import Image

from umrs.vision_service.adapters.base import VisionModelAdapter
from umrs.vision_service.output_specs import describe_output_artifacts

logger = logging.getLogger(__name__)


def _require_presigned_urls() -> bool:
    return os.environ.get("VISION_REQUIRE_PRESIGNED_URLS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _include_full_masks_in_results() -> bool:
    return os.environ.get("SAM3_INCLUDE_MASKS_IN_RESULTS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _build_results_payload(
    result: dict[str, Any], prompt: str, source_image: str
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "boxes": result["boxes"],
        "scores": result["scores"],
        "num_objects": result["num_objects"],
        "prompt": prompt,
        "source_image": source_image,
        "mask_shapes": [tuple(mask.shape) for mask in result["masks"]],
        "contains_full_masks": False,
    }

    if _include_full_masks_in_results():
        payload["masks"] = result["masks"]
        payload["contains_full_masks"] = True

    return payload


class SAM3Adapter(VisionModelAdapter):
    """Load and run SAM3 text-prompted segmentation."""

    def capabilities(self):
        return {
            "name": "sam3",
            "description": "SAM3 text-prompted image segmentation",
            "supported_tasks": ["image-segmentation", "custom:segmentation"],
            "supported_inputs": ["image_with_text", "single_image"],
            "output_types": ["masks", "bounding_boxes", "overlay_image", "scores"],
            "gpu_memory_mb": 6000,
            "supports_batch": False,
            "max_batch_size": 1,
            "typical_inference_seconds": 3.0,
            "model_source": "facebook/sam3",
            "framework": "pytorch",
        }

    def estimate_gpu_memory_mb(self, config: dict[str, Any]) -> int:
        checkpoint = config.get(
            "checkpoint", os.environ.get("SAM3_MODEL_PATH", "facebook/sam3")
        )
        if "large" in checkpoint.lower():
            return 8000
        return 6000

    def describe_outputs(self, key: str, config: dict[str, Any]) -> dict[str, str]:
        return describe_output_artifacts("sam3", key, config)

    def cache_key(self, config: dict[str, Any]) -> str:
        checkpoint = config.get(
            "checkpoint", os.environ.get("SAM3_MODEL_PATH", "facebook/sam3")
        )
        return f"sam3:{checkpoint}"

    def load(self, config: dict[str, Any]) -> Any:
        from umrs.inference.sam3_segmentation import load_sam3_model

        checkpoint = config.get(
            "checkpoint", os.environ.get("SAM3_MODEL_PATH", "facebook/sam3")
        )
        model, processor, device = load_sam3_model(checkpoint)
        return {"model": model, "processor": processor, "device": device}

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
        del raw_image_bytes

        from umrs.inference.sam3_segmentation import (
            overlay_masks_on_image,
            segment_image_with_text,
        )

        prompt = config.get("prompt", "object")
        threshold = config.get("threshold", 0.5)
        mask_threshold = config.get("mask_threshold", 0.5)
        device = model["device"]

        logger.debug(
            "SAM3 infer: bucket=%s key=%s prompt=%s device=%s "
            "output_bucket=%s presigned_output_urls=%s",
            bucket,
            key,
            prompt,
            device,
            output_bucket,
            list(presigned_output_urls.keys()) if presigned_output_urls else "(empty)",
        )

        result = segment_image_with_text(
            model["model"],
            model["processor"],
            image,
            prompt,
            device,
            threshold,
            mask_threshold,
        )

        response: dict[str, Any] = {
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

            logger.debug(
                "SAM3 uploading overlay: key=%s dst_bucket=%s s3=%s presigned=%s",
                overlay_key,
                dst_bucket,
                s3 is not None,
                overlay_key in (presigned_output_urls or {}),
            )

            # Prefer presigned PUT URL when available
            presigned_url = (presigned_output_urls or {}).get(overlay_key)
            if presigned_url:
                req = urllib.request.Request(
                    presigned_url,
                    data=png_data,
                    method="PUT",
                    headers={"Content-Type": "image/png"},
                )
                urllib.request.urlopen(req, timeout=120)
            else:
                if _require_presigned_urls() or s3 is None:
                    raise RuntimeError(f"Missing presigned PUT URL for {overlay_key}")
                s3.put_object(
                    Bucket=dst_bucket,
                    Key=overlay_key,
                    Body=png_data,
                    ContentType="image/png",
                )
            response["overlay_s3_key"] = f"s3://{dst_bucket}/{overlay_key}"

        # Upload a compact result payload by default. Full-resolution masks make
        # demo downloads multi-gigabyte on large images, so keep them opt-in.
        described = self.describe_outputs(key, config)
        pkl_key = next(k for k in described if k.endswith(".pkl"))
        dst_bucket = output_bucket or bucket
        pkl_data = pickle.dumps(
            _build_results_payload(result, prompt, response["source_image"]),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        presigned_url = (presigned_output_urls or {}).get(pkl_key)
        if presigned_url:
            req = urllib.request.Request(
                presigned_url,
                data=pkl_data,
                method="PUT",
                headers={"Content-Type": "application/octet-stream"},
            )
            urllib.request.urlopen(req, timeout=120)
        else:
            if _require_presigned_urls() or s3 is None:
                raise RuntimeError(f"Missing presigned PUT URL for {pkl_key}")
            s3.put_object(
                Bucket=dst_bucket,
                Key=pkl_key,
                Body=pkl_data,
                ContentType="application/octet-stream",
            )
        response["results_pkl_s3_key"] = f"s3://{dst_bucket}/{pkl_key}"

        logger.info(
            "SAM3 segmentation: found %d objects for prompt '%s'",
            result["num_objects"],
            prompt,
        )
        return response
