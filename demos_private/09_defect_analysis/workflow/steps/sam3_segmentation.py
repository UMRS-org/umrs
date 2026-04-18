"""
Step 1: SAM3 Segmentation — detect structural regions in materials images.

Submits each input image to the BentoML SAM3 task-queue endpoint, polls for
completion, and writes the segmentation results (bounding boxes, masks, scores)
as JSON metadata + the overlay visualization to S3.

Output keys:  one ``<basename>_sam3_meta.json`` per input image, containing
  - ``boxes``: list of [x1, y1, x2, y2] bounding boxes
  - ``scores``: confidence per box
  - ``source_key``: original S3 key
  - ``overlay_key``: S3 key for the overlay PNG
  - ``num_objects``: number of detected segments

Config (workflow.json → step config):
  - ``bentoml_url``: BentoML vision service URL (e.g. ``http://localhost:11434/vision``)
  - ``prompt``: text prompt for SAM3 (e.g. ``"crack"``, ``"void"``, ``"defect"``)
  - ``threshold``: confidence threshold (default 0.5)
  - ``mask_threshold``: mask binarization threshold (default 0.5)
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, List

import numpy as np

from umrs.pipeline_sdk import Step, StepContext, StepResult
from umrs.vision_service.bentoml_client import submit_vision_and_wait

logger = logging.getLogger(__name__)


class SAM3SegmentationStep(Step):
    """Detect structural regions via SAM3 text-prompted segmentation."""

    name = "sam3_segment"

    def setup(self, config: Dict[str, Any]) -> None:
        self.service_url = config["bentoml_url"].rstrip("/")
        self.prompt = config.get("prompt", "defect")
        self.threshold = config.get("threshold", 0.5)
        self.mask_threshold = config.get("mask_threshold", 0.5)
        self.poll_interval = config.get("poll_interval", 2.0)
        self.timeout = config.get("timeout", 300)

    def run(self, ctx: StepContext) -> StepResult:
        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []

        for idx, key in enumerate(ctx.input_keys):
            logger.info(
                "[%d/%d] SAM3 segmenting: %s (prompt=%r)",
                idx + 1, len(ctx.input_keys), key, self.prompt,
            )

            image_url = f"s3://{ctx.input_bucket}/{key}"

            # Submit BentoML task and poll
            result = self._submit_and_poll(image_url)

            if "error" in result:
                logger.warning("SAM3 failed for %s: %s", key, result["error"])
                result.setdefault("num_objects", 0)
                result.setdefault("boxes", [])
                result.setdefault("scores", [])

            # Build per-image metadata JSON
            base = key.rsplit(".", 1)[0] if "." in key else key
            meta = {
                "source_key": key,
                "source_bucket": ctx.input_bucket,
                "num_objects": result.get("num_objects", 0),
                "boxes": result.get("boxes", []),
                "scores": result.get("scores", []),
                "prompt": self.prompt,
                "overlay_key": result.get("overlay_s3_key", ""),
            }

            # Upload metadata JSON to output bucket
            meta_key = f"{base}_sam3_meta.json"
            ctx.upload_bytes(
                meta_key,
                json.dumps(meta, indent=2).encode(),
                content_type="application/json",
            )
            output_keys.append(meta_key)
            metadata_list.append(meta)

            # Save per-object masks as compressed NPZ when available
            masks = result.get("masks")
            if masks and len(masks) > 0:
                npz_buf = io.BytesIO()
                np.savez_compressed(
                    npz_buf,
                    **{f"mask_{i}": m for i, m in enumerate(masks)},
                )
                npz_buf.seek(0)
                masks_key = f"{base}_sam3_masks.npz"
                ctx.upload_bytes(
                    masks_key,
                    npz_buf.getvalue(),
                    content_type="application/octet-stream",
                )
                output_keys.append(masks_key)
                logger.info("  → saved %d masks to %s", len(masks), masks_key)

            # Retrieve checksum of the original image
            try:
                head = ctx.head_object(key)
                checksums.append(head.get("ETag", "").strip('"'))
            except Exception:
                checksums.append("")

            logger.info(
                "  → %d objects detected, overlay=%s",
                meta["num_objects"], meta["overlay_key"],
            )

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
            extra={
                "total_images": len(ctx.input_keys),
                "total_objects": sum(
                    m.get("num_objects", 0) for m in metadata_list
                ),
            },
        )

    def _submit_and_poll(self, image_url: str) -> Dict[str, Any]:
        """Submit SAM3 task and poll until completion."""
        try:
            return submit_vision_and_wait(
                self.service_url,
                "sam3",
                image_url,
                inference_config={
                    "prompt": self.prompt,
                    "threshold": self.threshold,
                    "mask_threshold": self.mask_threshold,
                },
                poll_interval=self.poll_interval,
                timeout=self.timeout,
            )
        except Exception as exc:
            return {"error": str(exc)}
