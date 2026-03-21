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
import time
from typing import Any, Dict, List

from umrs.steps import UMRSStep, StepContext, StepResult

logger = logging.getLogger(__name__)


class SAM3SegmentationStep(UMRSStep):
    """Detect structural regions via SAM3 text-prompted segmentation."""

    name = "sam3_segment"

    def setup(self, config: Dict[str, Any]) -> None:
        self.bentoml_url = config["bentoml_url"].rstrip("/")
        self.prompt = config.get("prompt", "defect")
        self.threshold = config.get("threshold", 0.5)
        self.mask_threshold = config.get("mask_threshold", 0.5)
        self.poll_interval = config.get("poll_interval", 2.0)
        self.timeout = config.get("timeout", 300)

    def run(self, ctx: StepContext) -> StepResult:
        import requests

        session = requests.Session()
        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []

        for idx, key in enumerate(ctx.input_keys):
            logger.info(
                "[%d/%d] SAM3 segmenting: %s (prompt=%r)",
                idx + 1, len(ctx.input_keys), key, self.prompt,
            )

            image_url = f"s3://{ctx.input_bucket}/{key}"

            # Submit BentoML task
            resp = session.post(
                f"{self.bentoml_url}/infer/submit",
                json={
                    "model_type": "sam3",
                    "model_config": {
                        "prompt": self.prompt,
                        "threshold": self.threshold,
                        "mask_threshold": self.mask_threshold,
                    },
                    "image_url": image_url,
                },
                timeout=30,
            )
            resp.raise_for_status()
            task_id = resp.json().get("task_id") or resp.json().get("id")

            # Poll for completion
            result = self._poll_result(session, task_id)

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

    def _poll_result(self, session, task_id: str) -> Dict[str, Any]:
        """Poll BentoML task queue until completion."""
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            resp = session.get(
                f"{self.bentoml_url}/infer/status",
                params={"task_id": task_id},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status") or data.get("value", "")

            if status in ("success", "completed"):
                result_resp = session.get(
                    f"{self.bentoml_url}/infer/get",
                    params={"task_id": task_id},
                    timeout=30,
                )
                result_resp.raise_for_status()
                return result_resp.json()

            if status in ("failure", "failed"):
                return {"error": f"SAM3 task {task_id} failed"}

            time.sleep(self.poll_interval)

        return {"error": f"SAM3 task {task_id} timed out after {self.timeout}s"}
