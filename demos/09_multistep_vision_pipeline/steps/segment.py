"""
Step 1: Segmentation — detect objects/regions in images.

Submits each input image to the BentoML vision model task-queue endpoint,
polls for completion, and writes segmentation metadata (bounding boxes,
scores) as JSON to S3.

This step demonstrates the BentoML task-queue pattern used by all UMRS
vision models.  Replace `model_type` and `inference_config` to use a
different model.

Output keys:  ``<basename>_segment_meta.json`` per input image

Config:
  - ``bentoml_url``: BentoML vision service URL
  - ``prompt``:      text prompt for the model (e.g. "object", "crack")
  - ``threshold``:   confidence threshold (default 0.5)
  - ``timeout``:     per-task timeout in seconds (default 300)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

import bentoml

from umrs.pipeline_sdk import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class SegmentStep(Step):
    """Detect regions via BentoML vision model."""

    name = "segment"

    def setup(self, config: Dict[str, Any]) -> None:
        self.client = bentoml.SyncHTTPClient(
            config["bentoml_url"].rstrip("/"),
            server_ready_timeout=60,
        )
        self.prompt = config.get("prompt", "object")
        self.threshold = config.get("threshold", 0.5)
        self.poll_interval = config.get("poll_interval", 2.0)
        self.timeout = config.get("timeout", 300)

    def run(self, ctx: StepContext) -> StepResult:
        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []

        for idx, key in enumerate(ctx.input_keys):
            logger.info(
                "[%d/%d] Segmenting: %s (prompt=%r)",
                idx + 1, len(ctx.input_keys), key, self.prompt,
            )

            image_url = f"s3://{ctx.input_bucket}/{key}"
            result = self._submit_and_poll(image_url)

            base = key.rsplit(".", 1)[0] if "." in key else key
            meta = {
                "source_key": key,
                "source_bucket": ctx.input_bucket,
                "num_objects": result.get("num_objects", 0),
                "boxes": result.get("boxes", []),
                "scores": result.get("scores", []),
                "prompt": self.prompt,
                "error": result.get("error", ""),
            }

            meta_key = f"{base}_segment_meta.json"
            ctx.upload_bytes(
                meta_key,
                json.dumps(meta, indent=2).encode(),
                content_type="application/json",
            )
            output_keys.append(meta_key)
            metadata_list.append(meta)

            try:
                head = ctx.head_object(key)
                checksums.append(head.get("ETag", "").strip('"'))
            except Exception:
                checksums.append("")

            logger.info("  → %d objects detected", meta["num_objects"])

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
        """Submit vision task and poll until completion."""
        try:
            task = self.client.infer.submit(
                model_type="sam3",
                inference_config={
                    "prompt": self.prompt,
                    "threshold": self.threshold,
                },
                image_url=image_url,
            )
        except Exception as exc:
            return {"error": str(exc)}

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                status = task.get_status()
                if status.value == "success":
                    return task.get()
                if status.value in ("failure", "cancelled"):
                    return {"error": f"Task {task.id} {status.value}"}
            except Exception as exc:
                return {"error": str(exc)}
            time.sleep(self.poll_interval)

        return {"error": f"Task timed out after {self.timeout}s"}
