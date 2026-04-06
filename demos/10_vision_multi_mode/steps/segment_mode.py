"""
Segment Mode — object detection / segmentation via BentoML.

Identical to demo 09's SegmentStep but packaged as an independent
parallel branch.  Outputs per-image JSON metadata with bounding boxes
and confidence scores.

Config:
  - ``bentoml_url``:  BentoML vision service base URL
  - ``model_type``:   vision model to invoke (default "sam3")
  - ``prompt``:       text prompt (default "object")
  - ``threshold``:    confidence threshold (default 0.5)
  - ``timeout``:      per-task timeout (default 300)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

import bentoml

from umrs.pipeline_sdk import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class SegmentModeStep(Step):
    """Run segmentation model for object detection."""

    name = "segment_mode"

    def setup(self, config: Dict[str, Any]) -> None:
        self.client = bentoml.SyncHTTPClient(
            config["bentoml_url"].rstrip("/"),
            server_ready_timeout=60,
        )
        self.model_type = config.get("model_type", "sam3")
        self.prompt = config.get("prompt", "object")
        self.threshold = config.get("threshold", 0.5)
        self.poll_interval = config.get("poll_interval", 2.0)
        self.timeout = config.get("timeout", 300)

    def run(self, ctx: StepContext) -> StepResult:
        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []

        for idx, key in enumerate(ctx.input_keys):
            logger.info("[%d/%d] Segment mode: %s", idx + 1, len(ctx.input_keys), key)

            image_url = f"s3://{ctx.input_bucket}/{key}"
            task = self.client.infer.submit(
                model_type=self.model_type,
                inference_config={"prompt": self.prompt, "threshold": self.threshold},
                image_url=image_url,
            )

            deadline = time.time() + self.timeout
            while time.time() < deadline:
                status = task.get_status()
                if status.state in ("SUCCESS", "FAILED"):
                    break
                time.sleep(self.poll_interval)

            result = {}
            if status.state == "SUCCESS":
                raw = task.get()
                result = raw if isinstance(raw, dict) else {"raw": str(raw)}
            else:
                result = {"error": f"Task {status.state} after {self.timeout}s"}

            base = key.rsplit(".", 1)[0] if "." in key else key
            meta = {
                "source_key": key,
                "mode": "segment",
                "model_type": self.model_type,
                "num_objects": result.get("num_objects", 0),
                "boxes": result.get("boxes", []),
                "scores": result.get("scores", []),
                "error": result.get("error", ""),
            }

            out_key = f"segment/{base}_segment.json"
            ctx.upload_bytes(
                out_key,
                json.dumps(meta, indent=2).encode(),
                content_type="application/json",
            )
            output_keys.append(out_key)
            metadata_list.append(meta)
            checksums.append("")

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
            extra={"mode": "segment", "total_images": len(ctx.input_keys)},
        )
