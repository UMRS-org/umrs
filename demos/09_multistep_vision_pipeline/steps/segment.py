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

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from umrs.pipeline_sdk import Step, StepContext, StepResult

logger = logging.getLogger(__name__)

_HELPER_MODULE_NAME = "_umrs_demo09_rivault_vision"
_helper = sys.modules.get(_HELPER_MODULE_NAME)
if _helper is None:
    _helper_path = Path(__file__).with_name("_rivault_vision.py")
    _helper_spec = importlib.util.spec_from_file_location(
        _HELPER_MODULE_NAME,
        _helper_path,
    )
    if _helper_spec is None or _helper_spec.loader is None:
        raise ImportError(f"Cannot load helper module from {_helper_path}")
    _helper = importlib.util.module_from_spec(_helper_spec)
    sys.modules[_HELPER_MODULE_NAME] = _helper
    _helper_spec.loader.exec_module(_helper)

get_rivault_model = _helper.get_rivault_model


class SegmentStep(Step):
    """Detect regions via a lazily launched RiVault vision model."""

    name = "segment"

    def setup(self, config: Dict[str, Any]) -> None:
        self.model_type = config.get("model_type", "sam3")
        self.prompt = config.get("prompt", "object")
        self.threshold = config.get("threshold", 0.5)
        self._model = None

    def run(self, ctx: StepContext) -> StepResult:
        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []

        for idx, key in enumerate(ctx.input_keys):
            logger.info(
                "[%d/%d] Segmenting: %s (prompt=%r)",
                idx + 1, len(ctx.input_keys), key, self.prompt,
            )

            result, checksum = self._run_inference(ctx, key)

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
            checksums.append(checksum)

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

    def _get_model(self):
        if self._model is None:
            self._model = get_rivault_model(
                self.model_type,
                model_name="SAM3 Segmentation",
                model_config={
                    "prompt": self.prompt,
                    "threshold": self.threshold,
                },
            )
        return self._model

    def _run_inference(self, ctx: StepContext, key: str) -> tuple[Dict[str, Any], str]:
        try:
            results, checksums, _, _ = self._get_model().compute_embeddings(
                ctx.s3,
                [key],
                ctx.input_bucket,
                ctx.output_bucket,
                skip_existing=False,
            )
        except Exception as exc:
            logger.exception("Segmentation failed for %s", key)
            return {"error": str(exc)}, ""

        result = results[0] if results else {}
        if not isinstance(result, dict):
            result = {"raw": str(result)}
        checksum = checksums[0] if checksums else ""
        return result, checksum
