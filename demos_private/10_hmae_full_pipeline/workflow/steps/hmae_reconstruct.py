"""
Step 1: HMAE Full Reconstruction — no masking, full encode + decode.

Submits each input image to the BentoML HMAE endpoint with ``mode=reconstruct``,
polls for completion, and writes per-image metadata JSON to S3.

Input keys:  raw image S3 keys (e.g. ``image1.png``)
Output keys: ``reconstruct/<basename>_recon.json`` per image

Each result JSON contains:
  - ``source_key``: original image S3 key
  - ``source_bucket``: bucket of the original image
  - ``output_s3_key``: S3 key/URL of the reconstruction PNG
  - ``mode``: ``"reconstruct"``

Also stores ``ctx.metadata["original_keys"]`` and
``ctx.metadata["original_bucket"]`` for downstream steps.

Config:
  - ``bentoml_url``: BentoML vision service URL
  - ``timeout``: per-task timeout in seconds (default 300)
  - ``poll_interval``: polling interval in seconds (default 2.0)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from umrs.pipeline_sdk import Step, StepContext, StepResult
from umrs.vision_service.bentoml_client import submit_vision_and_wait

from _hmae_runtime import resolve_bentoml_url, resolve_input_images

logger = logging.getLogger(__name__)


class HMAEReconstructStep(Step):
    """Full reconstruction (no masking) via BentoML HMAE endpoint."""

    name = "hmae_reconstruct"

    def setup(self, config: Dict[str, Any]) -> None:
        self.service_url = resolve_bentoml_url(config)
        self.timeout = config.get("timeout", 300)
        self.poll_interval = config.get("poll_interval", 2.0)

    def run(self, ctx: StepContext) -> StepResult:
        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []
        input_images = resolve_input_images(ctx)

        # Store original keys for downstream steps
        ctx.metadata["original_keys"] = [image.key for image in input_images]
        if input_images:
            ctx.metadata["original_bucket"] = input_images[0].bucket

        for idx, image in enumerate(input_images):
            key = image.key
            logger.info(
                "[%d/%d] HMAE reconstruct: %s",
                idx + 1,
                len(input_images),
                key,
            )

            result = self._submit_and_poll(image.s3_url)

            base = key.rsplit(".", 1)[0] if "." in key else key
            meta = {
                "source_key": key,
                "source_bucket": image.bucket,
                "task": "reconstruction",
                "output_s3_key": result.get("output_s3_key", ""),
                "error": result.get("error"),
            }

            meta_key = f"reconstruct/{base}_recon.json"
            ctx.upload_bytes(
                meta_key,
                json.dumps(meta, indent=2).encode(),
                content_type="application/json",
            )
            output_keys.append(meta_key)
            metadata_list.append(meta)

            try:
                head = ctx.head_object(key, bucket=image.bucket)
                checksums.append(head.get("ETag", "").strip('"'))
            except Exception:
                checksums.append("")

            if "error" in result:
                logger.warning("  reconstruct failed: %s", result["error"])
            else:
                logger.info("  → output: %s", meta["output_s3_key"])

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
            extra={
                "total_images": len(input_images),
                "task": "reconstruction",
                "items": metadata_list,
            },
        )

    def _submit_and_poll(self, image_url: str) -> Dict[str, Any]:
        """Submit HMAE reconstruct task and poll until done."""
        try:
            return submit_vision_and_wait(
                self.service_url,
                "hmae",
                image_url,
                inference_config={"task": "reconstruction"},
                poll_interval=self.poll_interval,
                timeout=self.timeout,
            )
        except Exception as exc:
            return {"error": str(exc)}
