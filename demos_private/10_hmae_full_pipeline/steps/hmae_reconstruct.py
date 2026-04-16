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
import time
from typing import Any, Dict, List

import bentoml

from umrs.pipeline_sdk import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class HMAEReconstructStep(Step):
    """Full reconstruction (no masking) via BentoML HMAE endpoint."""

    name = "hmae_reconstruct"

    def setup(self, config: Dict[str, Any]) -> None:
        self.client = bentoml.SyncHTTPClient(
            config["bentoml_url"].rstrip("/"),
            server_ready_timeout=60,
        )
        self.timeout = config.get("timeout", 300)
        self.poll_interval = config.get("poll_interval", 2.0)

    def run(self, ctx: StepContext) -> StepResult:
        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []

        # Store original keys for downstream steps
        ctx.metadata["original_keys"] = list(ctx.input_keys)
        ctx.metadata["original_bucket"] = ctx.input_bucket

        for idx, key in enumerate(ctx.input_keys):
            logger.info(
                "[%d/%d] HMAE reconstruct: %s",
                idx + 1, len(ctx.input_keys), key,
            )

            image_url = f"s3://{ctx.input_bucket}/{key}"
            result = self._submit_and_poll(image_url)

            base = key.rsplit(".", 1)[0] if "." in key else key
            meta = {
                "source_key": key,
                "source_bucket": ctx.input_bucket,
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
                head = ctx.head_object(key)
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
            extra={"total_images": len(ctx.input_keys), "task": "reconstruction"},
        )

    def _submit_and_poll(self, image_url: str) -> Dict[str, Any]:
        """Submit HMAE reconstruct task and poll until done."""
        try:
            task = self.client.infer.submit(
                model_type="hmae",
                inference_config={"task": "reconstruction"},
                image_url=image_url,
            )
        except Exception as exc:
            return {"error": str(exc)}

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                status = task.get_status()
                if status.value == "success":
                    raw = task.get()
                    if isinstance(raw, bytes):
                        import pickle
                        return pickle.loads(raw)  # noqa: S301
                    return raw
                if status.value in ("failure", "cancelled"):
                    return {"error": f"Reconstruct task {task.id} {status.value}"}
            except Exception as exc:
                return {"error": str(exc)}
            time.sleep(self.poll_interval)

        return {"error": f"Reconstruct task timed out after {self.timeout}s"}
