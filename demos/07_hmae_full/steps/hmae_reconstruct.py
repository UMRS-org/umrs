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

from umrs.steps import UMRSStep, StepContext, StepResult

logger = logging.getLogger(__name__)


class HMAEReconstructStep(UMRSStep):
    """Full reconstruction (no masking) via BentoML HMAE endpoint."""

    name = "hmae_reconstruct"

    def setup(self, config: Dict[str, Any]) -> None:
        self.bentoml_url = config["bentoml_url"].rstrip("/")
        self.timeout = config.get("timeout", 300)
        self.poll_interval = config.get("poll_interval", 2.0)

    def run(self, ctx: StepContext) -> StepResult:
        import requests

        session = requests.Session()
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
            result = self._submit_and_poll(session, image_url)

            base = key.rsplit(".", 1)[0] if "." in key else key
            meta = {
                "source_key": key,
                "source_bucket": ctx.input_bucket,
                "mode": "reconstruct",
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
            extra={"total_images": len(ctx.input_keys), "mode": "reconstruct"},
        )

    def _submit_and_poll(self, session, image_url: str) -> Dict[str, Any]:
        """Submit HMAE reconstruct task and poll until done."""
        try:
            resp = session.post(
                f"{self.bentoml_url}/infer/submit",
                json={
                    "model_type": "hmae",
                    "model_config": {"mode": "reconstruct"},
                    "image_url": image_url,
                },
                timeout=30,
            )
            resp.raise_for_status()
            task_id = resp.json().get("task_id") or resp.json().get("id")
        except Exception as exc:
            return {"error": str(exc)}

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                resp = session.get(
                    f"{self.bentoml_url}/infer/status",
                    params={"task_id": task_id},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status") or data.get("value", "")

                if status in ("success", "completed"):
                    rr = session.get(
                        f"{self.bentoml_url}/infer/get",
                        params={"task_id": task_id},
                        timeout=30,
                    )
                    rr.raise_for_status()
                    return rr.json()

                if status in ("failure", "failed"):
                    return {"error": f"Reconstruct task {task_id} failed"}
            except Exception as exc:
                return {"error": str(exc)}

            time.sleep(self.poll_interval)

        return {"error": f"Reconstruct task timed out after {self.timeout}s"}
