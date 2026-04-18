"""
Step 2: HMAE Masked Reconstruction — 50 %% random masking + anomaly scoring.

Reads the per-image metadata JSONs produced by Step 1 (or the shared
``ctx.metadata["original_keys"]``) to locate the original images, then
submits each to BentoML with ``mode=mae``.

Input keys:  ``reconstruct/<basename>_recon.json`` (from Step 1)
Output keys: ``mae/<basename>_mae.json`` per image

Each result JSON contains:
  - ``source_key``: original image S3 key
  - ``mae_loss``: reconstruction loss (float)
  - ``is_anomaly``: whether loss exceeds threshold
  - ``output_s3_key``: visualization PNG (original / mask / reconstruction)

Config:
  - ``bentoml_url``: BentoML vision service URL
  - ``anomaly_threshold``: mae_loss above this → anomaly (default 0.3)
  - ``timeout``: per-task timeout in seconds (default 300)
  - ``poll_interval``: polling interval in seconds (default 2.0)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from umrs.pipeline_sdk import Step, StepContext, StepResult
from umrs.vision_service.bentoml_client import submit_vision_and_wait

logger = logging.getLogger(__name__)


class HMAEMaeStep(Step):
    """Masked reconstruction + anomaly scoring via BentoML HMAE endpoint."""

    name = "hmae_mae"

    def setup(self, config: Dict[str, Any]) -> None:
        self.service_url = config["bentoml_url"].rstrip("/")
        self.anomaly_threshold = config.get("anomaly_threshold", 0.3)
        self.timeout = config.get("timeout", 300)
        self.poll_interval = config.get("poll_interval", 2.0)

    def run(self, ctx: StepContext) -> StepResult:
        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []
        anomaly_count = 0

        # Resolve original image keys — prefer shared metadata, fall back
        # to parsing the JSON files produced by the previous step.
        original_keys = ctx.metadata.get("original_keys")
        original_bucket = ctx.metadata.get("original_bucket", ctx.input_bucket)

        if not original_keys:
            original_keys = self._resolve_source_keys(ctx)

        for idx, key in enumerate(original_keys):
            logger.info(
                "[%d/%d] HMAE mae: %s",
                idx + 1, len(original_keys), key,
            )

            image_url = f"s3://{original_bucket}/{key}"
            result = self._submit_and_poll(image_url)

            mae_loss = float(result.get("mae_loss", -1.0))
            is_anomaly = (
                mae_loss > self.anomaly_threshold if mae_loss >= 0 else False
            )
            if is_anomaly:
                anomaly_count += 1

            base = key.rsplit(".", 1)[0] if "." in key else key
            meta = {
                "source_key": key,
                "source_bucket": original_bucket,
                "task": "mae_reconstruction",
                "mae_loss": mae_loss,
                "anomaly_threshold": self.anomaly_threshold,
                "is_anomaly": is_anomaly,
                "output_s3_key": result.get("output_s3_key", ""),
                "error": result.get("error"),
            }

            meta_key = f"mae/{base}_mae.json"
            ctx.upload_bytes(
                meta_key,
                json.dumps(meta, indent=2).encode(),
                content_type="application/json",
            )
            output_keys.append(meta_key)
            metadata_list.append(meta)
            checksums.append("")

            status_str = "ANOMALY" if is_anomaly else "normal"
            if "error" in result:
                logger.warning("  mae failed: %s", result["error"])
            else:
                logger.info(
                    "  → mae_loss=%.4f [%s], output: %s",
                    mae_loss, status_str, meta["output_s3_key"],
                )

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
            extra={
                "total_images": len(original_keys),
                "task": "mae_reconstruction",
                "anomaly_count": anomaly_count,
                "anomaly_rate": (
                    anomaly_count / len(original_keys)
                    if original_keys else 0.0
                ),
            },
        )

    # ------------------------------------------------------------------

    def _resolve_source_keys(self, ctx: StepContext) -> List[str]:
        """Parse Step 1 output JSONs to find original image keys."""
        keys = []
        for meta_key in ctx.input_keys:
            try:
                raw = ctx.download_bytes(meta_key)
                meta = json.loads(raw)
                keys.append(meta["source_key"])
            except Exception as exc:
                logger.warning("Cannot parse %s: %s", meta_key, exc)
        return keys

    def _submit_and_poll(self, image_url: str) -> Dict[str, Any]:
        """Submit HMAE mae task and poll until done."""
        try:
            return submit_vision_and_wait(
                self.service_url,
                "hmae",
                image_url,
                inference_config={"task": "mae_reconstruction"},
                poll_interval=self.poll_interval,
                timeout=self.timeout,
            )
        except Exception as exc:
            return {"error": str(exc)}
