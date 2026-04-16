"""
Step 4: Anomaly Detection — MAE reconstruction error scoring.

Submits each patch to the BentoML HMAE ``mae_reconstruction`` task endpoint to get a
masked-autoencoder reconstruction.  High reconstruction error indicates
anomalous / unusual structures.

Input keys:  ``patches/<basename>_patch_<i>.png`` (shared with Step 3)
Output keys: ``anomaly/<basename>_patch_<i>_anomaly.json``

Each output JSON contains:
  - ``mae_loss``: float — reconstruction error score
  - ``is_anomaly``: bool — whether the error exceeds the threshold
  - ``mae_output_key``: S3 key for the reconstruction PNG
  - ``source_patch``: original patch key

Config:
  - ``bentoml_url``: BentoML vision service URL
  - ``anomaly_threshold``: reconstruction error above this → anomaly (default 0.3)
  - ``timeout``: per-task timeout (default 300)
  - ``poll_interval``: (default 2.0)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

import bentoml

from umrs.pipeline_sdk import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class AnomalyDetectionStep(Step):
    """Score each patch by HMAE reconstruction error to detect anomalies."""

    name = "anomaly_detect"

    def setup(self, config: Dict[str, Any]) -> None:
        self.client = bentoml.SyncHTTPClient(
            config["bentoml_url"].rstrip("/"),
            server_ready_timeout=60,
        )
        self.anomaly_threshold = config.get("anomaly_threshold", 0.3)
        self.timeout = config.get("timeout", 300)
        self.poll_interval = config.get("poll_interval", 2.0)

    def run(self, ctx: StepContext) -> StepResult:
        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []
        anomaly_count = 0

        for idx, key in enumerate(ctx.input_keys):
            logger.info(
                "[%d/%d] HMAE MAE anomaly: %s",
                idx + 1, len(ctx.input_keys), key,
            )

            image_url = f"s3://{ctx.input_bucket}/{key}"

            # Submit BentoML HMAE mae task and poll
            result = self._submit_and_poll(image_url)

            mae_loss = float(result.get("mae_loss", -1.0))
            is_anomaly = mae_loss > self.anomaly_threshold if mae_loss >= 0 else False

            if is_anomaly:
                anomaly_count += 1

            base = key.rsplit(".", 1)[0] if "." in key else key
            anomaly_info = {
                "source_patch": key,
                "mae_loss": mae_loss,
                "anomaly_threshold": self.anomaly_threshold,
                "is_anomaly": is_anomaly,
                "mae_output_key": result.get("output_s3_key", ""),
                "error": result.get("error", ""),
            }

            out_key = base.replace("patches/", "anomaly/") + "_anomaly.json"
            ctx.upload_bytes(
                out_key,
                json.dumps(anomaly_info, indent=2).encode(),
                content_type="application/json",
            )
            output_keys.append(out_key)
            checksums.append("")
            metadata_list.append(anomaly_info)

            status_str = "ANOMALY" if is_anomaly else "normal"
            logger.info(
                "  → mae_loss=%.4f  [%s]", mae_loss, status_str,
            )

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
            extra={
                "total_patches": len(ctx.input_keys),
                "anomaly_count": anomaly_count,
                "anomaly_rate": (
                    anomaly_count / len(ctx.input_keys)
                    if ctx.input_keys else 0.0
                ),
                "threshold": self.anomaly_threshold,
            },
        )

    def _submit_and_poll(self, image_url: str) -> Dict[str, Any]:
        """Submit HMAE mae task and poll until completion."""
        try:
            task = self.client.infer.submit(
                model_type="hmae",
                inference_config={"task": "mae_reconstruction"},
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
                    return {"error": f"HMAE mae task {task.id} {status.value}"}
            except Exception as exc:
                return {"error": str(exc)}
            time.sleep(self.poll_interval)

        return {"error": f"HMAE mae task timed out after {self.timeout}s"}
