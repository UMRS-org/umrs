"""
Feature Mode — extract feature embeddings from images via BentoML.

Submits each image to the vision model with ``inference_mode="features"``
and writes the returned embedding vectors as JSON.  This demonstrates how
different model modes can run as independent, parallelisable pipeline
branches.

Config:
  - ``bentoml_url``:    BentoML vision service base URL
  - ``model_type``:     vision model to invoke (default "sam3")
  - ``feature_layer``:  which feature layer to extract (default "encoder")
  - ``timeout``:        per-task timeout (default 300)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

import bentoml

from umrs.pipeline_sdk import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class FeatureModeStep(Step):
    """Extract feature embeddings via BentoML vision model."""

    name = "feature_mode"

    def setup(self, config: Dict[str, Any]) -> None:
        self.client = bentoml.SyncHTTPClient(
            config["bentoml_url"].rstrip("/"),
            server_ready_timeout=60,
        )
        self.model_type = config.get("model_type", "sam3")
        self.feature_layer = config.get("feature_layer", "encoder")
        self.poll_interval = config.get("poll_interval", 2.0)
        self.timeout = config.get("timeout", 300)

    def run(self, ctx: StepContext) -> StepResult:
        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []
        embeddings: List[List[float]] = []

        for idx, key in enumerate(ctx.input_keys):
            logger.info("[%d/%d] Feature mode: %s", idx + 1, len(ctx.input_keys), key)

            image_url = f"s3://{ctx.input_bucket}/{key}"
            task = self.client.infer.submit(
                model_type=self.model_type,
                inference_config={
                    "mode": "features",
                    "feature_layer": self.feature_layer,
                },
                image_url=image_url,
            )

            deadline = time.time() + self.timeout
            while time.time() < deadline:
                status = task.get_status()
                if status.state in ("SUCCESS", "FAILED"):
                    break
                time.sleep(self.poll_interval)

            result = {}
            embedding = []
            if status.state == "SUCCESS":
                raw = task.get()
                result = raw if isinstance(raw, dict) else {"raw": str(raw)}
                embedding = result.get("embedding", [])
            else:
                result = {"error": f"Task {status.state} after {self.timeout}s"}

            base = key.rsplit(".", 1)[0] if "." in key else key
            meta = {
                "source_key": key,
                "mode": "features",
                "model_type": self.model_type,
                "feature_layer": self.feature_layer,
                "embedding_dim": len(embedding),
                "error": result.get("error", ""),
            }

            out_key = f"features/{base}_features.json"
            ctx.upload_bytes(
                out_key,
                json.dumps({**meta, "embedding": embedding}, indent=2).encode(),
                content_type="application/json",
            )
            output_keys.append(out_key)
            metadata_list.append(meta)
            checksums.append("")
            embeddings.append(embedding)

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
            embeddings=embeddings,
            extra={"mode": "features", "total_images": len(ctx.input_keys)},
        )
