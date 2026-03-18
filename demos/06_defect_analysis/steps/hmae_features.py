"""
Step 3: HMAE Feature Extraction — compute embeddings for each image patch.

Submits each patch to the BentoML HMAE ``features`` mode endpoint, collects
the CLS token (768-d) and patch tokens, and stores embeddings as ``.npy``.

Input keys:  ``patches/<basename>_patch_<i>.png`` (from Step 2)
Output keys: ``embeddings/<basename>_patch_<i>_emb.npy``

The step also accumulates all CLS embeddings into
``StepResult.embeddings`` (N×768 ndarray) for downstream clustering.

Config:
  - ``bentoml_url``: BentoML vision service URL
  - ``timeout``: per-task timeout in seconds (default 300)
  - ``poll_interval``: polling interval (default 2.0)
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from umrs.steps import UMRSStep, StepContext, StepResult

logger = logging.getLogger(__name__)


class HMAEFeatureStep(UMRSStep):
    """Extract HMAE feature embeddings from image patches."""

    name = "hmae_features"

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
        all_cls_embeddings: List[np.ndarray] = []

        for idx, key in enumerate(ctx.input_keys):
            logger.info(
                "[%d/%d] HMAE features: %s",
                idx + 1, len(ctx.input_keys), key,
            )

            image_url = f"s3://{ctx.input_bucket}/{key}"

            # Submit BentoML HMAE features task
            resp = session.post(
                f"{self.bentoml_url}/infer/submit",
                json={
                    "model_type": "hmae",
                    "model_config": {"mode": "features"},
                    "image_url": image_url,
                },
                timeout=30,
            )
            resp.raise_for_status()
            task_id = resp.json().get("task_id") or resp.json().get("id")

            # Poll
            result = self._poll_result(session, task_id)

            if "error" in result:
                logger.warning("HMAE features failed for %s: %s", key, result["error"])
                # Still produce an output key with a zero embedding
                emb = np.zeros(768, dtype=np.float32)
                all_cls_embeddings.append(emb)
            else:
                # Download the features .pt file and extract CLS token
                feat_s3_key = result.get("output_s3_key", "")
                cls_shape = result.get("cls_shape", [1, 768])
                emb = self._extract_cls_from_pt(ctx, feat_s3_key)
                all_cls_embeddings.append(emb)

            # Save embedding as .npy
            base = key.rsplit(".", 1)[0] if "." in key else key
            # patches/img_patch_0001 → embeddings/img_patch_0001_emb.npy
            emb_key = base.replace("patches/", "embeddings/") + "_emb.npy"
            ctx.upload_numpy(emb_key, emb)
            output_keys.append(emb_key)
            checksums.append("")
            metadata_list.append({
                "source_patch": key,
                "embedding_dim": int(emb.shape[0]) if emb.ndim == 1 else list(emb.shape),
            })

            logger.info("  → embedding shape=%s", emb.shape)

        # Stack all CLS embeddings into a single matrix
        embeddings_matrix = (
            np.stack(all_cls_embeddings) if all_cls_embeddings else None
        )

        return StepResult(
            output_keys=output_keys,
            embeddings=embeddings_matrix,
            checksums=checksums,
            metadata=metadata_list,
            extra={
                "total_embeddings": len(all_cls_embeddings),
                "embedding_dim": 768,
            },
        )

    def _extract_cls_from_pt(
        self, ctx: StepContext, feat_s3_url: str,
    ) -> np.ndarray:
        """Download a features.pt file from S3 and return the CLS vector."""
        # feat_s3_url is like "s3://bucket/key_features.pt"
        if feat_s3_url.startswith("s3://"):
            path = feat_s3_url[5:]
        else:
            path = feat_s3_url
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        try:
            import torch
            data = ctx.download_bytes(key, bucket=bucket)
            obj = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
            cls_token = obj["cls"]  # shape [1, 768]
            return cls_token.squeeze(0).numpy().astype(np.float32)
        except Exception as exc:
            logger.warning("Failed to extract CLS from %s: %s", feat_s3_url, exc)
            return np.zeros(768, dtype=np.float32)

    def _poll_result(self, session, task_id: str) -> Dict[str, Any]:
        """Poll BentoML HMAE task queue until completion."""
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
                return {"error": f"HMAE task {task_id} failed"}

            time.sleep(self.poll_interval)

        return {"error": f"HMAE task {task_id} timed out after {self.timeout}s"}
