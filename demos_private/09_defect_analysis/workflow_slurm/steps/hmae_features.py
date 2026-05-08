"""
Step 3: HMAE Feature Extraction — compute embeddings for each image patch.

Submits each patch to the BentoML HMAE ``feature_extraction`` task endpoint, collects
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
from typing import Any, Dict, List, Optional

import numpy as np

from umrs.pipeline_sdk import Step, StepContext, StepResult
from umrs.vision_service.bentoml_client import submit_vision_and_wait

logger = logging.getLogger(__name__)


class HMAEFeatureStep(Step):
    """Extract HMAE feature embeddings from image patches."""

    name = "hmae_features"

    def setup(self, config: Dict[str, Any]) -> None:
        self.service_url = config["bentoml_url"].rstrip("/")
        self.timeout = config.get("timeout", 300)
        self.poll_interval = config.get("poll_interval", 2.0)

    def run(self, ctx: StepContext) -> StepResult:
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

            # Submit BentoML HMAE features task and poll
            result = self._submit_and_poll(image_url)

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

    def _submit_and_poll(self, image_url: str) -> Dict[str, Any]:
        """Submit HMAE features task and poll until completion."""
        try:
            return submit_vision_and_wait(
                self.service_url,
                "hmae",
                image_url,
                inference_config={"task": "feature_extraction"},
                poll_interval=self.poll_interval,
                timeout=self.timeout,
            )
        except Exception as exc:
            return {"error": str(exc)}
