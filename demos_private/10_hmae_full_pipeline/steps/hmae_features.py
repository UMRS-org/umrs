"""
Step 3: HMAE Feature Extraction — encoder-only, produces CLS + patch embeddings.

Reads per-image metadata JSONs from Step 2 (or ``ctx.metadata["original_keys"]``)
to locate the original images, then submits each to BentoML with
``mode=features``.

Input keys:  ``mae/<basename>_mae.json`` (from Step 2)
Output keys: ``features/<basename>_feat.json`` per image

Each result JSON contains:
  - ``source_key``: original image S3 key
  - ``features_s3_key``: S3 key of the ``.pt`` file (CLS + patch tokens)
  - ``embedding_key``: S3 key of the ``.npy`` CLS embedding
  - ``embedding_dim``: dimension of the CLS embedding (768)

The step also returns ``StepResult.embeddings`` (N×768) for pgvector storage.

Config:
  - ``bentoml_url``: BentoML vision service URL
  - ``timeout``: per-task timeout in seconds (default 300)
  - ``poll_interval``: polling interval in seconds (default 2.0)
"""

from __future__ import annotations

import io
import json
import logging
import time
from typing import Any, Dict, List

import bentoml
import numpy as np

from umrs.pipeline_sdk import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class HMAEFeaturesStep(Step):
    """Feature extraction (encoder-only) via BentoML HMAE endpoint."""

    name = "hmae_features"

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
        all_cls: List[np.ndarray] = []

        # Resolve original image keys
        original_keys = ctx.metadata.get("original_keys")
        original_bucket = ctx.metadata.get("original_bucket", ctx.input_bucket)

        if not original_keys:
            original_keys = self._resolve_source_keys(ctx)

        for idx, key in enumerate(original_keys):
            logger.info(
                "[%d/%d] HMAE features: %s",
                idx + 1, len(original_keys), key,
            )

            image_url = f"s3://{original_bucket}/{key}"
            result = self._submit_and_poll(image_url)

            base = key.rsplit(".", 1)[0] if "." in key else key

            if "error" in result:
                logger.warning("  features failed: %s", result["error"])
                cls_emb = np.zeros(768, dtype=np.float32)
            else:
                cls_emb = self._extract_cls(
                    ctx, result.get("output_s3_key", ""),
                )

            all_cls.append(cls_emb)

            # Save CLS embedding as .npy
            emb_key = f"features/{base}_emb.npy"
            ctx.upload_numpy(emb_key, cls_emb)

            meta = {
                "source_key": key,
                "source_bucket": original_bucket,
                "task": "feature_extraction",
                "features_s3_key": result.get("output_s3_key", ""),
                "embedding_key": emb_key,
                "embedding_dim": int(cls_emb.shape[0]),
                "error": result.get("error"),
            }

            meta_key = f"features/{base}_feat.json"
            ctx.upload_bytes(
                meta_key,
                json.dumps(meta, indent=2).encode(),
                content_type="application/json",
            )
            output_keys.append(meta_key)
            metadata_list.append(meta)
            checksums.append("")

            logger.info("  → embedding shape=%s", cls_emb.shape)

        # Stack all CLS embeddings
        embeddings_matrix = np.stack(all_cls) if all_cls else None

        if embeddings_matrix is not None:
            ctx.upload_numpy("features/all_embeddings.npy", embeddings_matrix)

        return StepResult(
            output_keys=output_keys,
            embeddings=embeddings_matrix,
            checksums=checksums,
            metadata=metadata_list,
            extra={
                "total_images": len(original_keys),
                "task": "feature_extraction",
                "embedding_dim": 768,
            },
        )

    # ------------------------------------------------------------------

    def _resolve_source_keys(self, ctx: StepContext) -> List[str]:
        """Parse previous step's output JSONs to find original image keys."""
        keys = []
        for meta_key in ctx.input_keys:
            try:
                raw = ctx.download_bytes(meta_key)
                meta = json.loads(raw)
                keys.append(meta["source_key"])
            except Exception as exc:
                logger.warning("Cannot parse %s: %s", meta_key, exc)
        return keys

    def _extract_cls(self, ctx: StepContext, feat_s3_url: str) -> np.ndarray:
        """Download features.pt from S3 and return the CLS vector."""
        if not feat_s3_url:
            return np.zeros(768, dtype=np.float32)

        path = feat_s3_url[5:] if feat_s3_url.startswith("s3://") else feat_s3_url
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        try:
            import torch

            data = ctx.download_bytes(key, bucket=bucket)
            obj = torch.load(
                io.BytesIO(data), map_location="cpu", weights_only=False,
            )
            cls_token = obj["cls"]  # shape [1, 768]
            return cls_token.squeeze(0).numpy().astype(np.float32)
        except Exception as exc:
            logger.warning("CLS extraction failed for %s: %s", feat_s3_url, exc)
            return np.zeros(768, dtype=np.float32)

    def _submit_and_poll(self, image_url: str) -> Dict[str, Any]:
        """Submit HMAE features task and poll until done."""
        try:
            task = self.client.infer.submit(
                model_type="hmae",
                inference_config={"task": "feature_extraction"},
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
                    return {"error": f"Features task {task.id} {status.value}"}
            except Exception as exc:
                return {"error": str(exc)}
            time.sleep(self.poll_interval)

        return {"error": f"Features task timed out after {self.timeout}s"}
