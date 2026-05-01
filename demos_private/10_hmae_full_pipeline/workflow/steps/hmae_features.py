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
from typing import Any, Dict, List

import numpy as np

from umrs.pipeline_sdk import Step, StepContext, StepResult
from umrs.vision_service.bentoml_client import submit_vision_and_wait

from _hmae_runtime import download_s3_bytes, resolve_bentoml_url

logger = logging.getLogger(__name__)


class HMAEFeaturesStep(Step):
    """Feature extraction (encoder-only) via BentoML HMAE endpoint."""

    name = "hmae_features"

    def setup(self, config: Dict[str, Any]) -> None:
        self.service_url = resolve_bentoml_url(config)
        self.timeout = config.get("timeout", 300)
        self.poll_interval = config.get("poll_interval", 2.0)

    def run(self, ctx: StepContext) -> StepResult:
        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []
        all_cls: List[np.ndarray] = []
        original_bucket, original_keys = self._resolve_original_inputs(ctx)

        for idx, key in enumerate(original_keys):
            meta_key, meta, cls_emb = self._process_image(
                ctx,
                source_bucket=original_bucket,
                source_key=key,
                index=idx,
                total=len(original_keys),
            )
            output_keys.append(meta_key)
            metadata_list.append(meta)
            checksums.append("")
            all_cls.append(cls_emb)

        embeddings_matrix = self._store_embeddings(ctx, all_cls)
        return self._build_result(
            output_keys=output_keys,
            embeddings_matrix=embeddings_matrix,
            checksums=checksums,
            metadata_list=metadata_list,
            total_images=len(original_keys),
        )

    def _resolve_original_inputs(self, ctx: StepContext) -> tuple[str, List[str]]:
        original_keys = ctx.metadata.get("original_keys")
        original_bucket = ctx.metadata.get("original_bucket", ctx.input_bucket)
        if original_keys:
            return original_bucket, original_keys
        return original_bucket, self._resolve_source_keys(ctx)

    def _store_embeddings(
        self,
        ctx: StepContext,
        all_cls: List[np.ndarray],
    ) -> np.ndarray | None:
        embeddings_matrix = np.stack(all_cls) if all_cls else None
        if embeddings_matrix is not None:
            ctx.upload_numpy("features/all_embeddings.npy", embeddings_matrix)
        return embeddings_matrix

    def _build_result(
        self,
        *,
        output_keys: List[str],
        embeddings_matrix: np.ndarray | None,
        checksums: List[str],
        metadata_list: List[Dict[str, Any]],
        total_images: int,
    ) -> StepResult:
        return StepResult(
            output_keys=output_keys,
            embeddings=embeddings_matrix,
            checksums=checksums,
            metadata=metadata_list,
            extra={
                "total_images": total_images,
                "task": "feature_extraction",
                "embedding_dim": 768,
                "items": metadata_list,
            },
        )

    def _process_image(
        self,
        ctx: StepContext,
        *,
        source_bucket: str,
        source_key: str,
        index: int,
        total: int,
    ) -> tuple[str, Dict[str, Any], np.ndarray]:
        logger.info(
            "[%d/%d] HMAE features: %s",
            index + 1, total, source_key,
        )

        image_url = f"s3://{source_bucket}/{source_key}"
        result = self._submit_and_poll(image_url)
        base = source_key.rsplit(".", 1)[0] if "." in source_key else source_key
        cls_emb = self._resolve_embedding(ctx, result)

        emb_key = f"features/{base}_emb.npy"
        ctx.upload_numpy(emb_key, cls_emb)
        meta = {
            "source_key": source_key,
            "source_bucket": source_bucket,
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
        logger.info("  → embedding shape=%s", cls_emb.shape)
        return meta_key, meta, cls_emb

    def _resolve_embedding(
        self,
        ctx: StepContext,
        result: Dict[str, Any],
    ) -> np.ndarray:
        if "error" in result:
            logger.warning("  features failed: %s", result["error"])
            return np.zeros(768, dtype=np.float32)

        return self._extract_cls(
            ctx, result.get("output_s3_key", ""),
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

        try:
            import torch

            data = download_s3_bytes(feat_s3_url)
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
