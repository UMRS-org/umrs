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

import importlib.util
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from umrs.pipeline_sdk import Step, StepContext, StepResult

logger = logging.getLogger(__name__)

_HELPER_MODULE_NAME = "_umrs_demo10_rivault_vision"
_helper = sys.modules.get(_HELPER_MODULE_NAME)
if _helper is None:
    _helper_path = Path(__file__).with_name("_rivault_vision.py")
    _helper_spec = importlib.util.spec_from_file_location(
        _HELPER_MODULE_NAME,
        _helper_path,
    )
    if _helper_spec is None or _helper_spec.loader is None:
        raise ImportError(f"Cannot load helper module from {_helper_path}")
    _helper = importlib.util.module_from_spec(_helper_spec)
    sys.modules[_HELPER_MODULE_NAME] = _helper
    _helper_spec.loader.exec_module(_helper)

get_rivault_model = _helper.get_rivault_model
split_s3_uri = _helper.split_s3_uri


class FeatureModeStep(Step):
    """Extract HMAE features through the RiVault-backed vision adapter."""

    name = "feature_mode"

    def setup(self, config: Dict[str, Any]) -> None:
        self.model_type = config.get("model_type", "hmae")
        self.feature_layer = config.get("feature_layer", "encoder")
        self._model = None

    def run(self, ctx: StepContext) -> StepResult:
        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []
        embeddings: List[List[float]] = []

        for idx, key in enumerate(ctx.input_keys):
            logger.info("[%d/%d] Feature mode: %s", idx + 1, len(ctx.input_keys), key)

            result, checksum, artifact_uri = self._run_inference(ctx, key)
            embedding, patch_shape, coords_shape, extraction_error = self._extract_embedding(
                ctx,
                artifact_uri,
            )
            if extraction_error and not result.get("error"):
                result["error"] = extraction_error

            base = key.rsplit(".", 1)[0] if "." in key else key
            meta = {
                "source_key": key,
                "mode": "features",
                "model_type": self.model_type,
                "feature_layer": self.feature_layer,
                "embedding_dim": len(embedding),
                "patch_shape": patch_shape,
                "coords_shape": coords_shape,
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
            checksums.append(checksum)
            embeddings.append(embedding)

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
            embeddings=embeddings,
            extra={"mode": "features", "total_images": len(ctx.input_keys)},
        )

    def _get_model(self):
        if self._model is None:
            self._model = get_rivault_model(
                self.model_type,
                task="feature_extraction",
                model_name="HMAE Feature Extraction",
            )
        return self._model

    def _run_inference(
        self,
        ctx: StepContext,
        key: str,
    ) -> tuple[Dict[str, Any], str, str]:
        try:
            results, checksums, _, saved_keys = self._get_model().compute_embeddings(
                ctx.s3,
                [key],
                ctx.input_bucket,
                ctx.output_bucket,
                skip_existing=False,
            )
        except Exception as exc:
            logger.exception("Feature mode failed for %s", key)
            return {"error": str(exc)}, "", ""

        result = results[0] if results else {}
        if not isinstance(result, dict):
            result = {"raw": str(result)}
        checksum = checksums[0] if checksums else ""
        artifact_uri = result.get("output_s3_key") or (saved_keys[0] if saved_keys else "")
        return result, checksum, artifact_uri

    def _extract_embedding(
        self,
        ctx: StepContext,
        artifact_uri: str,
    ) -> tuple[List[float], List[int], List[int], str]:
        if not artifact_uri:
            return [], [], [], "No feature artifact was generated"

        try:
            bucket, key, payload = self._load_feature_payload(ctx, artifact_uri)
            embedding = self._materialize_embedding(payload.get("cls"))
            patch_tensor = payload.get("patches")
            coords = payload.get("coords")
            patch_shape = list(patch_tensor.shape) if hasattr(patch_tensor, "shape") else []
            coords_shape = list(coords.shape) if hasattr(coords, "shape") else []
            self._cleanup_artifact(ctx, bucket, key, artifact_uri)
            return embedding, patch_shape, coords_shape, ""
        except Exception as exc:
            logger.exception("Failed to extract feature embedding from %s", artifact_uri)
            return [], [], [], str(exc)

    def _load_feature_payload(self, ctx: StepContext, artifact_uri: str):
        import torch

        bucket, key = split_s3_uri(artifact_uri)
        payload = torch.load(
            io.BytesIO(ctx.download_bytes(key, bucket=bucket)),
            map_location="cpu",
        )
        return bucket, key, payload

    def _materialize_embedding(self, cls_tensor: Any) -> List[float]:
        if cls_tensor is None:
            return []
        if hasattr(cls_tensor, "detach"):
            cls_tensor = cls_tensor.detach().cpu()
        return cls_tensor.reshape(-1).tolist()

    def _cleanup_artifact(
        self,
        ctx: StepContext,
        bucket: str,
        key: str,
        artifact_uri: str,
    ) -> None:
        if bucket != ctx.output_bucket:
            return
        try:
            ctx.s3.delete_object(Bucket=bucket, Key=key)
        except Exception:
            logger.warning("Unable to remove temporary feature artifact %s", artifact_uri)
