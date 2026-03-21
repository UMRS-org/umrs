"""
Step 3: HMAE Analysis — combined feature extraction + anomaly detection.

For each image patch, this step submits **two** BentoML HMAE tasks:
  1. ``mode=features`` → CLS embedding (768-d vector)
  2. ``mode=mae``      → masked reconstruction error (anomaly score)

This eliminates the need for separate feature and anomaly steps in the
linear workflow chain.

Input keys:  ``patches/<basename>_patch_<i>.png`` (from Step 2)
Output keys: ``analysis/<basename>_patch_<i>_result.json`` per patch

Each result JSON contains:
  - ``source_patch``: patch S3 key
  - ``embedding_key``: S3 key for the .npy embedding
  - ``mae_loss``: reconstruction error
  - ``is_anomaly``: whether error exceeds threshold
  - ``mae_output_key``: visualization PNG

The step also writes a combined ``analysis/all_embeddings.npy`` (N×768)
and returns it in ``StepResult.embeddings``.

Config:
  - ``bentoml_url``: BentoML vision service URL
  - ``anomaly_threshold``: mae_loss above this → anomaly (default 0.3)
  - ``timeout``: per-task timeout (default 300)
  - ``poll_interval``: (default 2.0)
"""

from __future__ import annotations

import io
import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from umrs.steps import UMRSStep, StepContext, StepResult

logger = logging.getLogger(__name__)


class HMAEAnalysisStep(UMRSStep):
    """Combined HMAE feature extraction + anomaly scoring for each patch."""

    name = "hmae_analysis"

    def setup(self, config: Dict[str, Any]) -> None:
        self.bentoml_url = config["bentoml_url"].rstrip("/")
        self.anomaly_threshold = config.get("anomaly_threshold", 0.3)
        self.timeout = config.get("timeout", 300)
        self.poll_interval = config.get("poll_interval", 2.0)

    def run(self, ctx: StepContext) -> StepResult:
        import requests

        session = requests.Session()
        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []
        all_cls: List[np.ndarray] = []
        anomaly_count = 0

        for idx, key in enumerate(ctx.input_keys):
            logger.info(
                "[%d/%d] HMAE analysis: %s",
                idx + 1, len(ctx.input_keys), key,
            )

            image_url = f"s3://{ctx.input_bucket}/{key}"
            base = key.rsplit(".", 1)[0] if "." in key else key
            patch_base = base.replace("patches/", "")

            # ── 1. Feature extraction ────────────────────────────────────
            feat_result = self._submit_and_poll(
                session, image_url, mode="features",
            )
            if "error" in feat_result:
                logger.warning("  features failed: %s", feat_result["error"])
                cls_emb = np.zeros(768, dtype=np.float32)
            else:
                cls_emb = self._extract_cls(ctx, feat_result.get("output_s3_key", ""))

            all_cls.append(cls_emb)

            # Save embedding
            emb_key = f"embeddings/{patch_base}_emb.npy"
            ctx.upload_numpy(emb_key, cls_emb)

            # ── 2. MAE anomaly scoring ───────────────────────────────────
            mae_result = self._submit_and_poll(
                session, image_url, mode="mae",
            )
            mae_loss = float(mae_result.get("mae_loss", -1.0))
            is_anomaly = (
                mae_loss > self.anomaly_threshold if mae_loss >= 0 else False
            )
            if is_anomaly:
                anomaly_count += 1

            # ── 3. Build per-patch result JSON ───────────────────────────
            patch_result = {
                "source_patch": key,
                "embedding_key": emb_key,
                "embedding_dim": int(cls_emb.shape[0]),
                "mae_loss": mae_loss,
                "anomaly_threshold": self.anomaly_threshold,
                "is_anomaly": is_anomaly,
                "mae_output_key": mae_result.get("output_s3_key", ""),
                "features_output_key": feat_result.get("output_s3_key", ""),
            }

            out_key = f"analysis/{patch_base}_result.json"
            ctx.upload_bytes(
                out_key,
                json.dumps(patch_result, indent=2).encode(),
                content_type="application/json",
            )
            output_keys.append(out_key)
            checksums.append("")
            metadata_list.append(patch_result)

            status_str = "ANOMALY" if is_anomaly else "normal"
            logger.info(
                "  → emb=%s  mae_loss=%.4f [%s]",
                cls_emb.shape, mae_loss, status_str,
            )

        # ── Stack all embeddings ─────────────────────────────────────────
        embeddings_matrix = np.stack(all_cls) if all_cls else None

        # Save combined embeddings file
        if embeddings_matrix is not None:
            ctx.upload_numpy("analysis/all_embeddings.npy", embeddings_matrix)

        return StepResult(
            output_keys=output_keys,
            embeddings=embeddings_matrix,
            checksums=checksums,
            metadata=metadata_list,
            extra={
                "total_patches": len(ctx.input_keys),
                "anomaly_count": anomaly_count,
                "anomaly_rate": (
                    anomaly_count / len(ctx.input_keys)
                    if ctx.input_keys else 0.0
                ),
                "embedding_dim": 768,
            },
        )

    # ------------------------------------------------------------------

    def _submit_and_poll(
        self, session, image_url: str, mode: str,
    ) -> Dict[str, Any]:
        """Submit HMAE task and poll until done."""
        try:
            resp = session.post(
                f"{self.bentoml_url}/infer/submit",
                json={
                    "model_type": "hmae",
                    "model_config": {"mode": mode},
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
                    return {"error": f"HMAE {mode} task {task_id} failed"}
            except Exception as exc:
                return {"error": str(exc)}

            time.sleep(self.poll_interval)

        return {"error": f"HMAE {mode} task timed out after {self.timeout}s"}

    def _extract_cls(self, ctx: StepContext, feat_s3_url: str) -> np.ndarray:
        """Download features.pt and return CLS vector."""
        if not feat_s3_url:
            return np.zeros(768, dtype=np.float32)

        path = feat_s3_url[5:] if feat_s3_url.startswith("s3://") else feat_s3_url
        parts = path.split("/", 1)
        bucket, key = parts[0], (parts[1] if len(parts) > 1 else "")

        try:
            import torch
            data = ctx.download_bytes(key, bucket=bucket)
            obj = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
            cls_token = obj["cls"]  # [1, 768]
            return cls_token.squeeze(0).numpy().astype(np.float32)
        except Exception as exc:
            logger.warning("CLS extraction failed for %s: %s", feat_s3_url, exc)
            return np.zeros(768, dtype=np.float32)
