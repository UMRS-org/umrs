"""
Step 5: Structure Discovery — cluster embeddings and produce analysis report.

Reads the HMAE embeddings (.npy) produced by Step 3 and the anomaly JSON
produced by Step 4, runs dimensionality reduction + clustering, and
writes a comprehensive analysis report to S3.

This step is intentionally self-contained: it uses scikit-learn's
KMeans and UMAP (if available) or PCA as a fallback for 2-D projection.

Input keys:  ``embeddings/<basename>_patch_<i>_emb.npy`` (from Step 3)
Output keys: ``analysis/report.json``

Config:
  - ``n_clusters``: number of KMeans clusters (default 4)
  - ``anomaly_prefix``: S3 prefix where anomaly JSONs live (default "anomaly/")
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional

import numpy as np

from umrs.steps import UMRSStep, StepContext, StepResult

logger = logging.getLogger(__name__)


class StructureDiscoveryStep(UMRSStep):
    """Cluster HMAE embeddings and produce a dataset-level analysis report."""

    name = "structure_discovery"

    def setup(self, config: Dict[str, Any]) -> None:
        self.n_clusters = config.get("n_clusters", 4)
        self.anomaly_prefix = config.get("anomaly_prefix", "anomaly/")

    def run(self, ctx: StepContext) -> StepResult:
        # -- 1. Load all embeddings ----------------------------------------
        emb_keys = sorted(ctx.input_keys)
        vectors: List[np.ndarray] = []
        patch_names: List[str] = []

        for key in emb_keys:
            arr = ctx.download_numpy(key)
            if arr.ndim == 0:
                continue
            if arr.ndim > 1:
                arr = arr.flatten()[:768]
            vectors.append(arr)
            patch_names.append(key)

        if not vectors:
            logger.warning("No embeddings found — empty report")
            report_key = "analysis/report.json"
            ctx.upload_bytes(
                report_key,
                json.dumps({"error": "no embeddings"}).encode(),
                content_type="application/json",
            )
            return StepResult(output_keys=[report_key])

        X = np.stack(vectors)  # (N, 768)
        logger.info("Loaded %d embeddings, shape=%s", X.shape[0], X.shape)

        # -- 2. Dimensionality reduction (UMAP or PCA fallback) ------------
        coords_2d = self._reduce_2d(X)

        # -- 3. Clustering -------------------------------------------------
        labels, centroids = self._cluster(X)

        # -- 4. Load anomaly scores (if available) -------------------------
        anomaly_scores = self._load_anomaly_scores(ctx, patch_names)

        # -- 5. Build report -----------------------------------------------
        per_patch = []
        for i, name in enumerate(patch_names):
            entry: Dict[str, Any] = {
                "patch_key": name,
                "cluster": int(labels[i]) if labels is not None else -1,
                "umap_x": float(coords_2d[i, 0]) if coords_2d is not None else None,
                "umap_y": float(coords_2d[i, 1]) if coords_2d is not None else None,
            }
            if anomaly_scores and i < len(anomaly_scores):
                entry["mae_loss"] = anomaly_scores[i].get("mae_loss", None)
                entry["is_anomaly"] = anomaly_scores[i].get("is_anomaly", False)
            per_patch.append(entry)

        # Cluster summary
        cluster_summary = {}
        if labels is not None:
            for c in range(self.n_clusters):
                members = [p for p in per_patch if p["cluster"] == c]
                anomalies_in_c = [
                    p for p in members if p.get("is_anomaly", False)
                ]
                cluster_summary[f"cluster_{c}"] = {
                    "count": len(members),
                    "anomaly_count": len(anomalies_in_c),
                    "anomaly_rate": (
                        len(anomalies_in_c) / len(members)
                        if members else 0.0
                    ),
                }

        report = {
            "total_patches": len(patch_names),
            "n_clusters": self.n_clusters,
            "cluster_summary": cluster_summary,
            "patches": per_patch,
        }

        report_key = "analysis/report.json"
        ctx.upload_bytes(
            report_key,
            json.dumps(report, indent=2).encode(),
            content_type="application/json",
        )

        # Also save 2-D projection as .npy for visualization scripts
        if coords_2d is not None:
            ctx.upload_numpy("analysis/projection_2d.npy", coords_2d)

        logger.info(
            "Report: %d patches, %d clusters, %s",
            len(patch_names), self.n_clusters,
            {k: v["count"] for k, v in cluster_summary.items()},
        )

        return StepResult(
            output_keys=[report_key],
            metadata=[report],
            extra={"cluster_summary": cluster_summary},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reduce_2d(X: np.ndarray) -> Optional[np.ndarray]:
        """Project N×D embeddings to N×2 via UMAP (preferred) or PCA."""
        if X.shape[0] < 2:
            return None
        try:
            from umap import UMAP  # type: ignore[import-untyped]
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, X.shape[0]))
            return reducer.fit_transform(X).astype(np.float32)
        except ImportError:
            pass

        try:
            from sklearn.decomposition import PCA  # type: ignore[import-untyped]
            pca = PCA(n_components=2)
            return pca.fit_transform(X).astype(np.float32)
        except ImportError:
            pass

        logger.warning("Neither umap-learn nor scikit-learn available — no 2-D projection")
        return None

    def _cluster(self, X: np.ndarray):
        """KMeans clustering. Falls back to no clustering if sklearn missing."""
        if X.shape[0] < self.n_clusters:
            logger.warning(
                "Fewer samples (%d) than clusters (%d) — skipping clustering",
                X.shape[0], self.n_clusters,
            )
            return None, None
        try:
            from sklearn.cluster import KMeans  # type: ignore[import-untyped]
            km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            return labels, km.cluster_centers_
        except ImportError:
            logger.warning("scikit-learn not available — no clustering")
            return None, None

    def _load_anomaly_scores(
        self, ctx: StepContext, patch_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Try to load anomaly JSON files that correspond to the patches."""
        scores: List[Dict[str, Any]] = []
        for name in patch_names:
            # embeddings/img_patch_0001_emb.npy → anomaly/img_patch_0001_anomaly.json
            anomaly_key = (
                name
                .replace("embeddings/", "anomaly/")
                .replace("_emb.npy", "_anomaly.json")
            )
            try:
                data = ctx.download_bytes(anomaly_key)
                scores.append(json.loads(data))
            except Exception:
                scores.append({})
        return scores
