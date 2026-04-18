"""
Step 3: Image Statistics — compute per-patch statistics and aggregate report.

Reads cropped image patches from Step 2, computes basic image statistics
(mean, std, histogram, aspect ratio, brightness), and writes per-patch
JSON metadata and an aggregate report.

This step demonstrates a pure-Python analysis step that requires no GPU
and no external model — useful as a template for custom analysis.

Input keys:  ``patches/<basename>_patch_<i>.png`` (from Step 2)
Output keys: ``analysis/<basename>_patch_<i>_stats.json`` per patch
             ``analysis/report.json`` (aggregate summary)

Config:
  - ``output_prefix``: prefix for output keys (default "analysis/")
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, List

import numpy as np

from umrs.pipeline_sdk import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class ImageStatsStep(Step):
    """Compute image statistics for each cropped patch."""

    name = "image_stats"

    def setup(self, config: Dict[str, Any]) -> None:
        self.output_prefix = config.get("output_prefix", "analysis/")

    def run(self, ctx: StepContext) -> StepResult:
        from PIL import Image

        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []
        all_stats: List[Dict[str, Any]] = []

        for idx, key in enumerate(ctx.input_keys):
            logger.info("[%d/%d] Analyzing: %s", idx + 1, len(ctx.input_keys), key)

            img_bytes = ctx.download_bytes(key)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            arr = np.array(image, dtype=np.float32) / 255.0

            # Per-channel statistics
            stats = {
                "source_patch": key,
                "width": image.width,
                "height": image.height,
                "aspect_ratio": round(image.width / max(image.height, 1), 3),
                "mean_r": round(float(arr[:, :, 0].mean()), 4),
                "mean_g": round(float(arr[:, :, 1].mean()), 4),
                "mean_b": round(float(arr[:, :, 2].mean()), 4),
                "std_r": round(float(arr[:, :, 0].std()), 4),
                "std_g": round(float(arr[:, :, 1].std()), 4),
                "std_b": round(float(arr[:, :, 2].std()), 4),
                "brightness": round(float(arr.mean()), 4),
                "contrast": round(float(arr.std()), 4),
            }

            # Histogram (8 bins per channel)
            for ch, name in enumerate(["r", "g", "b"]):
                hist, _ = np.histogram(arr[:, :, ch], bins=8, range=(0, 1))
                stats[f"hist_{name}"] = hist.tolist()

            all_stats.append(stats)

            # Write per-patch JSON
            patch_base = key.replace("patches/", "").rsplit(".", 1)[0]
            out_key = f"{self.output_prefix}{patch_base}_stats.json"
            ctx.upload_bytes(
                out_key,
                json.dumps(stats, indent=2).encode(),
                content_type="application/json",
            )
            output_keys.append(out_key)
            metadata_list.append(stats)
            checksums.append("")

        # Aggregate report
        if all_stats:
            report = {
                "total_patches": len(all_stats),
                "avg_brightness": round(
                    sum(s["brightness"] for s in all_stats) / len(all_stats), 4
                ),
                "avg_contrast": round(
                    sum(s["contrast"] for s in all_stats) / len(all_stats), 4
                ),
                "size_range": {
                    "min_width": min(s["width"] for s in all_stats),
                    "max_width": max(s["width"] for s in all_stats),
                    "min_height": min(s["height"] for s in all_stats),
                    "max_height": max(s["height"] for s in all_stats),
                },
                "patches": all_stats,
            }
        else:
            report = {"total_patches": 0, "error": "no patches to analyze"}

        report_key = f"{self.output_prefix}report.json"
        ctx.upload_bytes(
            report_key,
            json.dumps(report, indent=2).encode(),
            content_type="application/json",
        )
        output_keys.append(report_key)

        logger.info("Report: %d patches analyzed", len(all_stats))

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
            extra={"total_patches": len(all_stats)},
        )
