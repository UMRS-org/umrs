"""
Step 2: Region Extraction — crop detected segments into individual patches.

Reads the SAM3 metadata JSON produced by Step 1 (bounding boxes), downloads
the original image from S3, crops each bounding box into a separate patch,
and uploads the patches back to S3.

Input keys:  ``<basename>_sam3_meta.json`` (from Step 1)
Output keys: ``patches/<basename>_patch_<i>.png`` per detected region

Config:
  - ``min_patch_size``:  minimum crop size in pixels (default 32)
  - ``padding``:         extra pixels around each box (default 10)
  - ``max_patches_per_image``: limit patches per image (default 50)
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, List

from umrs.pipeline_sdk import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class RegionExtractionStep(Step):
    """Crop detected SAM3 bounding boxes into individual image patches."""

    name = "region_extract"

    def setup(self, config: Dict[str, Any]) -> None:
        self.min_patch_size = config.get("min_patch_size", 32)
        self.padding = config.get("padding", 10)
        self.max_patches = config.get("max_patches_per_image", 50)

    def run(self, ctx: StepContext) -> StepResult:
        from PIL import Image

        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []

        total_patches = 0

        for meta_key in ctx.input_keys:
            # Read the SAM3 metadata JSON
            meta_bytes = ctx.download_bytes(meta_key)
            meta = json.loads(meta_bytes)

            source_key = meta["source_key"]
            source_bucket = meta.get("source_bucket", ctx.input_bucket)
            boxes = meta.get("boxes", [])
            scores = meta.get("scores", [])

            if not boxes:
                logger.info("No objects in %s — skipping", meta_key)
                continue

            # Download original image
            img_bytes = ctx.download_bytes(source_key, bucket=source_bucket)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = image.size

            base = source_key.rsplit(".", 1)[0] if "." in source_key else source_key

            # Sort by score descending and limit
            scored_boxes = sorted(zip(scores, boxes), key=lambda x: x[0], reverse=True)[
                : self.max_patches
            ]

            for i, (score, box) in enumerate(scored_boxes):
                x1, y1, x2, y2 = [int(v) for v in box]

                # Apply padding
                x1 = max(0, x1 - self.padding)
                y1 = max(0, y1 - self.padding)
                x2 = min(w, x2 + self.padding)
                y2 = min(h, y2 + self.padding)

                # Skip tiny patches
                pw, ph = x2 - x1, y2 - y1
                if pw < self.min_patch_size or ph < self.min_patch_size:
                    continue

                # Crop and upload
                patch = image.crop((x1, y1, x2, y2))
                buf = io.BytesIO()
                patch.save(buf, format="PNG")
                buf.seek(0)

                patch_key = f"patches/{base}_patch_{i:04d}.png"
                ctx.upload_bytes(
                    patch_key,
                    buf.getvalue(),
                    content_type="image/png",
                    metadata={
                        "source_image": source_key,
                        "box": json.dumps([x1, y1, x2, y2]),
                        "score": str(score),
                        "patch_index": str(i),
                    },
                )
                output_keys.append(patch_key)
                metadata_list.append(
                    {
                        "source_image": source_key,
                        "box": [x1, y1, x2, y2],
                        "score": score,
                        "patch_index": i,
                        "patch_size": [pw, ph],
                    }
                )
                checksums.append("")
                total_patches += 1

            logger.info(
                "Extracted %d patches from %s (%d boxes)",
                len(scored_boxes),
                source_key,
                len(boxes),
            )

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
            extra={
                "total_patches": total_patches,
                "images_processed": len(ctx.input_keys),
            },
        )
