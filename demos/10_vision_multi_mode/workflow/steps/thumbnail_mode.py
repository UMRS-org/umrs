"""
Thumbnail Mode — generate resized thumbnails of input images.

A pure-CPU step that creates JPEG thumbnails at multiple sizes.
This runs in parallel with the GPU-bound steps and demonstrates
that not every pipeline branch requires a model.

Config:
  - ``sizes``:   list of max-dimension sizes (default [128, 256])
  - ``format``:  output image format (default "JPEG")
  - ``quality``: JPEG quality 1-100 (default 85)
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, List

from umrs.pipeline_sdk import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class ThumbnailModeStep(Step):
    """Generate thumbnail variants of each input image."""

    name = "thumbnail_mode"

    def setup(self, config: Dict[str, Any]) -> None:
        self.sizes = config.get("sizes", [128, 256])
        self.fmt = config.get("format", "JPEG")
        self.quality = config.get("quality", 85)

    def run(self, ctx: StepContext) -> StepResult:
        from PIL import Image

        output_keys: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        checksums: List[str] = []

        for key in ctx.input_keys:
            img_bytes = ctx.download_bytes(key)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            orig_w, orig_h = image.size
            base = key.rsplit(".", 1)[0] if "." in key else key

            for size in self.sizes:
                thumb = image.copy()
                thumb.thumbnail((size, size), Image.LANCZOS)

                buf = io.BytesIO()
                thumb.save(buf, format=self.fmt, quality=self.quality)
                buf.seek(0)

                ext = "jpg" if self.fmt.upper() == "JPEG" else self.fmt.lower()
                out_key = f"thumbnails/{base}_{size}px.{ext}"
                ctx.upload_bytes(
                    out_key,
                    buf.getvalue(),
                    content_type=f"image/{ext}",
                )
                output_keys.append(out_key)
                metadata_list.append({
                    "source_key": key,
                    "size": size,
                    "original_size": [orig_w, orig_h],
                    "thumb_size": list(thumb.size),
                })
                checksums.append("")

            logger.info(
                "Generated %d thumbnails for %s (%dx%d)",
                len(self.sizes), key, orig_w, orig_h,
            )

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
            extra={"mode": "thumbnails", "total_images": len(ctx.input_keys)},
        )
