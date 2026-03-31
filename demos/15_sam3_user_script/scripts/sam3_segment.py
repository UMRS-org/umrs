"""
SAM3 segmentation step — user_script version.

Loads the SAM2 model from HuggingFace (facebook/sam2-hiera-small),
generates automatic masks for each input image, and uploads the
segmentation overlay as a PNG to the output bucket.

Requirements: torch, sam2, Pillow, numpy
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict

from umrs.pipeline_sdk import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class SAM3SegmentStep(Step):
    """Run SAM2 (SAM3) automatic mask generation on images."""

    name = "sam3_segment"

    def setup(self, config: dict) -> None:
        self.output_prefix = config.get("output_prefix", "segmented/")
        self.model_id = config.get("model_id", "facebook/sam2-hiera-small")
        self.points_per_side = config.get("points_per_side", 32)
        self._model: Any = None
        self._mask_generator: Any = None

    def _ensure_model(self) -> None:
        """Lazy-load model on first call so setup() stays lightweight."""
        if self._mask_generator is not None:
            return

        import torch
        from sam2.build_sam import build_sam2_hf
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading SAM2 model %s on %s", self.model_id, device)

        self._model = build_sam2_hf(self.model_id, device=device)
        self._mask_generator = SAM2AutomaticMaskGenerator(
            self._model,
            points_per_side=self.points_per_side,
        )
        logger.info("SAM2 model loaded")

    def run(self, ctx: StepContext) -> StepResult:
        import numpy as np
        from PIL import Image

        self._ensure_model()

        output_keys = []
        checksums = []
        metadata_list = []

        for key in ctx.input_keys:
            raw = ctx.download_bytes(key)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img_np = np.array(img)

            # Generate masks
            masks = self._mask_generator.generate(img_np)
            logger.info("  %s: %d masks generated", key, len(masks))

            # Build coloured overlay
            overlay = _create_overlay(img_np, masks)

            # Encode and upload
            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format="PNG")
            out_key = f"{self.output_prefix}{key}"
            ctx.upload_bytes(out_key, buf.getvalue(), content_type="image/png")
            output_keys.append(out_key)

            # Also upload mask metadata as JSON
            meta_key = f"{self.output_prefix}{key}.masks.json"
            mask_meta = [
                {
                    "id": i,
                    "area": int(m["area"]),
                    "bbox": [int(x) for x in m["bbox"]],
                    "predicted_iou": float(m["predicted_iou"]),
                    "stability_score": float(m["stability_score"]),
                }
                for i, m in enumerate(masks)
            ]
            ctx.upload_bytes(
                meta_key,
                json.dumps(mask_meta, indent=2).encode(),
                content_type="application/json",
            )

            head = ctx.head_object(key)
            checksums.append(head["ETag"].strip('"'))
            metadata_list.append(head.get("Metadata", {}))

        return StepResult(
            output_keys=output_keys,
            checksums=checksums,
            metadata=metadata_list,
        )


def _create_overlay(img_np, masks, alpha: float = 0.45):
    """Render coloured mask overlay on the original image."""
    import numpy as np

    overlay = img_np.copy().astype(np.float32)
    rng = np.random.RandomState(42)

    # Sort masks by area (largest first) for nicer layering
    sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    for m in sorted_masks:
        color = rng.randint(0, 256, size=3).astype(np.float32)
        seg = m["segmentation"]
        overlay[seg] = overlay[seg] * (1 - alpha) + color * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)
