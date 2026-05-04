"""``OCRStep`` — convenience wrapper over :class:`VisionStep` in OCR mode.

Provided as a separate entrypoint for ergonomics in ``workflow.json``::

    {
      "name": "ocr",
      "entrypoint": "umrs.pipeline_sdk.templates.ocr_step:OCRStep",
      "config": {"image_param": "image"}
    }
"""

from __future__ import annotations

from typing import Any

from .vision_step import VisionStep


class OCRStep(VisionStep):
    """A :class:`VisionStep` pre-configured to ``mode=ocr``."""

    name = "ocr"

    def setup(self, config: dict[str, Any]) -> None:
        config = dict(config)
        config["mode"] = "ocr"
        super().setup(config)
