"""Shared schemas exchanged between UMRS subsystems and pipeline steps.

The schemas in this package form the *contracts* used by built-in
pipeline step templates (``umrs.pipeline_sdk.templates``) so that
vision steps and LLM steps can be composed without each step needing
to know the other's internal representation.
"""

from .vision import (
    BoundingBox,
    DetectedObject,
    ImageInfo,
    ModelInfo,
    OCRItem,
    VisionPerception,
)

__all__ = [
    "BoundingBox",
    "DetectedObject",
    "ImageInfo",
    "ModelInfo",
    "OCRItem",
    "VisionPerception",
]
