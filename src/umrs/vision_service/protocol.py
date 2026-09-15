"""Generic Vision Model Serving Protocol for UMRS.

This module defines the standardized protocol for vision model serving in the
UMRS platform. It provides:

1. ``VisionTask`` - enumeration of supported vision task types
2. ``InputType`` / ``OutputType`` - typed I/O contracts
3. ``ModelCapabilities`` - declarative model metadata for orchestrator decisions
4. ``InferenceRequest`` / ``InferenceResult`` - service payload contracts

The protocol supports standard HuggingFace models through the generic
``huggingface`` adapter and custom research models such as HMAE through
dedicated adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VisionTask(str, Enum):
    """Supported vision inference task types."""

    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    IMAGE_SEGMENTATION = "image-segmentation"
    SEMANTIC_SEGMENTATION = "semantic-segmentation"
    INSTANCE_SEGMENTATION = "instance-segmentation"
    PANOPTIC_SEGMENTATION = "panoptic-segmentation"
    IMAGE_FEATURE_EXTRACTION = "image-feature-extraction"
    ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"
    ZERO_SHOT_OBJECT_DETECTION = "zero-shot-object-detection"
    DEPTH_ESTIMATION = "depth-estimation"
    IMAGE_TO_IMAGE = "image-to-image"

    CUSTOM_RECONSTRUCTION = "custom:reconstruction"
    CUSTOM_MAE_RECONSTRUCTION = "custom:mae-reconstruction"
    CUSTOM_FEATURE_EXTRACTION = "custom:feature-extraction"
    CUSTOM_SEGMENTATION = "custom:segmentation"


class InputType(str, Enum):
    """Types of input data the vision service accepts."""

    SINGLE_IMAGE = "single_image"
    IMAGE_BATCH = "image_batch"
    VOLUME_SLICES = "volume_slices"
    IMAGE_WITH_POINTS = "image_with_points"
    IMAGE_WITH_BOXES = "image_with_boxes"
    IMAGE_WITH_TEXT = "image_with_text"


class OutputType(str, Enum):
    """Types of output data the vision service produces."""

    MASKS = "masks"
    BOUNDING_BOXES = "bounding_boxes"
    CLASSIFICATIONS = "classifications"
    EMBEDDINGS = "embeddings"
    FEATURE_TENSORS = "feature_tensors"
    RECONSTRUCTED_IMAGE = "reconstructed_image"
    DEPTH_MAP = "depth_map"
    OVERLAY_IMAGE = "overlay_image"
    SCORES = "scores"


@dataclass
class ModelCapabilities:
    """Declarative metadata about a vision model's capabilities."""

    name: str
    version: str = "1.0.0"
    description: str = ""

    supported_tasks: list[VisionTask] = field(default_factory=list)
    supported_inputs: list[InputType] = field(default_factory=list)
    max_input_resolution: tuple[int, int] | None = None
    supported_input_formats: list[str] = field(
        default_factory=lambda: ["image/png", "image/jpeg", "image/tiff"]
    )
    output_types: list[OutputType] = field(default_factory=list)

    gpu_memory_mb: int = 4096
    min_gpu_compute_capability: float = 7.0
    supports_batch: bool = False
    max_batch_size: int = 1
    typical_inference_seconds: float = 5.0

    model_source: str = ""
    framework: str = "pytorch"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary for API responses."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "supported_tasks": [t.value for t in self.supported_tasks],
            "supported_inputs": [i.value for i in self.supported_inputs],
            "max_input_resolution": self.max_input_resolution,
            "supported_input_formats": self.supported_input_formats,
            "output_types": [o.value for o in self.output_types],
            "gpu_memory_mb": self.gpu_memory_mb,
            "supports_batch": self.supports_batch,
            "max_batch_size": self.max_batch_size,
            "typical_inference_seconds": self.typical_inference_seconds,
            "model_source": self.model_source,
            "framework": self.framework,
        }


@dataclass
class InferenceRequest:
    """Standardized inference request payload."""

    model_type: str
    image_url: str
    inference_config: dict[str, Any] = field(default_factory=dict)
    output_bucket: str = ""
    presigned_input_url: str = ""
    presigned_output_urls: dict[str, str] = field(default_factory=dict)

    image_urls: list[str] = field(default_factory=list)
    presigned_input_urls: list[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    """Standardized inference result."""

    status: str = "success"
    source_image: str = ""
    output_s3_key: str = ""
    results_pkl_s3_key: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary for API responses."""
        result: dict[str, Any] = {
            "status": self.status,
            "source_image": self.source_image,
        }
        if self.output_s3_key:
            result["output_s3_key"] = self.output_s3_key
        if self.results_pkl_s3_key:
            result["results_pkl_s3_key"] = self.results_pkl_s3_key
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result.update(self.metadata)
        return result
