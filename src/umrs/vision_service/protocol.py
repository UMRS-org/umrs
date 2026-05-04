"""Generic Vision Model Serving Protocol — UMRS R4 Platform.

This module defines the standardized protocol for vision model serving in the
UMRS platform.  It provides:

1. ``VisionTask`` — enumeration of supported vision task types
2. ``VisionInput`` / ``VisionOutput`` — typed I/O contracts
3. ``ModelCapabilities`` — declarative model metadata for orchestrator decisions
4. ``VisionModelAdapter`` v2 — enhanced base class with batch support

The protocol is designed to support both standard HuggingFace models (via the
generic ``huggingface`` adapter) and custom research models (HMAE, etc.) through
the same unified interface.

Architecture::

    ┌────────────────────────────────────────────────────────────────────────┐
    │  UMRS Orchestrator                                                     │
    │  • Selects model + adapter based on task/capabilities                  │
    │  • Pre-generates presigned I/O URLs                                    │
    │  • Submits Slurm job with Podman container                             │
    └──────────────────────────────┬─────────────────────────────────────────┘
                                   │
                                   ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  BentoML Vision Service (inside Podman container on Slurm node)        │
    │  ┌──────────────────────────────────────────────────────────┐          │
    │  │  POST /infer/submit                                      │          │
    │  │  {model_type, inference_config, image_url, ...}          │          │
    │  └──────────────────────────────────────────────────────────┘          │
    │                          │                                             │
    │                          ▼                                             │
    │  ┌──────────────────────────────────────────────────────────┐          │
    │  │  ADAPTER_REGISTRY[model_type]                            │          │
    │  │  ┌────────────┐  ┌──────────┐  ┌──────────────────┐     │          │
    │  │  │ SAM3       │  │  HMAE    │  │  HuggingFace     │     │          │
    │  │  │ (custom)   │  │ (custom) │  │  (generic/auto)  │     │          │
    │  │  └────────────┘  └──────────┘  └──────────────────┘     │          │
    │  └──────────────────────────────────────────────────────────┘          │
    └────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VisionTask(str, Enum):
    """Supported vision inference task types.

    These map to HuggingFace pipeline task names where applicable.
    Custom tasks (prefixed with ``custom:``) are handled by dedicated adapters.
    """

    # Standard HuggingFace pipeline tasks
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

    # Custom task types for research models
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
    """Declarative metadata about a vision model's capabilities.

    Used by the orchestrator to:
    - Validate that a model can handle a given request
    - Estimate GPU memory requirements for Slurm resource allocation
    - Determine output artifact types for presigned URL generation
    """

    # Identity
    name: str
    version: str = "1.0.0"
    description: str = ""

    # Tasks this model supports
    supported_tasks: List[VisionTask] = field(default_factory=list)

    # Input capabilities
    supported_inputs: List[InputType] = field(default_factory=list)
    max_input_resolution: Optional[tuple[int, int]] = None
    supported_input_formats: List[str] = field(
        default_factory=lambda: ["image/png", "image/jpeg", "image/tiff"]
    )

    # Output types produced
    output_types: List[OutputType] = field(default_factory=list)

    # Resource requirements (for Slurm allocation)
    gpu_memory_mb: int = 4096
    min_gpu_compute_capability: float = 7.0
    supports_batch: bool = False
    max_batch_size: int = 1
    typical_inference_seconds: float = 5.0

    # Model source
    model_source: str = ""  # HuggingFace repo ID, local path, or custom
    framework: str = "pytorch"  # pytorch, tensorflow, onnx

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict for API responses."""
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
    inference_config: Dict[str, Any] = field(default_factory=dict)
    output_bucket: str = ""
    presigned_input_url: str = ""
    presigned_output_urls: Dict[str, str] = field(default_factory=dict)

    # Optional batch mode
    image_urls: List[str] = field(default_factory=list)
    presigned_input_urls: List[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    """Standardized inference result."""

    status: str = "success"  # success, error, skipped
    source_image: str = ""
    output_s3_key: str = ""
    results_pkl_s3_key: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "status": self.status,
            "source_image": self.source_image,
        }
        if self.output_s3_key:
            d["output_s3_key"] = self.output_s3_key
        if self.results_pkl_s3_key:
            d["results_pkl_s3_key"] = self.results_pkl_s3_key
        if self.error:
            d["error"] = self.error
        if self.metadata:
            d.update(self.metadata)
        return d
