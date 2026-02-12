"""
Model Management Package (MM-001 ~ MM-004).

Provides:
- MM-001: Model registry service
- MM-002: Model staging to PVC/NFS
- MM-003: Model version management
- MM-004: Model conversion pipeline (FP16/FP8/GPTQ)

See DEVELOPMENT_OBJECTIVES.md for design details.
"""

from .registry import ModelRegistryService
from .staging import ModelStager, StagingConfig, StagingResult
from .versions import ModelVersionManager, ModelVersion
from .conversion import (
    ModelConverter,
    ConversionConfig,
    ConversionResult,
    QuantizationType,
)

__all__ = [
    # Registry
    "ModelRegistryService",
    # Staging
    "ModelStager",
    "StagingConfig",
    "StagingResult",
    # Versions
    "ModelVersionManager",
    "ModelVersion",
    # Conversion
    "ModelConverter",
    "ConversionConfig",
    "ConversionResult",
    "QuantizationType",
]
