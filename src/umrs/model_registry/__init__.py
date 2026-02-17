"""
Model Registry module for UMRS.

Provides model catalog management with:
- Scoped model entries (system, project, user)
- Access control (ACL) enforcement
- Path validation for secure serving
- Model metadata and serving profiles

See DEVELOPMENT_OBJECTIVES.md Section 1.9 for design details.
"""

from .models import (
    ModelEntry,
    ModelScope,
    ModelStatus,
    ModelType,
    generate_model_id,
)
from .schema import ensure_schema
from .store import (
    DuplicateModelError,
    ModelNotFoundError,
    ModelStore,
    PathNotAllowedError,
    PermissionDeniedError,
)

__all__ = [
    # Models
    "ModelEntry",
    "ModelScope",
    "ModelStatus",
    "ModelType",
    "generate_model_id",
    # Schema
    "ensure_schema",
    # Store
    "ModelStore",
    "ModelNotFoundError",
    "DuplicateModelError",
    "PermissionDeniedError",
    "PathNotAllowedError",
]
