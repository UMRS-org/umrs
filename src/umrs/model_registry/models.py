"""
Model Registry data models and enums.

Defines:
- ModelScope: system, project, user
- ModelType: llm, embedding
- ModelStatus: staging, available, deprecated
- ModelEntry: Core model metadata dataclass

These dataclasses mirror the database schema and provide type safety.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class ModelScope(Enum):
    """
    Scope of model visibility and ownership.

    - SYSTEM: Admin-managed, visible to all users
    - PROJECT: Project-owned, visible to project members
    - USER: User-owned, visible only to owner
    """

    SYSTEM = "system"
    PROJECT = "project"
    USER = "user"


class ModelType(Enum):
    """
    Type of model for inference.

    - LLM: Large language model for text generation (vLLM)
    - EMBEDDING: Embedding model for vector representations
    """

    LLM = "llm"
    EMBEDDING = "embedding"


class ModelStatus(Enum):
    """
    Model availability status.

    - STAGING: Being downloaded/prepared, not yet available
    - AVAILABLE: Ready for use
    - DEPRECATED: Marked for removal, still usable
    """

    STAGING = "staging"
    AVAILABLE = "available"
    DEPRECATED = "deprecated"


def generate_model_id() -> str:
    """Generate a unique model ID."""
    return f"mdl-{uuid.uuid4().hex[:12]}"


@dataclass
class ServingProfile:
    """
    vLLM/LiteLLM serving configuration for a model.

    These parameters are passed to vLLM when provisioning an endpoint.
    """

    # vLLM parameters
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    quantization: Optional[str] = None  # e.g., "awq", "gptq"

    # LiteLLM parameters
    served_model_name: Optional[str] = None  # Override model name in API
    api_base: Optional[str] = None

    # Resource hints
    min_gpu_memory_gb: Optional[float] = None
    recommended_gpu_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "served_model_name": self.served_model_name,
            "api_base": self.api_base,
            "min_gpu_memory_gb": self.min_gpu_memory_gb,
            "recommended_gpu_count": self.recommended_gpu_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServingProfile":
        """Create from dictionary."""
        return cls(
            tensor_parallel_size=data.get("tensor_parallel_size", 1),
            max_model_len=data.get("max_model_len"),
            gpu_memory_utilization=data.get("gpu_memory_utilization", 0.9),
            dtype=data.get("dtype", "auto"),
            quantization=data.get("quantization"),
            served_model_name=data.get("served_model_name"),
            api_base=data.get("api_base"),
            min_gpu_memory_gb=data.get("min_gpu_memory_gb"),
            recommended_gpu_count=data.get("recommended_gpu_count", 1),
        )


@dataclass
class ModelEntry:
    """
    Model catalog entry (U-801).

    Represents a registered model in the UMRS model registry.
    Maps to the model_catalog_entries database table.
    """

    # Identity
    model_id: str
    name: str

    # Ownership & scope
    scope: ModelScope
    owner_id: Optional[str] = None  # NULL for system models
    project_id: Optional[str] = None  # NULL for system/user models

    # Storage
    storage_ref: str = ""  # Filesystem path to model weights

    # Model metadata
    model_type: ModelType = ModelType.LLM
    huggingface_id: Optional[str] = None  # Source model on HF Hub
    description: Optional[str] = None

    # Serving configuration
    serving_profile: ServingProfile = field(default_factory=ServingProfile)

    # Status
    status: ModelStatus = ModelStatus.AVAILABLE

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Validate model entry after initialization."""
        # Convert string enums if needed
        if isinstance(self.scope, str):
            self.scope = ModelScope(self.scope)
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)
        if isinstance(self.status, str):
            self.status = ModelStatus(self.status)
        if isinstance(self.serving_profile, dict):
            self.serving_profile = ServingProfile.from_dict(self.serving_profile)

        # Validate scope-ownership consistency
        if self.scope == ModelScope.SYSTEM:
            if self.owner_id is not None or self.project_id is not None:
                raise ValueError("System models must have NULL owner_id and project_id")
        elif self.scope == ModelScope.PROJECT:
            if self.project_id is None:
                raise ValueError("Project models must have a project_id")
        elif self.scope == ModelScope.USER:
            if self.owner_id is None:
                raise ValueError("User models must have an owner_id")

    @property
    def display_name(self) -> str:
        """Human-readable name with scope prefix."""
        if self.scope == ModelScope.SYSTEM:
            return f"system/{self.name}"
        elif self.scope == ModelScope.PROJECT:
            return f"project:{self.project_id}/{self.name}"
        else:
            return f"user:{self.owner_id}/{self.name}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "scope": self.scope.value,
            "owner_id": self.owner_id,
            "project_id": self.project_id,
            "storage_ref": self.storage_ref,
            "model_type": self.model_type.value,
            "huggingface_id": self.huggingface_id,
            "description": self.description,
            "serving_profile": self.serving_profile.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEntry":
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            name=data["name"],
            scope=ModelScope(data["scope"]),
            owner_id=data.get("owner_id"),
            project_id=data.get("project_id"),
            storage_ref=data.get("storage_ref", ""),
            model_type=ModelType(data.get("model_type", "llm")),
            huggingface_id=data.get("huggingface_id"),
            description=data.get("description"),
            serving_profile=ServingProfile.from_dict(
                data.get("serving_profile", {})
            ),
            status=ModelStatus(data.get("status", "available")),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else None
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at")
                else None
            ),
        )

    @classmethod
    def from_row(cls, row: tuple) -> "ModelEntry":
        """
        Create from database row.

        Expected column order:
        model_id, name, scope, owner_id, project_id, storage_ref,
        model_type, huggingface_id, description, serving_profile,
        status, created_at, updated_at
        """
        return cls(
            model_id=row[0],
            name=row[1],
            scope=ModelScope(row[2]),
            owner_id=row[3],
            project_id=row[4],
            storage_ref=row[5] or "",
            model_type=ModelType(row[6]),
            huggingface_id=row[7],
            description=row[8],
            serving_profile=ServingProfile.from_dict(row[9] or {}),
            status=ModelStatus(row[10]),
            created_at=row[11],
            updated_at=row[12],
        )
