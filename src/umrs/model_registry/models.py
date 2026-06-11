"""
Model Registry data models and enums.

Defines:
- ModelScope: system, project, user
- ModelType: llm, embedding, vision
- ModelStatus: staging, available, deprecated
- ExecutionAuthority: slurm, k8s, compose (U-807)
- ModelRefType: filesystem_path, pvc_ref, s3_ref, hf_repo_ref (U-808)
- ModelRef: Unified model reference abstraction (U-808)
- ModelEntry: Core model metadata dataclass
- ModelPolicy: Access control and quotas (U-805)

These dataclasses mirror the database schema and provide type safety.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


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
    - VISION: Vision model for image analysis (SAM3, HMAE, etc.)
    """

    LLM = "llm"
    EMBEDDING = "embedding"
    VISION = "vision"


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


class ExecutionAuthority(Enum):
    """
    Execution authority for job scheduling (U-807).

    This is a hard invariant: chosen once per job, never changes.
    Prevents orchestrator recursion (Slurm launching K8s launching Slurm).

    - SLURM: Submit job to Slurm workload manager (HPC)
    - K8S: Submit job to Kubernetes cluster
    - COMPOSE: Use Docker Compose for local/dev environments
    """

    SLURM = "slurm"
    K8S = "k8s"
    COMPOSE = "compose"


class ModelRefType(Enum):
    """
    Type of model reference (U-808).

    Each type has specific resolution semantics per execution authority.
    """

    FILESYSTEM_PATH = "filesystem_path"  # Local filesystem path
    PVC_REF = "pvc_ref"  # Kubernetes PersistentVolumeClaim
    S3_REF = "s3_ref"  # S3-compatible object storage
    HF_REPO_REF = "hf_repo_ref"  # HuggingFace repository


@dataclass
class ModelRef:
    """
    Unified model reference abstraction (U-808).

    Represents a model's location across different storage systems.
    The resolver maps this to a concrete path based on execution authority.

    Examples:
        - filesystem_path: {"type": "filesystem_path", "path": "/models/qwen-7b"}
        - pvc_ref: {"type": "pvc_ref", "pvc_name": "models-pvc", "subpath": "qwen-7b"}
        - s3_ref: {"type": "s3_ref", "bucket": "models", "key": "qwen-7b/"}
        - hf_repo_ref: {"type": "hf_repo_ref", "repo_id": "Qwen/Qwen2.5-7B-Instruct"}
    """

    ref_type: ModelRefType

    # For filesystem_path
    path: str | None = None

    # For pvc_ref
    pvc_name: str | None = None
    pvc_namespace: str | None = None
    subpath: str | None = None

    # For s3_ref
    bucket: str | None = None
    key: str | None = None
    endpoint_url: str | None = None  # For S3-compatible storage

    # For hf_repo_ref
    repo_id: str | None = None
    revision: str | None = None  # Git ref (branch, tag, commit)

    def __post_init__(self) -> None:
        """Validate model ref after initialization."""
        if isinstance(self.ref_type, str):
            self.ref_type = ModelRefType(self.ref_type)

        # Validate required fields per type
        if self.ref_type == ModelRefType.FILESYSTEM_PATH:
            if not self.path:
                raise ValueError("filesystem_path ref requires 'path'")
        elif self.ref_type == ModelRefType.PVC_REF:
            if not self.pvc_name:
                raise ValueError("pvc_ref requires 'pvc_name'")
        elif self.ref_type == ModelRefType.S3_REF:
            if not self.bucket or not self.key:
                raise ValueError("s3_ref requires 'bucket' and 'key'")
        elif self.ref_type == ModelRefType.HF_REPO_REF and not self.repo_id:
            raise ValueError("hf_repo_ref requires 'repo_id'")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        data: dict[str, Any] = {"type": self.ref_type.value}

        if self.ref_type == ModelRefType.FILESYSTEM_PATH:
            data["path"] = self.path
        elif self.ref_type == ModelRefType.PVC_REF:
            data["pvc_name"] = self.pvc_name
            if self.pvc_namespace:
                data["pvc_namespace"] = self.pvc_namespace
            if self.subpath:
                data["subpath"] = self.subpath
        elif self.ref_type == ModelRefType.S3_REF:
            data["bucket"] = self.bucket
            data["key"] = self.key
            if self.endpoint_url:
                data["endpoint_url"] = self.endpoint_url
        elif self.ref_type == ModelRefType.HF_REPO_REF:
            data["repo_id"] = self.repo_id
            if self.revision:
                data["revision"] = self.revision

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelRef:
        """Create from dictionary."""
        ref_type = ModelRefType(data["type"])
        return cls(
            ref_type=ref_type,
            path=data.get("path"),
            pvc_name=data.get("pvc_name"),
            pvc_namespace=data.get("pvc_namespace"),
            subpath=data.get("subpath"),
            bucket=data.get("bucket"),
            key=data.get("key"),
            endpoint_url=data.get("endpoint_url"),
            repo_id=data.get("repo_id"),
            revision=data.get("revision"),
        )

    @classmethod
    def filesystem(cls, path: str) -> ModelRef:
        """Create a filesystem path reference."""
        return cls(ref_type=ModelRefType.FILESYSTEM_PATH, path=path)

    @classmethod
    def pvc(
        cls,
        pvc_name: str,
        subpath: str | None = None,
        namespace: str | None = None,
    ) -> ModelRef:
        """Create a PVC reference."""
        return cls(
            ref_type=ModelRefType.PVC_REF,
            pvc_name=pvc_name,
            pvc_namespace=namespace,
            subpath=subpath,
        )

    @classmethod
    def s3(
        cls,
        bucket: str,
        key: str,
        endpoint_url: str | None = None,
    ) -> ModelRef:
        """Create an S3 reference."""
        return cls(
            ref_type=ModelRefType.S3_REF,
            bucket=bucket,
            key=key,
            endpoint_url=endpoint_url,
        )

    @classmethod
    def huggingface(
        cls,
        repo_id: str,
        revision: str | None = None,
    ) -> ModelRef:
        """Create a HuggingFace repository reference."""
        return cls(
            ref_type=ModelRefType.HF_REPO_REF,
            repo_id=repo_id,
            revision=revision,
        )


def generate_model_id() -> str:
    """Generate a unique model ID."""
    return f"mdl-{uuid.uuid4().hex[:12]}"


@dataclass
class ModelCapabilities:
    """
    Model capability metadata.

    Describes what a model can do — tool calling, vision, structured output, etc.
    Stored as JSONB in the database for extensibility.
    """

    supports_tool_calling: bool = False
    supports_vision: bool = False
    supports_structured_output: bool = False
    max_output_tokens: int | None = None
    max_position_embeddings: int | None = None
    model_family: str | None = None  # e.g., "qwen", "llama", "mistral"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        d: dict[str, Any] = {
            "supports_tool_calling": self.supports_tool_calling,
            "supports_vision": self.supports_vision,
            "supports_structured_output": self.supports_structured_output,
        }
        if self.max_output_tokens is not None:
            d["max_output_tokens"] = self.max_output_tokens
        if self.max_position_embeddings is not None:
            d["max_position_embeddings"] = self.max_position_embeddings
        if self.model_family is not None:
            d["model_family"] = self.model_family
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCapabilities:
        """Create from dictionary."""
        if not data:
            return cls()
        return cls(
            supports_tool_calling=data.get("supports_tool_calling", False),
            supports_vision=data.get("supports_vision", False),
            supports_structured_output=data.get("supports_structured_output", False),
            max_output_tokens=data.get("max_output_tokens"),
            max_position_embeddings=data.get("max_position_embeddings"),
            model_family=data.get("model_family"),
        )


@dataclass
class ServingProfile:
    """
    Serving configuration for a model.

    vLLM/LiteLLM fields remain the default for LLM and embedding models.
    Backend/service metadata lets UMRS select alternate serving surfaces such
    as BentoML vision without introducing a separate endpoint registry.
    """

    # Backend/service selection
    serving_backend: str | None = None  # vllm, bentoml
    service_kind: str | None = None  # openai, bentoml_vision
    launch_profiles: list[str] = field(default_factory=list)
    container_image: str | None = None
    service_port: int | None = None
    readiness_probe: str | None = None  # models, chat, http
    readiness_path: str | None = None
    invoke_base_path: str | None = None

    # vLLM parameters
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    quantization: str | None = None  # e.g., "awq", "gptq"

    # LiteLLM parameters
    served_model_name: str | None = None  # Override model name in API
    api_base: str | None = None

    # Resource hints
    min_gpu_memory_gb: float | None = None
    recommended_gpu_count: int = 1
    recommended_node_count: int = 1  # >1 for multi-node PP

    # Vision adapter hints
    vision_adapter: str | None = None
    vision_task: str | None = None
    output_format: str | None = None
    co_located: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "serving_backend": self.serving_backend,
            "service_kind": self.service_kind,
            "launch_profiles": list(self.launch_profiles),
            "container_image": self.container_image,
            "service_port": self.service_port,
            "readiness_probe": self.readiness_probe,
            "readiness_path": self.readiness_path,
            "invoke_base_path": self.invoke_base_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "served_model_name": self.served_model_name,
            "api_base": self.api_base,
            "min_gpu_memory_gb": self.min_gpu_memory_gb,
            "recommended_gpu_count": self.recommended_gpu_count,
            "recommended_node_count": self.recommended_node_count,
            "vision_adapter": self.vision_adapter,
            "vision_task": self.vision_task,
            "output_format": self.output_format,
            "co_located": self.co_located,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServingProfile:
        """Create from dictionary."""
        raw_profiles = data.get("launch_profiles") or data.get("runtime_profiles") or []
        launch_profiles = (
            list(raw_profiles)
            if isinstance(raw_profiles, list)
            else str(raw_profiles).split()
        )
        return cls(
            serving_backend=data.get("serving_backend"),
            service_kind=data.get("service_kind"),
            launch_profiles=launch_profiles,
            container_image=data.get("container_image"),
            service_port=data.get("service_port"),
            readiness_probe=data.get("readiness_probe"),
            readiness_path=data.get("readiness_path"),
            invoke_base_path=data.get("invoke_base_path"),
            tensor_parallel_size=data.get("tensor_parallel_size", 1),
            pipeline_parallel_size=data.get("pipeline_parallel_size", 1),
            max_model_len=data.get("max_model_len"),
            gpu_memory_utilization=data.get("gpu_memory_utilization", 0.9),
            dtype=data.get("dtype", "auto"),
            quantization=data.get("quantization"),
            served_model_name=data.get("served_model_name"),
            api_base=data.get("api_base"),
            min_gpu_memory_gb=data.get("min_gpu_memory_gb"),
            recommended_gpu_count=data.get("recommended_gpu_count", 1),
            recommended_node_count=data.get("recommended_node_count", 1),
            vision_adapter=data.get("vision_adapter"),
            vision_task=data.get("vision_task"),
            output_format=data.get("output_format"),
            co_located=data.get("co_located"),
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
    owner_id: str | None = None  # NULL for system models
    project_id: str | None = None  # NULL for system/user models

    # Storage (legacy field for backwards compatibility)
    storage_ref: str = ""  # Filesystem path to model weights

    # ModelRef abstraction (U-808) - preferred over storage_ref
    model_ref: ModelRef | None = None

    # Model metadata
    model_type: ModelType = ModelType.LLM
    huggingface_id: str | None = None  # Source model on HF Hub
    description: str | None = None

    # Serving configuration
    serving_profile: ServingProfile = field(default_factory=ServingProfile)

    # Model capabilities metadata
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    # Execution authority default (U-807)
    execution_authority_default: ExecutionAuthority = ExecutionAuthority.COMPOSE

    # Status
    status: ModelStatus = ModelStatus.AVAILABLE

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None

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
        if isinstance(self.capabilities, dict):
            self.capabilities = ModelCapabilities.from_dict(self.capabilities)
        if isinstance(self.execution_authority_default, str):
            self.execution_authority_default = ExecutionAuthority(
                self.execution_authority_default
            )
        if isinstance(self.model_ref, dict):
            self.model_ref = ModelRef.from_dict(self.model_ref)

        # Validate scope-ownership consistency
        if self.scope == ModelScope.SYSTEM:
            if self.owner_id is not None or self.project_id is not None:
                raise ValueError("System models must have NULL owner_id and project_id")
        elif self.scope == ModelScope.PROJECT:
            if self.project_id is None:
                raise ValueError("Project models must have a project_id")
        elif self.scope == ModelScope.USER and self.owner_id is None:
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "scope": self.scope.value,
            "owner_id": self.owner_id,
            "project_id": self.project_id,
            "storage_ref": self.storage_ref,
            "model_ref": self.model_ref.to_dict() if self.model_ref else None,
            "model_type": self.model_type.value,
            "huggingface_id": self.huggingface_id,
            "description": self.description,
            "serving_profile": self.serving_profile.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "execution_authority_default": self.execution_authority_default.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelEntry:
        """Create from dictionary."""
        model_ref_data = data.get("model_ref")
        return cls(
            model_id=data["model_id"],
            name=data["name"],
            scope=ModelScope(data["scope"]),
            owner_id=data.get("owner_id"),
            project_id=data.get("project_id"),
            storage_ref=data.get("storage_ref", ""),
            model_ref=ModelRef.from_dict(model_ref_data) if model_ref_data else None,
            model_type=ModelType(data.get("model_type", "llm")),
            huggingface_id=data.get("huggingface_id"),
            description=data.get("description"),
            serving_profile=ServingProfile.from_dict(data.get("serving_profile", {})),
            capabilities=ModelCapabilities.from_dict(data.get("capabilities", {})),
            execution_authority_default=ExecutionAuthority(
                data.get("execution_authority_default", "compose")
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
    def from_row(cls, row: tuple) -> ModelEntry:
        """
        Create from database row.

        Expected column order:
        model_id, name, scope, owner_id, project_id, storage_ref,
        model_type, huggingface_id, description, serving_profile,
        status, model_ref_json, execution_authority_default,
        created_at, updated_at, capabilities
        """
        model_ref_data = row[11]  # model_ref_json
        capabilities_data = row[15] if len(row) > 15 else None
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
            model_ref=ModelRef.from_dict(model_ref_data) if model_ref_data else None,
            execution_authority_default=ExecutionAuthority(row[12] or "compose"),
            created_at=row[13],
            updated_at=row[14],
            capabilities=ModelCapabilities.from_dict(capabilities_data or {}),
        )


def generate_policy_id() -> str:
    """Generate a unique policy ID."""
    return f"pol-{uuid.uuid4().hex[:12]}"


@dataclass
class ModelPolicy:
    """
    Model access policy (U-805).

    Defines access control, quotas, and warm pool settings for a model.
    Used primarily for system-level models shared across projects.
    """

    # Identity
    policy_id: str
    model_name: str  # Links to model_catalog_entries.name

    # Access control
    allowed_projects: list[str] = field(default_factory=list)  # Empty = all allowed
    denied_projects: list[str] = field(default_factory=list)  # Takes precedence

    # Quota management
    max_concurrent_endpoints: int | None = None
    max_tokens_per_day: int | None = None
    priority: int = 0

    # Warm pool settings
    warm_pool_min: int = 0
    warm_pool_max: int = 0
    warm_pool_ttl_seconds: int = 3600
    enable_warm_pool: bool = False

    # System model settings
    is_system_model: bool = False
    default_serving_profile: ServingProfile = field(default_factory=ServingProfile)

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        """Convert types after initialization."""
        if isinstance(self.default_serving_profile, dict):
            self.default_serving_profile = ServingProfile.from_dict(
                self.default_serving_profile
            )

    def is_project_allowed(self, project_id: str) -> bool:
        """
        Check if a project is allowed to use this model.

        Deny list takes precedence over allow list.
        Empty allow list means all projects are allowed (except denied).
        """
        if project_id in self.denied_projects:
            return False
        if not self.allowed_projects:
            return True  # Empty = all allowed
        return project_id in self.allowed_projects

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "policy_id": self.policy_id,
            "model_name": self.model_name,
            "allowed_projects": self.allowed_projects,
            "denied_projects": self.denied_projects,
            "max_concurrent_endpoints": self.max_concurrent_endpoints,
            "max_tokens_per_day": self.max_tokens_per_day,
            "priority": self.priority,
            "warm_pool_min": self.warm_pool_min,
            "warm_pool_max": self.warm_pool_max,
            "warm_pool_ttl_seconds": self.warm_pool_ttl_seconds,
            "enable_warm_pool": self.enable_warm_pool,
            "is_system_model": self.is_system_model,
            "default_serving_profile": self.default_serving_profile.to_dict(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelPolicy:
        """Create from dictionary."""
        return cls(
            policy_id=data["policy_id"],
            model_name=data["model_name"],
            allowed_projects=data.get("allowed_projects", []),
            denied_projects=data.get("denied_projects", []),
            max_concurrent_endpoints=data.get("max_concurrent_endpoints"),
            max_tokens_per_day=data.get("max_tokens_per_day"),
            priority=data.get("priority", 0),
            warm_pool_min=data.get("warm_pool_min", 0),
            warm_pool_max=data.get("warm_pool_max", 0),
            warm_pool_ttl_seconds=data.get("warm_pool_ttl_seconds", 3600),
            enable_warm_pool=data.get("enable_warm_pool", False),
            is_system_model=data.get("is_system_model", False),
            default_serving_profile=ServingProfile.from_dict(
                data.get("default_serving_profile", {})
            ),
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
    def from_row(cls, row: tuple) -> ModelPolicy:
        """
        Create from database row.

        Expected column order:
        policy_id, model_name, allowed_projects, denied_projects,
        max_concurrent_endpoints, max_tokens_per_day, priority,
        warm_pool_min, warm_pool_max, warm_pool_ttl_seconds, enable_warm_pool,
        is_system_model, default_serving_profile, created_at, updated_at
        """
        return cls(
            policy_id=row[0],
            model_name=row[1],
            allowed_projects=list(row[2] or []),
            denied_projects=list(row[3] or []),
            max_concurrent_endpoints=row[4],
            max_tokens_per_day=row[5],
            priority=row[6] or 0,
            warm_pool_min=row[7] or 0,
            warm_pool_max=row[8] or 0,
            warm_pool_ttl_seconds=row[9] or 3600,
            enable_warm_pool=row[10] or False,
            is_system_model=row[11] or False,
            default_serving_profile=ServingProfile.from_dict(row[12] or {}),
            created_at=row[13],
            updated_at=row[14],
        )
