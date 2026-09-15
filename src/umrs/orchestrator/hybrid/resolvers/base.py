"""
ModelRefResolver base classes and shared types (U-1121).

ResolvedModel, ModelRefResolver ABC, ResolutionError, IncompatibleAuthorityError.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from umrs.model_registry.models import ExecutionAuthority, ModelRef, ModelRefType

logger = logging.getLogger(__name__)


@dataclass
class ResolvedModel:
    """
    Result of ModelRef resolution.

    Contains all information needed to mount the model in a container.
    """

    mount_type: str  # "bind", "pvc", "volume"
    host_path: str = ""
    pvc_name: str = ""
    subpath: str = ""
    container_path: str = "/models"
    read_only: bool = True
    model_name: str = ""
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_mount_spec(self) -> dict[str, Any]:
        """Convert to MountSpec-compatible dict."""
        from ..wrappers import MountSpec

        if self.mount_type == "bind":
            return MountSpec(
                source=self.host_path,
                target=self.container_path,
                mount_type="bind",
                read_only=self.read_only,
            )
        elif self.mount_type == "pvc":
            return MountSpec(
                source=self.pvc_name,
                target=self.container_path,
                mount_type="pvc",
                read_only=self.read_only,
                options={"subPath": self.subpath} if self.subpath else {},
            )
        else:
            return MountSpec(
                source=self.host_path or self.pvc_name,
                target=self.container_path,
                mount_type=self.mount_type,
                read_only=self.read_only,
            )


class ModelRefResolver(ABC):
    """
    Abstract base for model reference resolution (U-1121).

    Responsibilities:
    - Translate ModelRef to mount configuration
    - Validate model exists/is accessible
    - Trigger downloads if needed (HuggingFace)
    """

    @property
    @abstractmethod
    def ref_type(self) -> ModelRefType:
        """Return the ModelRefType this resolver handles."""
        ...

    @abstractmethod
    async def resolve(
        self,
        ref: ModelRef,
        authority: ExecutionAuthority,
    ) -> ResolvedModel:
        """
        Resolve ModelRef to mount configuration.

        Args:
            ref: The model reference to resolve.
            authority: Target execution authority.

        Returns:
            ResolvedModel with mount configuration.

        Raises:
            ResolutionError: If resolution fails.
        """
        ...

    @abstractmethod
    async def validate(self, ref: ModelRef) -> bool:
        """
        Validate that the model reference is valid and accessible.

        Args:
            ref: The model reference to validate.

        Returns:
            True if valid and accessible.
        """
        ...


class ResolutionError(Exception):
    """Error during model reference resolution."""

    pass


class IncompatibleAuthorityError(ResolutionError):
    """ModelRef not compatible with target execution authority."""

    pass
