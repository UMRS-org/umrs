"""FilesystemResolver implementation (U-1122)."""

from __future__ import annotations

import logging
import os
from datetime import datetime

from umrs.model_registry.models import ExecutionAuthority, ModelRef, ModelRefType

from .base import ModelRefResolver, ResolutionError, ResolvedModel

logger = logging.getLogger(__name__)


class FilesystemResolver(ModelRefResolver):
    """
    Filesystem path resolver (U-1122).

    Handles ModelRefType.FILESYSTEM_PATH by creating bind mounts.
    Primary resolver for Slurm and Docker Compose deployments.
    """

    def __init__(
        self,
        allowed_prefixes: list[str] | None = None,
        container_path: str = "/models",
    ):
        self.allowed_prefixes = allowed_prefixes or [
            "/shared/models",
            "/opt/models",
            "/models",
        ]
        self.container_path = container_path

    @property
    def ref_type(self) -> ModelRefType:
        return ModelRefType.FILESYSTEM_PATH

    async def resolve(
        self,
        ref: ModelRef,
        authority: ExecutionAuthority,
    ) -> ResolvedModel:
        """Resolve filesystem path to bind mount."""
        if ref.type != ModelRefType.FILESYSTEM_PATH:
            raise ResolutionError(
                f"FilesystemResolver cannot handle ref type '{ref.type}'"
            )

        path = ref.path
        if not path:
            raise ResolutionError("Filesystem path is empty")

        if not await self.validate(ref):
            raise ResolutionError(f"Path not allowed: {path}")

        if authority == ExecutionAuthority.K8S:
            logger.warning(
                f"Filesystem path {path} used with K8s authority - "
                "ensure path is accessible from cluster nodes"
            )

        path = os.path.normpath(path)

        return ResolvedModel(
            mount_type="bind",
            host_path=path,
            container_path=self.container_path,
            read_only=True,
            model_name=os.path.basename(path),
            resolved_at=datetime.now(),
            metadata={"source": "filesystem"},
        )

    async def validate(self, ref: ModelRef) -> bool:
        """Validate filesystem path exists and is allowed."""
        if ref.type != ModelRefType.FILESYSTEM_PATH:
            return False

        path = ref.path
        if not path:
            return False

        path_normalized = os.path.normpath(path)
        allowed = any(
            path_normalized.startswith(prefix)
            for prefix in self.allowed_prefixes
        )

        if not allowed:
            logger.warning(f"Path {path} not in allowed prefixes: {self.allowed_prefixes}")
            return False

        if not os.path.exists(path):
            logger.warning(f"Path does not exist: {path}")
            return False

        return True
