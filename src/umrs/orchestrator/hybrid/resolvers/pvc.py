"""PVCResolver implementation (U-1123)."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

from umrs.model_registry.models import ExecutionAuthority, ModelRef, ModelRefType

from .base import (
    IncompatibleAuthorityError,
    ModelRefResolver,
    ResolutionError,
    ResolvedModel,
)

logger = logging.getLogger(__name__)


class PVCResolver(ModelRefResolver):
    """
    Kubernetes PVC resolver (U-1123).

    Handles ModelRefType.PVC_REF by creating PVC volume mounts.
    Only valid for K8s execution authority.
    """

    def __init__(
        self,
        namespace: str = "rivault-system",
        container_path: str = "/models",
    ):
        self.namespace = namespace or os.getenv("K8S_NAMESPACE", "rivault-system")
        self.container_path = container_path

    @property
    def ref_type(self) -> ModelRefType:
        return ModelRefType.PVC_REF

    async def resolve(
        self,
        ref: ModelRef,
        authority: ExecutionAuthority,
    ) -> ResolvedModel:
        """Resolve PVC reference to volume mount."""
        if ref.type != ModelRefType.PVC_REF:
            raise ResolutionError(
                f"PVCResolver cannot handle ref type '{ref.type}'"
            )

        if authority != ExecutionAuthority.K8S:
            raise IncompatibleAuthorityError(
                f"PVC refs only valid for K8s authority, got '{authority.value}'"
            )

        pvc_data = ref.to_dict()
        pvc_name = pvc_data.get("claim", "")
        subpath = pvc_data.get("subpath", "")

        if not pvc_name:
            raise ResolutionError("PVC claim name is empty")

        if not await self.validate(ref):
            raise ResolutionError(f"PVC not found: {pvc_name}")

        return ResolvedModel(
            mount_type="pvc",
            pvc_name=pvc_name,
            subpath=subpath,
            container_path=self.container_path,
            read_only=True,
            model_name=subpath.split("/")[-1] if subpath else pvc_name,
            resolved_at=datetime.now(),
            metadata={
                "source": "pvc",
                "namespace": self.namespace,
            },
        )

    async def validate(self, ref: ModelRef) -> bool:
        """Validate PVC exists in namespace."""
        if ref.type != ModelRefType.PVC_REF:
            return False

        pvc_data = ref.to_dict()
        pvc_name = pvc_data.get("claim", "")

        if not pvc_name:
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                "kubectl", "get", "pvc", pvc_name,
                "--namespace", self.namespace,
                "-o", "name",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return proc.returncode == 0 and stdout.decode().strip() != ""
        except Exception as e:
            logger.warning(f"Failed to validate PVC {pvc_name}: {e}")
            return False
