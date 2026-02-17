"""
ModelRefResolver Interface and Implementations (U-1121 to U-1124).

ModelRefResolver translates ModelRef (from model registry) into concrete
mount specifications for the target execution authority.

Implementations:
- FilesystemResolver: Direct filesystem paths (Slurm/Compose)
- PVCResolver: Kubernetes PersistentVolumeClaim mounts
- HuggingFaceResolver: Triggers download job, returns cached path
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from umrs.model_registry.models import ModelRef, ModelRefType, ExecutionAuthority

logger = logging.getLogger(__name__)


@dataclass
class ResolvedModel:
    """
    Result of ModelRef resolution.

    Contains all information needed to mount the model in a container.
    """

    # Mount configuration
    mount_type: str  # "bind", "pvc", "volume"
    host_path: str = ""  # For bind mounts
    pvc_name: str = ""  # For PVC mounts
    subpath: str = ""  # Subpath within volume
    container_path: str = "/models"  # Mount point in container
    read_only: bool = True

    # Model info
    model_name: str = ""
    resolved_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_mount_spec(self) -> Dict[str, Any]:
        """Convert to MountSpec-compatible dict."""
        from .wrappers import MountSpec

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


# =============================================================================
# FilesystemResolver Implementation (U-1122)
# =============================================================================


class FilesystemResolver(ModelRefResolver):
    """
    Filesystem path resolver (U-1122).

    Handles ModelRefType.FILESYSTEM_PATH by creating bind mounts.
    Primary resolver for Slurm and Docker Compose deployments.
    """

    def __init__(
        self,
        allowed_prefixes: Optional[list[str]] = None,
        container_path: str = "/models",
    ):
        """
        Initialize filesystem resolver.

        Args:
            allowed_prefixes: Valid path prefixes (security check).
            container_path: Default mount point in container.
        """
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

        # Validate path is allowed
        if not await self.validate(ref):
            raise ResolutionError(f"Path not allowed: {path}")

        # Check authority compatibility
        if authority == ExecutionAuthority.K8S:
            logger.warning(
                f"Filesystem path {path} used with K8s authority - "
                "ensure path is accessible from cluster nodes"
            )

        # Normalize path
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

        # Check prefix allowlist
        path_normalized = os.path.normpath(path)
        allowed = any(
            path_normalized.startswith(prefix)
            for prefix in self.allowed_prefixes
        )

        if not allowed:
            logger.warning(f"Path {path} not in allowed prefixes: {self.allowed_prefixes}")
            return False

        # Check existence
        if not os.path.exists(path):
            logger.warning(f"Path does not exist: {path}")
            return False

        return True


# =============================================================================
# PVCResolver Implementation (U-1123)
# =============================================================================


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
        """
        Initialize PVC resolver.

        Args:
            namespace: K8s namespace for PVC lookup.
            container_path: Default mount point in container.
        """
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

        # PVC refs only valid for K8s
        if authority != ExecutionAuthority.K8S:
            raise IncompatibleAuthorityError(
                f"PVC refs only valid for K8s authority, got '{authority.value}'"
            )

        pvc_data = ref.to_dict()
        pvc_name = pvc_data.get("claim", "")
        subpath = pvc_data.get("subpath", "")

        if not pvc_name:
            raise ResolutionError("PVC claim name is empty")

        # Validate PVC exists
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


# =============================================================================
# HuggingFaceResolver Implementation (U-1124)
# =============================================================================


class HuggingFaceResolver(ModelRefResolver):
    """
    HuggingFace model resolver (U-1124).

    Handles ModelRefType.HF_REPO_REF by:
    1. Checking if model is already cached locally
    2. If not, triggering a download job through UMRS
    3. Returning the cached filesystem path

    Download jobs run in special "egress-allowed" mode with audit logging.
    """

    def __init__(
        self,
        cache_dir: str = "",
        umrs_client: Optional[Any] = None,
        download_timeout_sec: int = 3600,
    ):
        """
        Initialize HuggingFace resolver.

        Args:
            cache_dir: Local cache directory for downloaded models.
            umrs_client: UMRS client for triggering download jobs.
            download_timeout_sec: Timeout for model downloads.
        """
        self.cache_dir = cache_dir or os.getenv(
            "HF_HOME",
            os.path.expanduser("~/.cache/huggingface/hub")
        )
        self.umrs_client = umrs_client
        self.download_timeout_sec = download_timeout_sec

        # Filesystem resolver for cached models
        self._fs_resolver = FilesystemResolver(
            allowed_prefixes=[self.cache_dir, "/shared/models"],
        )

    @property
    def ref_type(self) -> ModelRefType:
        return ModelRefType.HF_REPO_REF

    async def resolve(
        self,
        ref: ModelRef,
        authority: ExecutionAuthority,
    ) -> ResolvedModel:
        """Resolve HuggingFace repo to local path (downloading if needed)."""
        if ref.type != ModelRefType.HF_REPO_REF:
            raise ResolutionError(
                f"HuggingFaceResolver cannot handle ref type '{ref.type}'"
            )

        hf_data = ref.to_dict()
        repo_id = hf_data.get("repo_id", "")
        revision = hf_data.get("revision", "main")

        if not repo_id:
            raise ResolutionError("HuggingFace repo_id is empty")

        # Check if already cached
        cached_path = self._get_cache_path(repo_id, revision)

        if os.path.exists(cached_path):
            logger.info(f"Model {repo_id} found in cache: {cached_path}")
            return await self._fs_resolver.resolve(
                ModelRef.filesystem(cached_path),
                authority,
            )

        # Need to download - this requires UMRS client
        if not self.umrs_client:
            raise ResolutionError(
                f"Model {repo_id} not in cache and no UMRS client for download"
            )

        logger.info(f"Model {repo_id} not cached, triggering download job")

        # Trigger download through UMRS
        cached_path = await self._download_model(repo_id, revision)

        return await self._fs_resolver.resolve(
            ModelRef.filesystem(cached_path),
            authority,
        )

    async def validate(self, ref: ModelRef) -> bool:
        """Validate HuggingFace repo reference."""
        if ref.type != ModelRefType.HF_REPO_REF:
            return False

        hf_data = ref.to_dict()
        repo_id = hf_data.get("repo_id", "")

        if not repo_id:
            return False

        # Basic format validation: org/model or just model
        if "/" in repo_id:
            parts = repo_id.split("/")
            if len(parts) != 2 or not all(parts):
                return False

        # Could optionally check HuggingFace Hub API here
        # but that requires network access

        return True

    def _get_cache_path(self, repo_id: str, revision: str = "main") -> str:
        """Get expected cache path for a HuggingFace model."""
        # HuggingFace uses models--org--repo format
        safe_repo_id = repo_id.replace("/", "--")
        return os.path.join(self.cache_dir, f"models--{safe_repo_id}")

    async def _download_model(self, repo_id: str, revision: str) -> str:
        """
        Trigger model download through UMRS.

        Downloads run in special jobs with:
        - Egress enabled (TRANSFORMERS_OFFLINE=0)
        - Audit logging of the download
        - Cached to shared storage
        """
        if not self.umrs_client:
            raise ResolutionError("UMRS client required for model download")

        # Create download job request
        # This uses the model download API added in Phase 1 (U-818)
        try:
            response = await self.umrs_client.post(
                "/umrs/models/download",
                json={
                    "repo_id": repo_id,
                    "revision": revision,
                    "source": "huggingface",
                },
            )

            if response.status_code != 200:
                raise ResolutionError(
                    f"Download request failed: {response.text}"
                )

            result = response.json()

            # Poll for completion
            job_id = result.get("job_id")
            if job_id:
                await self._wait_for_download(job_id)

            return result.get("path", self._get_cache_path(repo_id, revision))

        except Exception as e:
            raise ResolutionError(f"Model download failed: {e}")

    async def _wait_for_download(self, job_id: str) -> None:
        """Wait for download job to complete."""
        start = datetime.now()

        while True:
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed > self.download_timeout_sec:
                raise ResolutionError(
                    f"Download job {job_id} timed out after {self.download_timeout_sec}s"
                )

            try:
                response = await self.umrs_client.get(f"/umrs/jobs/{job_id}")
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "unknown")

                    if status == "completed":
                        return
                    elif status in ("failed", "cancelled"):
                        raise ResolutionError(
                            f"Download job {job_id} {status}: {data.get('error')}"
                        )
            except ResolutionError:
                raise
            except Exception as e:
                logger.warning(f"Error checking download status: {e}")

            await asyncio.sleep(10)


# =============================================================================
# S3Resolver Implementation (Bonus)
# =============================================================================


class S3Resolver(ModelRefResolver):
    """
    S3 model resolver (bonus, not in original spec).

    Handles ModelRefType.S3_REF by downloading to local cache.
    Similar pattern to HuggingFaceResolver.
    """

    def __init__(
        self,
        cache_dir: str = "/shared/models/s3-cache",
        s3_endpoint: str = "",
    ):
        """
        Initialize S3 resolver.

        Args:
            cache_dir: Local cache directory.
            s3_endpoint: S3 endpoint URL.
        """
        self.cache_dir = cache_dir
        self.s3_endpoint = s3_endpoint or os.getenv("S3_ENDPOINT", "")

        self._fs_resolver = FilesystemResolver(
            allowed_prefixes=[cache_dir],
        )

    @property
    def ref_type(self) -> ModelRefType:
        return ModelRefType.S3_REF

    async def resolve(
        self,
        ref: ModelRef,
        authority: ExecutionAuthority,
    ) -> ResolvedModel:
        """Resolve S3 reference to local path."""
        if ref.type != ModelRefType.S3_REF:
            raise ResolutionError(
                f"S3Resolver cannot handle ref type '{ref.type}'"
            )

        s3_data = ref.to_dict()
        bucket = s3_data.get("bucket", "")
        key = s3_data.get("key", "")

        if not bucket or not key:
            raise ResolutionError("S3 bucket or key is empty")

        # Check cache
        cache_path = self._get_cache_path(bucket, key)

        if os.path.exists(cache_path):
            return await self._fs_resolver.resolve(
                ModelRef.filesystem(cache_path),
                authority,
            )

        # Download from S3
        await self._download_from_s3(bucket, key, cache_path)

        return await self._fs_resolver.resolve(
            ModelRef.filesystem(cache_path),
            authority,
        )

    async def validate(self, ref: ModelRef) -> bool:
        """Validate S3 reference."""
        if ref.type != ModelRefType.S3_REF:
            return False

        s3_data = ref.to_dict()
        return bool(s3_data.get("bucket")) and bool(s3_data.get("key"))

    def _get_cache_path(self, bucket: str, key: str) -> str:
        """Get local cache path for S3 object."""
        # Hash the S3 path to avoid path issues
        path_hash = hashlib.sha256(f"{bucket}/{key}".encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, bucket, path_hash)

    async def _download_from_s3(
        self,
        bucket: str,
        key: str,
        local_path: str,
    ) -> None:
        """Download model from S3 to local cache."""
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            # Use AWS CLI for simplicity (boto3 would be better)
            cmd = ["aws", "s3", "cp", f"s3://{bucket}/{key}", local_path]

            if self.s3_endpoint:
                cmd.extend(["--endpoint-url", self.s3_endpoint])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise ResolutionError(f"S3 download failed: {stderr.decode()}")

        except Exception as e:
            raise ResolutionError(f"S3 download failed: {e}")


# =============================================================================
# Resolver Registry
# =============================================================================


class ResolverRegistry:
    """
    Registry of ModelRef resolvers.

    Dispatches resolution to the appropriate resolver based on ModelRefType.
    """

    def __init__(self):
        self._resolvers: Dict[ModelRefType, ModelRefResolver] = {}

    def register(self, resolver: ModelRefResolver) -> None:
        """Register a resolver for its ref type."""
        self._resolvers[resolver.ref_type] = resolver

    def get(self, ref_type: ModelRefType) -> Optional[ModelRefResolver]:
        """Get resolver for a ref type."""
        return self._resolvers.get(ref_type)

    async def resolve(
        self,
        ref: ModelRef,
        authority: ExecutionAuthority,
    ) -> ResolvedModel:
        """Resolve a ModelRef using the appropriate resolver."""
        resolver = self._resolvers.get(ref.type)

        if not resolver:
            raise ResolutionError(f"No resolver registered for ref type '{ref.type}'")

        return await resolver.resolve(ref, authority)

    @classmethod
    def default(cls) -> "ResolverRegistry":
        """Create registry with default resolvers."""
        registry = cls()
        registry.register(FilesystemResolver())
        registry.register(PVCResolver())
        registry.register(HuggingFaceResolver())
        registry.register(S3Resolver())
        return registry
