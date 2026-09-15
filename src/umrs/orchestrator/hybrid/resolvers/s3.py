"""S3Resolver implementation."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os

from umrs.model_registry.models import ExecutionAuthority, ModelRef, ModelRefType

from .base import ModelRefResolver, ResolutionError, ResolvedModel
from .filesystem import FilesystemResolver

logger = logging.getLogger(__name__)


class S3Resolver(ModelRefResolver):
    """
    S3 model resolver (bonus, not in original spec).

    Handles ModelRefType.S3_REF by downloading to local cache.
    """

    def __init__(
        self,
        cache_dir: str = "/shared/models/s3-cache",
        s3_endpoint: str = "",
    ):
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
            raise ResolutionError(f"S3Resolver cannot handle ref type '{ref.type}'")

        s3_data = ref.to_dict()
        bucket = s3_data.get("bucket", "")
        key = s3_data.get("key", "")

        if not bucket or not key:
            raise ResolutionError("S3 bucket or key is empty")

        cache_path = self._get_cache_path(bucket, key)

        if os.path.exists(cache_path):
            return await self._fs_resolver.resolve(
                ModelRef.filesystem(cache_path),
                authority,
            )

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
            raise ResolutionError(f"S3 download failed: {e}") from e
