"""HuggingFaceResolver implementation (U-1124)."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

from umrs.model_registry.models import ExecutionAuthority, ModelRef, ModelRefType

from .base import ModelRefResolver, ResolutionError, ResolvedModel
from .filesystem import FilesystemResolver

logger = logging.getLogger(__name__)


class HuggingFaceResolver(ModelRefResolver):
    """
    HuggingFace model resolver (U-1124).

    Handles ModelRefType.HF_REPO_REF by:
    1. Checking if model is already cached locally
    2. If not, triggering a download job through UMRS
    3. Returning the cached filesystem path
    """

    def __init__(
        self,
        cache_dir: str = "",
        umrs_client: Any | None = None,
        download_timeout_sec: int = 3600,
    ):
        self.cache_dir = cache_dir or os.getenv(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface/hub")
        )
        self.umrs_client = umrs_client
        self.download_timeout_sec = download_timeout_sec
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

        cached_path = self._get_cache_path(repo_id, revision)

        if os.path.exists(cached_path):
            logger.info(f"Model {repo_id} found in cache: {cached_path}")
            return await self._fs_resolver.resolve(
                ModelRef.filesystem(cached_path),
                authority,
            )

        if not self.umrs_client:
            raise ResolutionError(
                f"Model {repo_id} not in cache and no UMRS client for download"
            )

        logger.info(f"Model {repo_id} not cached, triggering download job")
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

        if "/" in repo_id:
            parts = repo_id.split("/")
            if len(parts) != 2 or not all(parts):
                return False

        return True

    def _get_cache_path(self, repo_id: str, revision: str = "main") -> str:
        """Get expected cache path for a HuggingFace model."""
        safe_repo_id = repo_id.replace("/", "--")
        return os.path.join(self.cache_dir, f"models--{safe_repo_id}")

    async def _download_model(self, repo_id: str, revision: str) -> str:
        """Trigger model download through UMRS."""
        if not self.umrs_client:
            raise ResolutionError("UMRS client required for model download")

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
                raise ResolutionError(f"Download request failed: {response.text}")

            result = response.json()

            job_id = result.get("job_id")
            if job_id:
                await self._wait_for_download(job_id)

            return result.get("path", self._get_cache_path(repo_id, revision))

        except Exception as e:
            raise ResolutionError(f"Model download failed: {e}") from e

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
