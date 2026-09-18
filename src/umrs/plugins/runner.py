"""
Plugin runner base class (X-102).

Defines the interface for executing plugins in sandboxed environments.
Concrete implementations for Slurm and K8s extend this base.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .models import PluginResult, PluginSpec, PluginStatus

if TYPE_CHECKING:
    from .audit import PluginAuditLog

logger = logging.getLogger(__name__)


# Base container image for plugin execution
DEFAULT_PLUGIN_IMAGE = os.environ.get(
    "PLUGIN_RUNNER_IMAGE", "ghcr.io/rivault/plugin-runner:latest"
)

# Approved output destinations
APPROVED_OUTPUT_PREFIXES = [
    "s3://",  # S3/MinIO storage
    "pgvector://",  # Vector database
    "umrs://",  # UMRS internal storage
]


class PluginRunner(abc.ABC):
    """
    Abstract base class for plugin execution (X-102).

    Handles plugin lifecycle:
    1. Validate plugin specification
    2. Prepare execution environment
    3. Execute plugin code in sandbox
    4. Collect results and cleanup
    """

    def __init__(
        self,
        audit_log: PluginAuditLog | None = None,
        base_image: str = DEFAULT_PLUGIN_IMAGE,
    ):
        """
        Initialize plugin runner.

        Args:
            audit_log: Optional audit logger for tracking execution.
            base_image: Base container image for plugin execution.
        """
        self.audit_log = audit_log
        self.base_image = base_image
        self._active_plugins: dict[str, PluginSpec] = {}

    def generate_plugin_id(self) -> str:
        """Generate unique plugin execution ID."""
        return f"plugin-{uuid.uuid4().hex[:12]}"

    async def _require_git_ref(self, uri: str, ref: str) -> None:
        """OI-backed git existence check: repo must be reachable AND the ref
        must be a resolvable commit prefix (never trust-only-the-form)."""
        if not uri or not ref:
            raise ValueError("Git source requires source_uri and source_ref")
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "ls-remote",
                uri,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20)
        except TimeoutError as original:
            if proc:
                proc.kill()
                await proc.wait()
            raise ValueError(f"git ls-remote timed out for {uri!r}") from original
        except FileNotFoundError as original:
            raise ValueError(f"git binary unavailable in runner image: {original}") from original
        if proc.returncode != 0:
            raise ValueError(
                f"git ls-remote failed for {uri!r}: {stderr.decode(errors='ignore').strip()}"
            )
        refs = stdout.decode(errors="ignore").strip().splitlines()
        if not any(line.startswith(ref) for line in refs):
            raise ValueError(
                f"git commit/branch {ref!r} not found in {uri!r} (ls-remote got {len(refs)} refs; none matched prefix)"
            )

    async def _require_s3_object(self, uri: str) -> None:
        """OI-backed s3 existence check: `s3://bucket/key.whl` must resolve to
        an existing object (404-family errors become a named ValueError)."""
        rest = uri.removeprefix("s3://") if uri.startswith("s3://") else ""
        if not rest or "/" not in rest:
            raise ValueError(f"invalid s3 uri: {uri!r}")
        bucket, _, key = rest.partition("/")
        if not bucket or not key:
            raise ValueError(f"invalid s3 uri: {uri!r}")
        from umrs.shared.storage.s3_config import get_s3_config

        client = get_s3_config().get_client()
        from botocore.exceptions import ClientError

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, client.head_object, {"Bucket": bucket, "Key": key}
            )
        except ClientError as exc:
            raise ValueError(
                f"s3 object not found/unreachable: s3://{bucket}/{key} ({exc})"
            ) from exc

    async def validate_spec(self, spec: PluginSpec) -> None:
        """
        Validate plugin specification before execution.

        Args:
            spec: Plugin specification to validate.

        Raises:
            ValueError: If specification is invalid.
        """
        # Check source accessibility (TODO-era sweep 2026-09-19: real checks,
        # never fabricated pass — each failure raises ValueError named for
        # the ground that failed).
        if spec.source_type.value == "git_repo":
            await self._require_git_ref(spec.source_uri, spec.source_ref or "")
        elif spec.source_type.value == "s3_wheel":
            await self._require_s3_object(spec.source_uri)

        # Validate output destinations
        for dest in spec.output_destinations:
            if not any(dest.startswith(prefix) for prefix in APPROVED_OUTPUT_PREFIXES):
                raise ValueError(
                    f"Output destination not approved: {dest}. "
                    f"Must start with: {APPROVED_OUTPUT_PREFIXES}"
                )

        # Validate resource limits
        if spec.gpu_count > 8:
            raise ValueError("Maximum 8 GPUs allowed per plugin")

        # Validate entrypoint format
        parts = spec.entrypoint.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid entrypoint format: {spec.entrypoint}. "
                "Expected module:function"
            )

    async def execute(self, spec: PluginSpec) -> PluginResult:
        """
        Execute a plugin with the given specification.

        Args:
            spec: Plugin specification.

        Returns:
            PluginResult with execution outcome.
        """
        plugin_id = self.generate_plugin_id()
        started_at = datetime.now(timezone.utc)

        # Log submission
        if self.audit_log:
            await self.audit_log.log_event(
                plugin_id=plugin_id,
                event_type="submitted",
                user_id=spec.user_id,
                project_id=spec.project_id,
                details={"spec_hash": spec.spec_hash()},
            )

        try:
            # Validate specification
            await self.validate_spec(spec)

            # Track active plugin
            self._active_plugins[plugin_id] = spec

            # Log start
            if self.audit_log:
                await self.audit_log.log_event(
                    plugin_id=plugin_id,
                    event_type="started",
                    user_id=spec.user_id,
                    project_id=spec.project_id,
                )

            # Execute in backend
            result = await self._execute_backend(plugin_id, spec)
            result.started_at = started_at
            result.ended_at = datetime.now(timezone.utc)

            # Log completion
            if self.audit_log:
                await self.audit_log.log_event(
                    plugin_id=plugin_id,
                    event_type="succeeded"
                    if result.status == PluginStatus.SUCCEEDED
                    else "failed",
                    user_id=spec.user_id,
                    project_id=spec.project_id,
                    details={
                        "exit_code": result.exit_code,
                        "duration_seconds": result.duration_seconds,
                        "output_files": result.output_files,
                    },
                )

            return result

        except Exception as e:
            logger.exception(f"Plugin execution failed: {plugin_id}")

            # Log failure
            if self.audit_log:
                await self.audit_log.log_event(
                    plugin_id=plugin_id,
                    event_type="failed",
                    user_id=spec.user_id,
                    project_id=spec.project_id,
                    details={"error": str(e)},
                )

            return PluginResult(
                plugin_id=plugin_id,
                status=PluginStatus.FAILED,
                error_message=str(e),
                started_at=started_at,
                ended_at=datetime.now(timezone.utc),
            )

        finally:
            # Cleanup
            self._active_plugins.pop(plugin_id, None)

    @abc.abstractmethod
    async def _execute_backend(
        self,
        plugin_id: str,
        spec: PluginSpec,
    ) -> PluginResult:
        """
        Execute plugin in the specific backend.

        Must be implemented by subclasses.

        Args:
            plugin_id: Unique plugin execution ID.
            spec: Plugin specification.

        Returns:
            PluginResult with execution outcome.
        """
        ...

    @abc.abstractmethod
    async def cancel(self, plugin_id: str) -> bool:
        """
        Cancel a running plugin.

        Args:
            plugin_id: Plugin execution ID.

        Returns:
            True if cancellation succeeded.
        """
        ...

    @abc.abstractmethod
    async def get_status(self, plugin_id: str) -> PluginStatus | None:
        """
        Get current status of a plugin execution.

        Args:
            plugin_id: Plugin execution ID.

        Returns:
            Current status or None if not found.
        """
        ...

    def _generate_container_env(self, spec: PluginSpec) -> dict[str, str]:
        """Generate environment variables for container."""
        env = {
            "PLUGIN_ENTRYPOINT": spec.entrypoint,
            "PLUGIN_SOURCE_TYPE": spec.source_type.value,
            "PLUGIN_SOURCE_URI": spec.source_uri,
            "PLUGIN_TIMEOUT": str(spec.timeout_seconds),
        }

        if spec.source_ref:
            env["PLUGIN_SOURCE_REF"] = spec.source_ref

        if spec.project_id:
            env["PLUGIN_PROJECT_ID"] = spec.project_id

        if spec.user_id:
            env["PLUGIN_USER_ID"] = spec.user_id

        # Add user-provided env vars (filtered)
        for key, value in spec.env_vars.items():
            # Prevent overriding system vars — but let UMRS_* vars
            # (set by the step adapter) through unprefixed so the
            # in-container step executor can read them directly.
            if key.startswith("PLUGIN_"):
                continue  # reserved namespace — skip
            if key.startswith("UMRS_"):
                env[key] = value  # step-adapter contract — pass as-is
            else:
                env[f"USER_{key}"] = value

        return env
