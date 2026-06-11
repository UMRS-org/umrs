"""
Plugin specification and data models (X-101).

Defines how users specify plugin code and configuration.
Supports Git repositories and S3 wheel paths as sources.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PluginSourceType(Enum):
    """Types of plugin sources."""

    GIT_REPO = "git_repo"  # Git repository + commit
    S3_WHEEL = "s3_wheel"  # S3 path to wheel file
    LOCAL_PATH = "local_path"  # Local filesystem path (dev only)


class PluginStatus(Enum):
    """Status of a plugin execution."""

    PENDING = "pending"  # Queued for execution
    VALIDATING = "validating"  # Checking plugin spec
    RUNNING = "running"  # Execution in progress
    SUCCEEDED = "succeeded"  # Completed successfully
    FAILED = "failed"  # Execution failed
    CANCELLED = "cancelled"  # User cancelled


@dataclass
class PluginSpec:
    """
    Specification for a user-provided plugin (X-101).

    Plugin sources:
    - Git: Clone repo at specific commit, run entrypoint
    - S3: Download wheel, install in sandbox, run entrypoint
    - Local: Mount path directly (development only)

    Examples:
        # Git repository
        PluginSpec(
            source_type=PluginSourceType.GIT_REPO,
            source_uri="https://github.com/org/plugin.git",
            source_ref="abc123",  # commit hash
            entrypoint="my_plugin.main:run",
        )

        # S3 wheel
        PluginSpec(
            source_type=PluginSourceType.S3_WHEEL,
            source_uri="s3://bucket/plugins/my_plugin-1.0.0-py3-none-any.whl",
            entrypoint="my_plugin.main:run",
        )
    """

    source_type: PluginSourceType
    source_uri: str
    entrypoint: str  # module:function format

    # Optional commit/version reference for Git repos
    source_ref: str | None = None

    # Additional pip requirements
    requirements: list[str] = field(default_factory=list)

    # Environment variables for the plugin
    env_vars: dict[str, str] = field(default_factory=dict)

    # Resource requirements
    cpu_limit: str = "1"
    memory_limit: str = "1Gi"
    gpu_count: int = 0

    # Execution timeout in seconds
    timeout_seconds: int = 3600

    # Allowed output destinations
    output_destinations: list[str] = field(default_factory=list)

    # Project and user context
    project_id: str | None = None
    user_id: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate spec after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate plugin specification."""
        # Validate source URI
        if self.source_type == PluginSourceType.GIT_REPO:
            if not self.source_ref:
                raise ValueError("Git source requires source_ref (commit hash)")
            if not (
                self.source_uri.startswith("https://")
                or self.source_uri.startswith("git@")
            ):
                raise ValueError("Git URI must start with https:// or git@")

        elif self.source_type == PluginSourceType.S3_WHEEL:
            if not self.source_uri.startswith("s3://"):
                raise ValueError("S3 source must start with s3://")
            if not self.source_uri.endswith(".whl"):
                raise ValueError("S3 source must be a wheel file (.whl)")

        elif self.source_type == PluginSourceType.LOCAL_PATH:
            # Local paths only allowed in development
            if not self.source_uri.startswith("/"):
                raise ValueError("Local path must be absolute")

        # Validate entrypoint format
        if ":" not in self.entrypoint:
            raise ValueError(
                "Entrypoint must be in module:function format (e.g., my_plugin.main:run)"
            )

    def spec_hash(self) -> str:
        """Generate hash for idempotency checking."""
        content = json.dumps(
            {
                "source_type": self.source_type.value,
                "source_uri": self.source_uri,
                "source_ref": self.source_ref,
                "entrypoint": self.entrypoint,
                "requirements": sorted(self.requirements),
                "env_vars": dict(sorted(self.env_vars.items())),
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_type": self.source_type.value,
            "source_uri": self.source_uri,
            "source_ref": self.source_ref,
            "entrypoint": self.entrypoint,
            "requirements": self.requirements,
            "env_vars": self.env_vars,
            "cpu_limit": self.cpu_limit,
            "memory_limit": self.memory_limit,
            "gpu_count": self.gpu_count,
            "timeout_seconds": self.timeout_seconds,
            "output_destinations": self.output_destinations,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginSpec:
        """Create from dictionary."""
        return cls(
            source_type=PluginSourceType(data["source_type"]),
            source_uri=data["source_uri"],
            source_ref=data.get("source_ref"),
            entrypoint=data["entrypoint"],
            requirements=data.get("requirements", []),
            env_vars=data.get("env_vars", {}),
            cpu_limit=data.get("cpu_limit", "1"),
            memory_limit=data.get("memory_limit", "1Gi"),
            gpu_count=data.get("gpu_count", 0),
            timeout_seconds=data.get("timeout_seconds", 3600),
            output_destinations=data.get("output_destinations", []),
            project_id=data.get("project_id"),
            user_id=data.get("user_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PluginResult:
    """Result of a plugin execution."""

    plugin_id: str
    status: PluginStatus
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    output_files: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error_message: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        """Calculate execution duration."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plugin_id": self.plugin_id,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "output_files": self.output_files,
            "metrics": self.metrics,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class PluginAuditEntry:
    """Audit log entry for plugin execution (X-106)."""

    plugin_id: str
    event_type: str  # submitted, started, output_written, succeeded, failed
    timestamp: datetime
    user_id: str | None = None
    project_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plugin_id": self.plugin_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "project_id": self.project_id,
            "details": self.details,
        }
