"""
Model Version Management (MM-003).

NOTE: This module is production-ready but not yet surfaced in demos.
      See Demo 08 (model_registry) for model registration & staging usage.
      Version tracking will be integrated when the model registry CLI
      commands (umrs model register/list/download) are wired in.

Provides version tracking for models, allowing:
- Multiple versions per model
- Version comparison and rollback
- Release notes and changelogs
- Deprecation workflow

See DEVELOPMENT_OBJECTIVES.md Section 8 for design details.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class VersionStatus(Enum):
    """Status of a model version."""

    DRAFT = "draft"  # Being prepared, not yet released
    RELEASED = "released"  # Available for use
    DEPRECATED = "deprecated"  # Marked for removal
    ARCHIVED = "archived"  # No longer available


@dataclass
class ModelVersion:
    """
    Represents a specific version of a model (MM-003).

    Tracks:
    - Version number (semver-like)
    - Storage reference for this version
    - Release notes and changelog
    - Parent version for lineage
    """

    version_id: str
    model_id: str
    version: str  # e.g., "1.0.0", "1.1.0-beta"

    # Storage
    storage_ref: str
    storage_size_bytes: int | None = None

    # Metadata
    description: str | None = None
    changelog: str | None = None
    release_notes: str | None = None

    # Lineage
    parent_version_id: str | None = None
    created_from: str | None = None  # "training", "conversion", "import"

    # Status
    status: VersionStatus = VersionStatus.DRAFT
    is_latest: bool = False
    is_stable: bool = False
    source_type: str = "manual"

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    released_at: datetime | None = None
    deprecated_at: datetime | None = None

    # Metrics (optional)
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def generate_version_id(cls) -> str:
        """Generate a unique version ID."""
        return f"ver-{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "version_id": self.version_id,
            "model_id": self.model_id,
            "version": self.version,
            "storage_ref": self.storage_ref,
            "storage_size_bytes": self.storage_size_bytes,
            "description": self.description,
            "changelog": self.changelog,
            "release_notes": self.release_notes,
            "parent_version_id": self.parent_version_id,
            "created_from": self.created_from,
            "status": self.status.value,
            "is_latest": self.is_latest,
            "is_stable": self.is_stable,
            "source_type": self.source_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "released_at": self.released_at.isoformat() if self.released_at else None,
            "deprecated_at": self.deprecated_at.isoformat()
            if self.deprecated_at
            else None,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelVersion:
        """Create from dictionary."""
        return cls(
            version_id=data["version_id"],
            model_id=data["model_id"],
            version=data["version"],
            storage_ref=data["storage_ref"],
            storage_size_bytes=data.get("storage_size_bytes"),
            description=data.get("description"),
            changelog=data.get("changelog"),
            release_notes=data.get("release_notes"),
            parent_version_id=data.get("parent_version_id"),
            created_from=data.get("created_from"),
            status=VersionStatus(data.get("status", "draft")),
            is_latest=data.get("is_latest", False),
            is_stable=data.get("is_stable", False),
            source_type=data.get("source_type", "manual"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now()
            ),
            released_at=(
                datetime.fromisoformat(data["released_at"])
                if data.get("released_at")
                else None
            ),
            deprecated_at=(
                datetime.fromisoformat(data["deprecated_at"])
                if data.get("deprecated_at")
                else None
            ),
            metrics=data.get("metrics", {}),
        )


class VersionComparator:
    """Compare semantic versions."""

    @staticmethod
    def parse_version(version: str) -> tuple:
        """Parse version string into comparable tuple."""
        # Handle pre-release suffixes
        base = version.split("-")[0]
        prerelease = version.split("-")[1] if "-" in version else None

        parts = base.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0

        # Pre-release versions come before release versions
        prerelease_order = 0 if prerelease is None else -1

        return (major, minor, patch, prerelease_order, prerelease or "")

    @classmethod
    def compare(cls, v1: str, v2: str) -> int:
        """
        Compare two versions.

        Returns:
            -1 if v1 < v2
             0 if v1 == v2
             1 if v1 > v2
        """
        t1 = cls.parse_version(v1)
        t2 = cls.parse_version(v2)

        if t1 < t2:
            return -1
        elif t1 > t2:
            return 1
        return 0

    @classmethod
    def is_newer(cls, v1: str, v2: str) -> bool:
        """Check if v1 is newer than v2."""
        return cls.compare(v1, v2) > 0


@dataclass
class VersionHistory:
    """
    Version history for a model.

    Provides:
    - List of all versions
    - Current/latest version
    - Version lineage
    """

    model_id: str
    versions: list[ModelVersion] = field(default_factory=list)

    @property
    def latest(self) -> ModelVersion | None:
        """Get the latest released version."""
        released = [v for v in self.versions if v.status == VersionStatus.RELEASED]
        if not released:
            return None

        active = next((version for version in released if version.is_latest), None)
        if active:
            return active

        # Sort by version number
        released.sort(
            key=lambda v: VersionComparator.parse_version(v.version),
            reverse=True,
        )
        return released[0]

    @property
    def current(self) -> ModelVersion | None:
        """Alias for latest."""
        return self.latest

    def get_version(self, version_str: str) -> ModelVersion | None:
        """Get a specific version."""
        for v in self.versions:
            if v.version == version_str:
                return v
        return None

    def get_lineage(self, version_id: str) -> list[ModelVersion]:
        """Get version lineage (parent chain)."""
        lineage = []
        current_id = version_id

        version_map = {v.version_id: v for v in self.versions}

        while current_id:
            if current_id not in version_map:
                break
            version = version_map[current_id]
            lineage.append(version)
            current_id = version.parent_version_id

        return lineage

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "versions": [v.to_dict() for v in self.versions],
            "latest_version": self.latest.version if self.latest else None,
        }
