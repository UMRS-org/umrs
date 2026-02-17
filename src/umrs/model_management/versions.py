"""
Model Version Management (MM-003).

Handles model versioning and lifecycle:
- Create and track model versions
- Version aliasing (latest, stable, canary)
- Version comparison and diff
- Rollback support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    """Version lifecycle status."""

    DRAFT = "draft"
    STAGING = "staging"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class VersionAlias(Enum):
    """Well-known version aliases."""

    LATEST = "latest"
    STABLE = "stable"
    CANARY = "canary"
    PREVIOUS = "previous"


@dataclass
class ModelVersion:
    """Represents a model version."""

    id: str
    model_id: str
    version: str
    status: VersionStatus
    created_at: datetime
    updated_at: datetime

    # Version metadata
    parent_version: Optional[str] = None
    commit_hash: Optional[str] = None
    description: Optional[str] = None

    # File information
    file_path: str = ""
    file_size_bytes: int = 0
    checksum: Optional[str] = None

    # Runtime info
    quantization: Optional[str] = None
    context_length: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    tokens_per_second: Optional[float] = None
    memory_usage_gb: Optional[float] = None
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)

    # Aliases pointing to this version
    aliases: List[str] = field(default_factory=list)


@dataclass
class VersionDiff:
    """Difference between two versions."""

    from_version: str
    to_version: str
    changes: Dict[str, Any]
    file_changes: List[str]
    parameter_changes: Dict[str, tuple]


class ModelVersionManager:
    """
    Model Version Manager (MM-003).

    Manages model version lifecycle including creation, aliasing,
    comparison, and rollback.
    """

    def __init__(self, db_session: AsyncSession):
        """Initialize version manager."""
        self.db = db_session

    async def create_version(
        self,
        model_id: str,
        version: str,
        file_path: str,
        description: Optional[str] = None,
        parent_version: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """
        Create a new model version.

        Args:
            model_id: Parent model ID
            version: Version string (semver recommended)
            file_path: Path to version files
            description: Version description
            parent_version: Previous version (for lineage)
            parameters: Version parameters

        Returns:
            Created ModelVersion
        """
        from ..models.model_registry import ModelVersionRecord

        # Check for duplicate version
        existing = await self.db.execute(
            select(ModelVersionRecord).where(
                ModelVersionRecord.model_id == model_id,
                ModelVersionRecord.version == version,
            )
        )
        if existing.scalar_one_or_none():
            raise ValueError(f"Version {version} already exists for model {model_id}")

        # Create version record
        record = ModelVersionRecord(
            model_id=model_id,
            version=version,
            status=VersionStatus.DRAFT.value,
            file_path=file_path,
            description=description,
            parent_version=parent_version,
            parameters=parameters or {},
        )

        self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)

        logger.info("Created version %s for model %s", version, model_id)

        return self._to_model_version(record)

    async def get_version(
        self,
        model_id: str,
        version: str,
    ) -> Optional[ModelVersion]:
        """Get a specific version or resolve alias."""
        from ..models.model_registry import ModelVersionRecord

        # Check if version is an alias
        resolved = await self._resolve_alias(model_id, version)

        result = await self.db.execute(
            select(ModelVersionRecord).where(
                ModelVersionRecord.model_id == model_id,
                ModelVersionRecord.version == resolved,
            )
        )
        record = result.scalar_one_or_none()

        if not record:
            return None

        model_version = self._to_model_version(record)

        # Add aliases
        model_version.aliases = await self._get_aliases(model_id, resolved)

        return model_version

    async def list_versions(
        self,
        model_id: str,
        include_archived: bool = False,
    ) -> List[ModelVersion]:
        """List all versions of a model."""
        from ..models.model_registry import ModelVersionRecord

        query = select(ModelVersionRecord).where(
            ModelVersionRecord.model_id == model_id,
        )

        if not include_archived:
            query = query.where(
                ModelVersionRecord.status != VersionStatus.ARCHIVED.value,
            )

        query = query.order_by(ModelVersionRecord.created_at.desc())

        result = await self.db.execute(query)
        records = result.scalars().all()

        versions = []
        for record in records:
            mv = self._to_model_version(record)
            mv.aliases = await self._get_aliases(model_id, mv.version)
            versions.append(mv)

        return versions

    async def update_status(
        self,
        model_id: str,
        version: str,
        status: VersionStatus,
    ) -> Optional[ModelVersion]:
        """Update version status."""
        from ..models.model_registry import ModelVersionRecord

        await self.db.execute(
            update(ModelVersionRecord)
            .where(
                ModelVersionRecord.model_id == model_id,
                ModelVersionRecord.version == version,
            )
            .values(
                status=status.value,
                updated_at=datetime.utcnow(),
            )
        )
        await self.db.commit()

        logger.info(
            "Updated version %s of %s to status %s",
            version,
            model_id,
            status.value,
        )

        return await self.get_version(model_id, version)

    async def promote(
        self,
        model_id: str,
        version: str,
        to_status: VersionStatus = VersionStatus.ACTIVE,
    ) -> Optional[ModelVersion]:
        """
        Promote version to higher status.

        Draft -> Staging -> Active
        """
        valid_promotions = {
            VersionStatus.DRAFT: VersionStatus.STAGING,
            VersionStatus.STAGING: VersionStatus.ACTIVE,
        }

        current = await self.get_version(model_id, version)
        if not current:
            return None

        current_status = current.status
        expected_to = valid_promotions.get(current_status)

        if expected_to is None:
            raise ValueError(f"Cannot promote version in status {current_status.value}")

        # Allow skipping staging
        if to_status == VersionStatus.ACTIVE and current_status == VersionStatus.DRAFT:
            pass
        elif to_status != expected_to:
            raise ValueError(
                f"Expected promotion to {expected_to.value}, got {to_status.value}"
            )

        return await self.update_status(model_id, version, to_status)

    async def set_alias(
        self,
        model_id: str,
        version: str,
        alias: str,
    ) -> None:
        """
        Set alias to point to a version.

        Args:
            model_id: Model ID
            version: Target version
            alias: Alias name (latest, stable, canary, or custom)
        """
        from ..models.model_registry import ModelAliasRecord

        # Verify version exists
        ver = await self.get_version(model_id, version)
        if not ver:
            raise ValueError(f"Version {version} not found for model {model_id}")

        # Update or create alias
        existing = await self.db.execute(
            select(ModelAliasRecord).where(
                ModelAliasRecord.model_id == model_id,
                ModelAliasRecord.alias == alias,
            )
        )
        record = existing.scalar_one_or_none()

        if record:
            # Save previous for rollback
            await self._save_previous(model_id, alias, record.version)
            record.version = version
            record.updated_at = datetime.utcnow()
        else:
            record = ModelAliasRecord(
                model_id=model_id,
                alias=alias,
                version=version,
            )
            self.db.add(record)

        await self.db.commit()

        logger.info(
            "Set alias %s to version %s for model %s",
            alias,
            version,
            model_id,
        )

    async def get_alias(
        self,
        model_id: str,
        alias: str,
    ) -> Optional[str]:
        """Get version pointed to by alias."""
        return await self._resolve_alias(model_id, alias)

    async def rollback(
        self,
        model_id: str,
        alias: str = "stable",
    ) -> Optional[ModelVersion]:
        """
        Rollback alias to previous version.

        Args:
            model_id: Model ID
            alias: Alias to rollback

        Returns:
            Previous version (now current)
        """
        from ..models.model_registry import ModelAliasPreviousRecord

        # Get previous version for this alias
        result = await self.db.execute(
            select(ModelAliasPreviousRecord)
            .where(
                ModelAliasPreviousRecord.model_id == model_id,
                ModelAliasPreviousRecord.alias == alias,
            )
            .order_by(ModelAliasPreviousRecord.created_at.desc())
            .limit(1)
        )
        previous = result.scalar_one_or_none()

        if not previous:
            raise ValueError(f"No previous version to rollback for alias {alias}")

        # Set alias to previous version
        await self.set_alias(model_id, previous.previous_version, alias)

        return await self.get_version(model_id, previous.previous_version)

    async def compare_versions(
        self,
        model_id: str,
        from_version: str,
        to_version: str,
    ) -> VersionDiff:
        """
        Compare two versions.

        Args:
            model_id: Model ID
            from_version: Base version
            to_version: Target version

        Returns:
            VersionDiff with changes
        """
        v1 = await self.get_version(model_id, from_version)
        v2 = await self.get_version(model_id, to_version)

        if not v1 or not v2:
            raise ValueError("One or both versions not found")

        changes = {}
        parameter_changes = {}

        # Compare basic fields
        for field in ["status", "description", "quantization", "context_length"]:
            val1 = getattr(v1, field)
            val2 = getattr(v2, field)
            if val1 != val2:
                changes[field] = {"from": val1, "to": val2}

        # Compare parameters
        all_keys = set(v1.parameters.keys()) | set(v2.parameters.keys())
        for key in all_keys:
            val1 = v1.parameters.get(key)
            val2 = v2.parameters.get(key)
            if val1 != val2:
                parameter_changes[key] = (val1, val2)

        # Compare metrics
        if v1.accuracy_metrics != v2.accuracy_metrics:
            changes["accuracy_metrics"] = {
                "from": v1.accuracy_metrics,
                "to": v2.accuracy_metrics,
            }

        return VersionDiff(
            from_version=from_version,
            to_version=to_version,
            changes=changes,
            file_changes=[],  # Would need file-level comparison
            parameter_changes=parameter_changes,
        )

    async def get_version_lineage(
        self,
        model_id: str,
        version: str,
    ) -> List[ModelVersion]:
        """Get version lineage (parent chain)."""
        lineage = []
        current = version

        while current:
            ver = await self.get_version(model_id, current)
            if not ver:
                break
            lineage.append(ver)
            current = ver.parent_version

        return lineage

    # =========================================================================
    # Private helpers
    # =========================================================================

    async def _resolve_alias(self, model_id: str, version: str) -> str:
        """Resolve alias to actual version."""
        from ..models.model_registry import ModelAliasRecord

        # Check if it's an alias
        result = await self.db.execute(
            select(ModelAliasRecord).where(
                ModelAliasRecord.model_id == model_id,
                ModelAliasRecord.alias == version,
            )
        )
        alias_record = result.scalar_one_or_none()

        if alias_record:
            return alias_record.version

        return version

    async def _get_aliases(self, model_id: str, version: str) -> List[str]:
        """Get all aliases pointing to a version."""
        from ..models.model_registry import ModelAliasRecord

        result = await self.db.execute(
            select(ModelAliasRecord).where(
                ModelAliasRecord.model_id == model_id,
                ModelAliasRecord.version == version,
            )
        )
        records = result.scalars().all()

        return [r.alias for r in records]

    async def _save_previous(
        self,
        model_id: str,
        alias: str,
        version: str,
    ) -> None:
        """Save previous version for rollback."""
        from ..models.model_registry import ModelAliasPreviousRecord

        record = ModelAliasPreviousRecord(
            model_id=model_id,
            alias=alias,
            previous_version=version,
        )
        self.db.add(record)

    def _to_model_version(self, record: Any) -> ModelVersion:
        """Convert DB record to ModelVersion."""
        return ModelVersion(
            id=str(record.id),
            model_id=str(record.model_id),
            version=record.version,
            status=VersionStatus(record.status),
            created_at=record.created_at,
            updated_at=record.updated_at,
            parent_version=record.parent_version,
            commit_hash=getattr(record, "commit_hash", None),
            description=record.description,
            file_path=record.file_path,
            file_size_bytes=getattr(record, "file_size_bytes", 0),
            checksum=getattr(record, "checksum", None),
            quantization=getattr(record, "quantization", None),
            context_length=getattr(record, "context_length", None),
            parameters=record.parameters or {},
            tokens_per_second=getattr(record, "tokens_per_second", None),
            memory_usage_gb=getattr(record, "memory_usage_gb", None),
            accuracy_metrics=getattr(record, "accuracy_metrics", {}),
        )
