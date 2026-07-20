"""Artifact DTOs — stdlib-only dataclasses for the artifact registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass
class Artifact:
    """A registered artifact in the registry."""

    artifact_type: str  # dataset|model|embedding|result|image|checkpoint|config|log
    name: str
    owner_id: str
    uri: str  # s3://bucket/key or file path
    project_id: str | None = None
    bucket: str | None = None
    s3_key: str | None = None
    size_bytes: int | None = None
    checksum: str | None = None
    metadata: dict = field(default_factory=dict)
    execution_id: str | None = None
    workflow_id: str | None = None
    job_id: str | None = None
    version: int = 1
    id: UUID | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row: tuple | list) -> "Artifact":
        """Construct an Artifact from a DB row.

        Expected column order matches the artifacts table SELECT.
        """
        return cls(
            id=row[0],
            artifact_type=row[1],
            name=row[2],
            owner_id=row[3],
            project_id=row[4],
            uri=row[5],
            bucket=row[6],
            s3_key=row[7],
            size_bytes=row[8],
            checksum=row[9],
            metadata=row[10] if row[10] else {},
            execution_id=row[11],
            workflow_id=row[12],
            job_id=row[13],
            version=row[14],
            created_at=row[15],
            updated_at=row[16],
        )

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "id": str(self.id) if self.id else None,
            "artifact_type": self.artifact_type,
            "name": self.name,
            "owner_id": self.owner_id,
            "project_id": self.project_id,
            "uri": self.uri,
            "bucket": self.bucket,
            "s3_key": self.s3_key,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "metadata": self.metadata,
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "job_id": self.job_id,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class ArtifactLineageEdge:
    """A directed edge between two artifacts."""

    artifact_id: UUID
    parent_artifact_id: UUID
    relation: str = "derived_from"


@dataclass
class ArtifactQuery:
    """Filters for listing artifacts."""

    artifact_type: str | None = None
    owner_id: str | None = None
    project_id: str | None = None
    execution_id: str | None = None
    name_prefix: str | None = None
    limit: int = 100
    offset: int = 0
