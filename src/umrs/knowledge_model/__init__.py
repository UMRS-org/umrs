"""Knowledge model — first-class scientific object lifecycle management.

Phase 1, Step 2.2 of the RC1 migration plan.

UMRS owns the knowledge model. R4 consumes this API.

Scientific object types (M3):
  Input artifacts:
    - PAPER        — Input paper (PDF, arXiv, etc.)
    - NOTEBOOK     — Jupyter notebook
    - CODE         — Source code (Python, C++, CUDA)
    - DATASET      — Input dataset (CSV, JSON, etc.)

  Research outputs:
    - HYPOTHESIS   — Testable hypothesis
    - OBSERVATION  — Experimental observation
    - CONCLUSION   — Drawn conclusion
    - FINDING      — Research finding
    - EXPERIMENT   — Experiment definition + results

  Derived artifacts:
    - FIGURE       — Generated figure/plot
    - EQUATION     — Mathematical equation
    - ALGORITHM    — Algorithm description

  Supporting objects:
    - EVIDENCE     — Evidence item
    - REFERENCE    — Reference to external work
    - ASSUMPTION   — Working assumption
    - QUESTION     — Open scientific question

  NOT KnowledgeObjects (R4 orchestration state):
    - ResearchPlan  — Mutable, versioned, regenerated frequently
    - ResearchLoop  — Orchestration artifact
    - VerificationAnswer — Ephemeral LLM output, can be regenerated
    - Workflow      — Orchestration artifact

Objects are versioned. Mutable objects can be updated, creating a new version.
Immutable objects (e.g., content-addressed artifacts) cannot be updated.

Soft deletes are supported via tombstones. Objects are never hard-deleted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

__all__ = [
    "KnowledgeObjectType",
    "KnowledgeObjectStatus",
    "KnowledgeObject",
    "ObjectVersion",
    "Tombstone",
    "KnowledgeObjectNotFoundError",
    "DuplicateKnowledgeObjectError",
    "InMemoryKnowledgeModelStore",
    "build_default_store",
]


# ── Object Types ──────────────────────────────────────────────────────────


class KnowledgeObjectType(str, Enum):
    """Types of scientific knowledge objects.

    M3 Decision: ResearchPlan, ResearchLoop, VerificationAnswer, and
    Workflow are NOT KnowledgeObjects. They are R4 orchestration state.

    Scientific knowledge objects represent facts about science:
      - Papers, datasets, code, notebooks are input artifacts
      - Hypotheses, observations, conclusions, findings are research outputs
      - Equations, figures are derived artifacts
      - Evidence, references, assumptions are supporting objects
    """

    # Input artifacts
    PAPER = "paper"
    NOTEBOOK = "notebook"
    CODE = "code"
    DATASET = "dataset"

    # Research outputs
    HYPOTHESIS = "hypothesis"
    OBSERVATION = "observation"
    CONCLUSION = "conclusion"
    FINDING = "finding"
    EXPERIMENT = "experiment"

    # Derived artifacts
    FIGURE = "figure"
    EQUATION = "equation"
    ALGORITHM = "algorithm"

    # Supporting objects
    EVIDENCE = "evidence"
    REFERENCE = "reference"
    ASSUMPTION = "assumption"
    QUESTION = "question"


class KnowledgeObjectStatus(str, Enum):
    """Status of a knowledge object."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    DELETED = "deleted"


# ── Knowledge Object ──────────────────────────────────────────────────────


@dataclass
class KnowledgeObject:
    """A first-class scientific knowledge object.

    Attributes:
        object_id: Unique identifier for the object
        project_id: The project this object belongs to
        object_type: The type of object (paper, code, hypothesis, etc.)
        content: The object's content (JSON-serializable)
        metadata: Additional metadata
        version: Current version number (1, 2, 3, ...)
        parent_version_id: ID of the previous version (if this is a new version)
        status: Current status of the object
        is_immutable: Whether this object can be updated
        created_at: When the object was created
        updated_at: When the object was last updated
        created_by: User or agent ID that created the object
    """

    object_id: str = ""
    project_id: str = ""
    object_type: KnowledgeObjectType = KnowledgeObjectType.PAPER
    content: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    parent_version_id: str | None = None
    status: KnowledgeObjectStatus = KnowledgeObjectStatus.ACTIVE
    is_immutable: bool = False
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""

    def __post_init__(self) -> None:
        if not self.object_id:
            self.object_id = f"obj-{uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "object_id": self.object_id,
            "project_id": self.project_id,
            "object_type": self.object_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "version": self.version,
            "parent_version_id": self.parent_version_id,
            "status": self.status.value,
            "is_immutable": self.is_immutable,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeObject":
        """Create from dictionary."""
        object_type_raw = data.get("object_type", "paper")
        status_raw = data.get("status", "active")
        object_type = (
            KnowledgeObjectType(object_type_raw)
            if isinstance(object_type_raw, str)
            else object_type_raw
        )
        status = (
            KnowledgeObjectStatus(status_raw)
            if isinstance(status_raw, str)
            else status_raw
        )
        return cls(
            object_id=data.get("object_id", ""),
            project_id=data.get("project_id", ""),
            object_type=object_type,
            content=data.get("content", {}),
            metadata=data.get("metadata", {}),
            version=data.get("version", 1),
            parent_version_id=data.get("parent_version_id"),
            status=status,
            is_immutable=data.get("is_immutable", False),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            created_by=data.get("created_by", ""),
        )


# ── Object Version ───────────────────────────────────────────────────────


@dataclass
class ObjectVersion:
    """A single version of a knowledge object.

    Attributes:
        version_id: Unique identifier for this version
        object_id: The object this version belongs to
        version: Version number
        content: Content at this version
        metadata: Metadata at this version
        created_at: When this version was created
        created_by: Who created this version
    """

    version_id: str = ""
    object_id: str = ""
    version: int = 1
    content: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    created_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "object_id": self.object_id,
            "version": self.version,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


# ── Tombstone ────────────────────────────────────────────────────────────


@dataclass
class Tombstone:
    """Records the deletion of an object.

    Attributes:
        object_id: The object that was deleted
        object_type: The type of object
        deleted_at: When the object was deleted
        deleted_by: Who deleted the object
        deletion_reason: Why the object was deleted
    """

    object_id: str = ""
    object_type: KnowledgeObjectType = KnowledgeObjectType.PAPER
    deleted_at: str = ""
    deleted_by: str = ""
    deletion_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "object_id": self.object_id,
            "object_type": self.object_type.value,
            "deleted_at": self.deleted_at,
            "deleted_by": self.deleted_by,
            "deletion_reason": self.deletion_reason,
        }


# ── Exceptions ────────────────────────────────────────────────────────────


class KnowledgeObjectNotFoundError(Exception):
    """Raised when a knowledge object is not found."""


class DuplicateKnowledgeObjectError(Exception):
    """Raised when a duplicate knowledge object is added."""


class ImmutableObjectError(Exception):
    """Raised when attempting to update an immutable object."""


# ── In-Memory Store ───────────────────────────────────────────────────────


class InMemoryKnowledgeModelStore:
    """In-memory store for knowledge objects.

    This is the reference implementation. The Postgres-backed implementation
    will follow the same interface.
    """

    def __init__(self) -> None:
        self._objects: dict[str, KnowledgeObject] = {}
        self._versions: dict[str, list[ObjectVersion]] = {}
        self._tombstones: dict[str, Tombstone] = {}

    def create_object(
        self,
        *,
        project_id: str,
        object_type: KnowledgeObjectType,
        content: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        is_immutable: bool = False,
        created_by: str = "",
    ) -> str:
        """Create a new knowledge object.

        Args:
            project_id: The project this object belongs to
            object_type: The type of object
            content: The object's content
            metadata: Additional metadata
            is_immutable: Whether this object can be updated
            created_by: Who created the object

        Returns:
            The object_id of the created object

        Raises:
            DuplicateKnowledgeObjectError: If object_id already exists
        """
        obj = KnowledgeObject(
            project_id=project_id,
            object_type=object_type,
            content=content or {},
            metadata=metadata or {},
            is_immutable=is_immutable,
            created_by=created_by,
        )
        if obj.object_id in self._objects:
            raise DuplicateKnowledgeObjectError(
                f"Object {obj.object_id} already exists"
            )
        self._objects[obj.object_id] = obj
        # Record initial version
        self._versions[obj.object_id] = [
            ObjectVersion(
                version_id=f"ver-{uuid4().hex[:12]}",
                object_id=obj.object_id,
                version=1,
                content=obj.content,
                metadata=obj.metadata,
                created_at=obj.created_at,
                created_by=created_by,
            )
        ]
        return obj.object_id

    def get_object(self, object_id: str) -> KnowledgeObject | None:
        """Get an object by ID.

        Args:
            object_id: The object ID

        Returns:
            The object, or None if not found
        """
        return self._objects.get(object_id)

    def update_object(
        self,
        object_id: str,
        *,
        content: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update an object's content and/or metadata.

        This creates a new version of the object.

        Args:
            object_id: The object ID
            content: New content (optional)
            metadata: New metadata (optional)

        Returns:
            True if the object was found and updated, False otherwise

        Raises:
            ImmutableObjectError: If the object is immutable
        """
        obj = self._objects.get(object_id)
        if not obj:
            return False
        if obj.is_immutable:
            raise ImmutableObjectError(
                f"Object {object_id} is immutable and cannot be updated"
            )
        # Update content and metadata
        if content is not None:
            obj.content = content
        if metadata is not None:
            obj.metadata = metadata
        # Create new version
        obj.version += 1
        obj.updated_at = datetime.now(timezone.utc).isoformat()
        self._versions[object_id].append(
            ObjectVersion(
                version_id=f"ver-{uuid4().hex[:12]}",
                object_id=object_id,
                version=obj.version,
                content=obj.content,
                metadata=obj.metadata,
                created_at=obj.updated_at,
            )
        )
        return True

    def delete_object(
        self,
        object_id: str,
        *,
        deleted_by: str = "",
        deletion_reason: str = "",
    ) -> bool:
        """Delete an object (soft delete via tombstone).

        Args:
            object_id: The object ID
            deleted_by: Who deleted the object
            deletion_reason: Why the object was deleted

        Returns:
            True if the object was found and deleted, False otherwise
        """
        obj = self._objects.get(object_id)
        if not obj:
            return False
        # Create tombstone
        self._tombstones[object_id] = Tombstone(
            object_id=object_id,
            object_type=obj.object_type,
            deleted_at=datetime.now(timezone.utc).isoformat(),
            deleted_by=deleted_by,
            deletion_reason=deletion_reason,
        )
        # Mark as deleted
        obj.status = KnowledgeObjectStatus.DELETED
        obj.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    def list_objects(
        self,
        *,
        project_id: str | None = None,
        object_type: KnowledgeObjectType | None = None,
        status: KnowledgeObjectStatus | None = None,
        limit: int = 100,
    ) -> list[KnowledgeObject]:
        """List objects with optional filters.

        Args:
            project_id: Filter by project ID
            object_type: Filter by object type
            status: Filter by status (default: ACTIVE only)
            limit: Maximum number of objects to return

        Returns:
            List of matching objects
        """
        if status is None:
            status = KnowledgeObjectStatus.ACTIVE
        results: list[KnowledgeObject] = []
        for obj in self._objects.values():
            if project_id and obj.project_id != project_id:
                continue
            if object_type and obj.object_type != object_type:
                continue
            # Filter by status (show only active by default)
            if status is not None and obj.status != status:
                continue
            results.append(obj)
        results.sort(key=lambda o: o.created_at, reverse=True)
        return results[:limit]

    def search_objects(
        self,
        query: str,
        *,
        project_id: str | None = None,
        object_type: KnowledgeObjectType | None = None,
        limit: int = 20,
    ) -> list[KnowledgeObject]:
        """Search objects by content, metadata, or object_id.

        Args:
            query: The search query
            project_id: Filter by project ID
            object_type: Filter by object type
            limit: Maximum number of results

        Returns:
            List of matching objects
        """
        query_lower = query.lower()
        results: list[KnowledgeObject] = []
        for obj in self._objects.values():
            if obj.status == KnowledgeObjectStatus.DELETED:
                continue
            if project_id and obj.project_id != project_id:
                continue
            if object_type and obj.object_type != object_type:
                continue
            # Search in content, metadata, and object_id
            content_str = str(obj.content).lower()
            metadata_str = str(obj.metadata).lower()
            if (
                query_lower in content_str
                or query_lower in metadata_str
                or query_lower in obj.object_id.lower()
            ):
                results.append(obj)
        results.sort(key=lambda o: o.created_at, reverse=True)
        return results[:limit]

    def get_object_history(self, object_id: str) -> list[ObjectVersion]:
        """Get the version history of an object.

        Args:
            object_id: The object ID

        Returns:
            List of versions, sorted by version number ascending

        Raises:
            KnowledgeObjectNotFoundError: If the object is not found
        """
        if object_id not in self._objects:
            raise KnowledgeObjectNotFoundError(
                f"Object {object_id} not found"
            )
        versions = self._versions.get(object_id, [])
        return sorted(versions, key=lambda v: v.version)

    def get_tombstone(self, object_id: str) -> Tombstone | None:
        """Get the tombstone for a deleted object.

        Args:
            object_id: The object ID

        Returns:
            The tombstone, or None if not found
        """
        return self._tombstones.get(object_id)


# ── Default Store Factory ───────────────────────────────────────────────


def build_default_store(db_pool: Any = None) -> Any:
    """Build the default knowledge model store backend.

    Selection via ``UMRS_KNOWLEDGE_MODEL_STORE_BACKEND`` (default ``auto``):

      - ``pg``     — force Postgres (raises if unavailable)
      - ``memory`` — force the in-memory reference store
      - ``auto``   — Postgres when a pool is provided or the environment
                     points at a database, otherwise in-memory

    Postgres uses ``sat_knowledge_objects`` / ``sat_knowledge_object_versions``
    / ``sat_knowledge_tombstones`` (migration 0060).
    """
    import os as _os

    from umrs.shared.pg import resolve_satellite_pool

    backend = _os.environ.get("UMRS_KNOWLEDGE_MODEL_STORE_BACKEND", "auto").lower()
    if backend == "memory":
        return InMemoryKnowledgeModelStore()
    if backend not in ("auto", "pg"):
        logger.warning("Unknown store backend %r; using auto", backend)
        backend = "auto"

    pool = resolve_satellite_pool(backend, db_pool, "Knowledge model store")
    if pool is None:
        return InMemoryKnowledgeModelStore()

    from .pg_store import PgKnowledgeModelStore

    return PgKnowledgeModelStore(pool)
