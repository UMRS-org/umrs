"""Evidence — evidence items with source, content, relevance score, and citation.

Phase 1, Step 2.5 of the RC1 migration plan.

UMRS owns evidence tracking. R4 consumes this API.

This module provides an evidence store for tracking evidence items that
support or refute scientific claims:

  - Create evidence items linked to scientific objects
  - Track evidence source type (uploaded file, notebook cell, source code,
    experiment result, external paper, web search, dataset, observation,
    conclusion, reference)
  - Track evidence content and relevance score
  - Provide evidence citation strings
  - Support evidence-based reasoning (every claim cites evidence)
  - Link evidence to claims (hypotheses, conclusions, findings)

The store is append-only — evidence items are never deleted, only retracted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

__all__ = [
    "EvidenceSourceType",
    "EvidenceStatus",
    "Evidence",
    "EvidenceNotFoundError",
    "DuplicateEvidenceError",
    "InMemoryEvidenceStore",
]


# ── Evidence Source Types ─────────────────────────────────────────────────


class EvidenceSourceType(str, Enum):
    """Types of evidence sources."""

    UPLOADED_FILE = "uploaded_file"
    NOTEBOOK_CELL = "notebook_cell"
    SOURCE_CODE = "source_code"
    EXPERIMENT_RESULT = "experiment_result"
    EXTERNAL_PAPER = "external_paper"
    WEB_SEARCH = "web_search"
    DATASET = "dataset"
    OBSERVATION = "observation"
    CONCLUSION = "conclusion"
    REFERENCE = "reference"
    CHAT_RESPONSE = "chat_response"
    MANUAL = "manual"


class EvidenceStatus(str, Enum):
    """Status of an evidence item."""

    ACTIVE = "active"
    RETRACTED = "retracted"
    SUPERSEDED = "superseded"


# ── Evidence ─────────────────────────────────────────────────────────────


@dataclass
class Evidence:
    """An evidence item linked to a scientific claim.

    Attributes:
        evidence_id: Unique identifier for the evidence
        project_id: The project this evidence belongs to
        claim_id: The claim (hypothesis, conclusion, finding) this evidence supports
        source_type: Type of evidence source
        source_id: ID of the source object (e.g., paper_id, experiment_id)
        content: The evidence content (text, data, etc.)
        content_uri: S3 URI if content is large
        relevance_score: How relevant this evidence is to the claim (0.0–1.0)
        citation: Citation string for this evidence
        status: Current status of the evidence
        metadata: Additional metadata
        created_at: When the evidence was created
        updated_at: When the evidence was last updated
        created_by: Who created the evidence
    """

    evidence_id: str = ""
    project_id: str = ""
    claim_id: str = ""
    source_type: EvidenceSourceType = EvidenceSourceType.MANUAL
    source_id: str = ""
    content: str = ""
    content_uri: str = ""
    relevance_score: float = 0.0
    citation: str = ""
    status: EvidenceStatus = EvidenceStatus.ACTIVE
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id:
            self.evidence_id = f"evd-{uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "evidence_id": self.evidence_id,
            "project_id": self.project_id,
            "claim_id": self.claim_id,
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "content": self.content,
            "content_uri": self.content_uri,
            "relevance_score": self.relevance_score,
            "citation": self.citation,
            "status": self.status.value,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        """Create from dictionary."""
        source_type_raw = data.get("source_type", "manual")
        status_raw = data.get("status", "active")
        source_type = (
            EvidenceSourceType(source_type_raw)
            if isinstance(source_type_raw, str)
            else source_type_raw
        )
        status = (
            EvidenceStatus(status_raw)
            if isinstance(status_raw, str)
            else status_raw
        )
        return cls(
            evidence_id=data.get("evidence_id", ""),
            project_id=data.get("project_id", ""),
            claim_id=data.get("claim_id", ""),
            source_type=source_type,
            source_id=data.get("source_id", ""),
            content=data.get("content", ""),
            content_uri=data.get("content_uri", ""),
            relevance_score=data.get("relevance_score", 0.0),
            citation=data.get("citation", ""),
            status=status,
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            created_by=data.get("created_by", ""),
        )


# ── Exceptions ────────────────────────────────────────────────────────────


class EvidenceNotFoundError(Exception):
    """Raised when an evidence item is not found."""


class DuplicateEvidenceError(Exception):
    """Raised when a duplicate evidence item is added."""


# ── In-Memory Evidence Store ──────────────────────────────────────────────


class InMemoryEvidenceStore:
    """In-memory store for evidence items.

    This is the reference implementation. The Postgres-backed implementation
    will follow the same interface.
    """

    def __init__(self) -> None:
        self._evidence: dict[str, Evidence] = {}
        # Index: claim_id -> [evidence_id, ...]
        self._by_claim: dict[str, list[str]] = {}

    def add_evidence(self, evidence: Evidence) -> str:
        """Add a new evidence item.

        Args:
            evidence: The evidence to add

        Returns:
            The evidence_id of the added evidence

        Raises:
            DuplicateEvidenceError: If evidence_id already exists
        """
        if evidence.evidence_id in self._evidence:
            raise DuplicateEvidenceError(
                f"Evidence {evidence.evidence_id} already exists"
            )
        self._evidence[evidence.evidence_id] = evidence
        # Index by claim_id
        if evidence.claim_id:
            self._by_claim.setdefault(evidence.claim_id, []).append(
                evidence.evidence_id
            )
        return evidence.evidence_id

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        """Get an evidence item by ID.

        Args:
            evidence_id: The evidence ID

        Returns:
            The evidence, or None if not found
        """
        return self._evidence.get(evidence_id)

    def list_evidence(
        self,
        *,
        project_id: str | None = None,
        source_type: EvidenceSourceType | None = None,
        status: EvidenceStatus | None = None,
        claim_id: str | None = None,
        limit: int = 100,
    ) -> list[Evidence]:
        """List evidence items with optional filters.

        Args:
            project_id: Filter by project ID
            source_type: Filter by source type
            status: Filter by status (default: ACTIVE only)
            claim_id: Filter by claim ID
            limit: Maximum number of items to return

        Returns:
            List of matching evidence items, sorted by relevance_score descending
        """
        if status is None:
            status = EvidenceStatus.ACTIVE
        results: list[Evidence] = []
        for evd in self._evidence.values():
            if project_id and evd.project_id != project_id:
                continue
            if source_type and evd.source_type != source_type:
                continue
            if status is not None and evd.status != status:
                continue
            if claim_id and evd.claim_id != claim_id:
                continue
            results.append(evd)
        # Sort by relevance_score descending (most relevant first)
        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results[:limit]

    def search_evidence(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[Evidence]:
        """Search evidence items by content or citation.

        Args:
            query: The search query
            project_id: Filter by project ID
            limit: Maximum number of results

        Returns:
            List of matching evidence items
        """
        query_lower = query.lower()
        results: list[Evidence] = []
        for evd in self._evidence.values():
            if evd.status != EvidenceStatus.ACTIVE:
                continue
            if project_id and evd.project_id != project_id:
                continue
            # Search in content and citation
            if (
                query_lower in evd.content.lower()
                or query_lower in evd.citation.lower()
            ):
                results.append(evd)
        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results[:limit]

    def get_evidence_for_claim(self, claim_id: str) -> list[Evidence]:
        """Get all evidence items for a specific claim.

        Args:
            claim_id: The claim ID (hypothesis, conclusion, finding)

        Returns:
            List of evidence items for the claim, sorted by relevance_score descending
        """
        evidence_ids = self._by_claim.get(claim_id, [])
        results = [
            self._evidence[eid]
            for eid in evidence_ids
            if eid in self._evidence
            and self._evidence[eid].status == EvidenceStatus.ACTIVE
        ]
        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results

    def update_status(
        self,
        evidence_id: str,
        status: EvidenceStatus,
    ) -> bool:
        """Update the status of an evidence item.

        Args:
            evidence_id: The evidence ID
            status: The new status

        Returns:
            True if the evidence was found and updated, False otherwise
        """
        evd = self._evidence.get(evidence_id)
        if not evd:
            return False
        evd.status = status
        evd.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    def count(
        self,
        *,
        project_id: str | None = None,
        source_type: EvidenceSourceType | None = None,
        status: EvidenceStatus | None = None,
    ) -> int:
        """Count evidence items with optional filters.

        Args:
            project_id: Filter by project ID
            source_type: Filter by source type
            status: Filter by status

        Returns:
            Number of matching evidence items
        """
        count = 0
        for evd in self._evidence.values():
            if project_id and evd.project_id != project_id:
                continue
            if source_type and evd.source_type != source_type:
                continue
            if status and evd.status != status:
                continue
            count += 1
        return count
