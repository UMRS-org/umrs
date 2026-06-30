"""Scientific memory — automatic registration of all generated artifacts.

Phase 9 of the Scientific Knowledge Build-up roadmap.

Everything generated should become searchable:
  - Generated code
  - Experiment logs
  - Plots/figures
  - Analysis results
  - New hypotheses
  - Review comments
  - Accepted findings
  - Failed experiments

Store history. Knowledge grows continuously.

Acceptance: Knowledge grows continuously. Nothing is lost.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Memory Entry Types ─────────────────────────────────────────────────────


class MemoryEntryType(str, Enum):
    """Types of entries in scientific memory."""

    GENERATED_CODE = "generated_code"
    EXPERIMENT_LOG = "experiment_log"
    PLOT_FIGURE = "plot_figure"
    ANALYSIS_RESULT = "analysis_result"
    NEW_HYPOTHESIS = "new_hypothesis"
    REVIEW_COMMENT = "review_comment"
    ACCEPTED_FINDING = "accepted_finding"
    FAILED_EXPERIMENT = "failed_experiment"
    OBSERVATION = "observation"
    CONCLUSION = "conclusion"
    QUESTION = "question"
    ASSUMPTION = "assumption"
    REFERENCE = "reference"
    NOTE = "note"


class MemoryEntryStatus(str, Enum):
    """Status of a memory entry."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    RETRACTED = "retracted"


# ── Memory Entry Model ─────────────────────────────────────────────────────


@dataclass
class MemoryEntry:
    """A single entry in scientific memory.

    Every entry has a type, source, content, and metadata. Entries are
    immutable once created (except for status changes).
    """

    entry_id: str                          # Unique identifier
    entry_type: MemoryEntryType            # Type of entry
    project_id: str = ""                   # Project this belongs to
    experiment_id: str = ""                # Experiment this relates to
    source: str = "system"                 # user | agent | system
    content: str = ""                      # The actual content (text, code, etc.)
    content_uri: str = ""                  # S3 URI if content is large
    summary: str = ""                      # Short summary
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    status: MemoryEntryStatus = MemoryEntryStatus.ACTIVE
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""                   # user_id or agent_id
    confidence: float = 0.0                # 0.0–1.0
    references: list[str] = field(default_factory=list)  # Other entry_ids

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "entry_type": self.entry_type.value,
            "project_id": self.project_id,
            "experiment_id": self.experiment_id,
            "source": self.source,
            "content": self.content,
            "content_uri": self.content_uri,
            "summary": self.summary,
            "metadata": self.metadata,
            "tags": self.tags,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "confidence": self.confidence,
            "references": self.references,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            entry_id=data["entry_id"],
            entry_type=MemoryEntryType(data.get("entry_type", "note")),
            project_id=data.get("project_id", ""),
            experiment_id=data.get("experiment_id", ""),
            source=data.get("source", "system"),
            content=data.get("content", ""),
            content_uri=data.get("content_uri", ""),
            summary=data.get("summary", ""),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            status=MemoryEntryStatus(data.get("status", "active")),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            created_by=data.get("created_by", ""),
            confidence=data.get("confidence", 0.0),
            references=data.get("references", []),
        )


# ── Scientific Memory Store ────────────────────────────────────────────────


class ScientificMemoryStore:
    """Persistent store for scientific memory entries.

    Uses a JSON file for persistence. Each project has its own memory store.
    The store is append-only — entries are never deleted, only archived.
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._storage_path = Path(storage_path) if storage_path else None
        self._entries: dict[str, MemoryEntry] = {}
        self._lock = None  # Not needed for single-process; placeholder for future

    def add(self, entry: MemoryEntry) -> str:
        """Add a new entry to the store.

        Returns the entry_id.
        """
        if entry.entry_id in self._entries:
            raise ValueError(f"Entry {entry.entry_id} already exists")
        self._entries[entry.entry_id] = entry
        self._save()
        return entry.entry_id

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Get an entry by ID."""
        return self._entries.get(entry_id)

    def list_entries(
        self,
        *,
        entry_type: MemoryEntryType | None = None,
        project_id: str | None = None,
        experiment_id: str | None = None,
        status: MemoryEntryStatus | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List entries with optional filters."""
        results: list[MemoryEntry] = []
        for entry in self._entries.values():
            if entry_type and entry.entry_type != entry_type:
                continue
            if project_id and entry.project_id != project_id:
                continue
            if experiment_id and entry.experiment_id != experiment_id:
                continue
            if status and entry.status != status:
                continue
            results.append(entry)
            if len(results) >= limit:
                break
        return results

    def search(
        self,
        query: str,
        *,
        entry_type: MemoryEntryType | None = None,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Search entries by content, summary, or tags."""
        query_lower = query.lower()
        results: list[MemoryEntry] = []
        for entry in self._entries.values():
            if entry_type and entry.entry_type != entry_type:
                continue
            if project_id and entry.project_id != project_id:
                continue
            # Search in content, summary, and tags
            if (
                query_lower in entry.content.lower()
                or query_lower in entry.summary.lower()
                or any(query_lower in tag.lower() for tag in entry.tags)
            ):
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def update_status(
        self,
        entry_id: str,
        status: MemoryEntryStatus,
    ) -> bool:
        """Update the status of an entry (e.g., archive, retract)."""
        entry = self._entries.get(entry_id)
        if not entry:
            return False
        entry.status = status
        entry.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return True

    def count(
        self,
        *,
        entry_type: MemoryEntryType | None = None,
        project_id: str | None = None,
        status: MemoryEntryStatus | None = None,
    ) -> int:
        """Count entries with optional filters."""
        count = 0
        for entry in self._entries.values():
            if entry_type and entry.entry_type != entry_type:
                continue
            if project_id and entry.project_id != project_id:
                continue
            if status and entry.status != status:
                continue
            count += 1
        return count

    def _save(self) -> None:
        """Save entries to storage."""
        if not self._storage_path:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": {
                k: v.to_dict() for k, v in self._entries.items()
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._storage_path.write_text(json.dumps(data, indent=2, default=str))

    def load(self) -> None:
        """Load entries from storage."""
        if not self._storage_path or not self._storage_path.exists():
            return
        try:
            data = json.loads(self._storage_path.read_text())
            for entry_id, entry_data in data.get("entries", {}).items():
                self._entries[entry_id] = MemoryEntry.from_dict(entry_data)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to load memory store: %s", exc)


# ── Scientific Memory Manager ─────────────────────────────────────────────


class ScientificMemoryManager:
    """Manage scientific memory across a project.

    Provides a high-level API for adding, searching, and managing
    memory entries. Handles automatic registration of generated artifacts.
    """

    def __init__(self, store: ScientificMemoryStore | None = None) -> None:
        self._store = store or ScientificMemoryStore()

    @property
    def store(self) -> ScientificMemoryStore:
        return self._store

    def register_generated_code(
        self,
        *,
        project_id: str,
        code: str,
        summary: str = "",
        language: str = "python",
        created_by: str = "",
        experiment_id: str = "",
    ) -> str:
        """Register generated code in scientific memory."""
        entry = MemoryEntry(
            entry_id=f"code-{_generate_id()}",
            entry_type=MemoryEntryType.GENERATED_CODE,
            project_id=project_id,
            experiment_id=experiment_id,
            source="agent",
            content=code,
            summary=summary or f"Generated {language} code",
            metadata={"language": language},
            tags=[language, "generated"],
            created_by=created_by,
            confidence=0.7,
        )
        return self._store.add(entry)

    def register_experiment_log(
        self,
        *,
        project_id: str,
        experiment_id: str,
        log_content: str,
        summary: str = "",
        created_by: str = "",
    ) -> str:
        """Register an experiment log in scientific memory."""
        entry = MemoryEntry(
            entry_id=f"explog-{_generate_id()}",
            entry_type=MemoryEntryType.EXPERIMENT_LOG,
            project_id=project_id,
            experiment_id=experiment_id,
            source="system",
            content=log_content,
            summary=summary or f"Experiment log for {experiment_id}",
            tags=["experiment", "log"],
            created_by=created_by,
            confidence=1.0,
        )
        return self._store.add(entry)

    def register_plot_figure(
        self,
        *,
        project_id: str,
        figure_uri: str,
        summary: str = "",
        experiment_id: str = "",
        created_by: str = "",
    ) -> str:
        """Register a plot/figure in scientific memory."""
        entry = MemoryEntry(
            entry_id=f"fig-{_generate_id()}",
            entry_type=MemoryEntryType.PLOT_FIGURE,
            project_id=project_id,
            experiment_id=experiment_id,
            source="agent",
            content_uri=figure_uri,
            summary=summary or "Generated figure",
            tags=["figure", "plot"],
            created_by=created_by,
            confidence=0.8,
        )
        return self._store.add(entry)

    def register_analysis_result(
        self,
        *,
        project_id: str,
        analysis: str,
        summary: str = "",
        experiment_id: str = "",
        created_by: str = "",
    ) -> str:
        """Register an analysis result in scientific memory."""
        entry = MemoryEntry(
            entry_id=f"analysis-{_generate_id()}",
            entry_type=MemoryEntryType.ANALYSIS_RESULT,
            project_id=project_id,
            experiment_id=experiment_id,
            source="agent",
            content=analysis,
            summary=summary or "Analysis result",
            tags=["analysis"],
            created_by=created_by,
            confidence=0.8,
        )
        return self._store.add(entry)

    def register_new_hypothesis(
        self,
        *,
        project_id: str,
        hypothesis: str,
        confidence: float = 0.5,
        created_by: str = "",
    ) -> str:
        """Register a new hypothesis in scientific memory."""
        entry = MemoryEntry(
            entry_id=f"hyp-{_generate_id()}",
            entry_type=MemoryEntryType.NEW_HYPOTHESIS,
            project_id=project_id,
            source="agent",
            content=hypothesis,
            summary=f"Hypothesis: {hypothesis[:100]}",
            tags=["hypothesis"],
            created_by=created_by,
            confidence=confidence,
        )
        return self._store.add(entry)

    def register_review_comment(
        self,
        *,
        project_id: str,
        comment: str,
        target_entry_id: str = "",
        created_by: str = "",
    ) -> str:
        """Register a review comment in scientific memory."""
        entry = MemoryEntry(
            entry_id=f"review-{_generate_id()}",
            entry_type=MemoryEntryType.REVIEW_COMMENT,
            project_id=project_id,
            source="user",
            content=comment,
            summary=f"Review: {comment[:100]}",
            tags=["review", "comment"],
            created_by=created_by,
            confidence=1.0,
            references=[target_entry_id] if target_entry_id else [],
        )
        return self._store.add(entry)

    def register_accepted_finding(
        self,
        *,
        project_id: str,
        finding: str,
        confidence: float = 0.9,
        created_by: str = "",
    ) -> str:
        """Register an accepted finding in scientific memory."""
        entry = MemoryEntry(
            entry_id=f"finding-{_generate_id()}",
            entry_type=MemoryEntryType.ACCEPTED_FINDING,
            project_id=project_id,
            source="user",
            content=finding,
            summary=f"Finding: {finding[:100]}",
            tags=["finding", "accepted"],
            created_by=created_by,
            confidence=confidence,
        )
        return self._store.add(entry)

    def register_failed_experiment(
        self,
        *,
        project_id: str,
        experiment_id: str,
        failure_reason: str,
        log_content: str = "",
        created_by: str = "",
    ) -> str:
        """Register a failed experiment in scientific memory."""
        entry = MemoryEntry(
            entry_id=f"failed-{_generate_id()}",
            entry_type=MemoryEntryType.FAILED_EXPERIMENT,
            project_id=project_id,
            experiment_id=experiment_id,
            source="system",
            content=f"Failure: {failure_reason}\n\nLog:\n{log_content}",
            summary=f"Failed experiment: {failure_reason[:100]}",
            tags=["failed", "experiment"],
            created_by=created_by,
            confidence=1.0,
        )
        return self._store.add(entry)

    def search(
        self,
        query: str,
        *,
        project_id: str | None = None,
        entry_type: MemoryEntryType | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Search scientific memory."""
        return self._store.search(
            query,
            entry_type=entry_type,
            project_id=project_id,
            limit=limit,
        )

    def list_by_type(
        self,
        entry_type: MemoryEntryType,
        *,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List entries by type."""
        return self._store.list_entries(
            entry_type=entry_type,
            project_id=project_id,
            limit=limit,
        )


# ── Helper Functions ───────────────────────────────────────────────────────


def _generate_id() -> str:
    """Generate a unique ID for a memory entry."""
    import uuid
    return uuid.uuid4().hex[:12]


# ── Convenience Functions ──────────────────────────────────────────────────


def create_scientific_memory(
    storage_path: str | Path | None = None,
) -> ScientificMemoryManager:
    """Create a scientific memory manager with optional storage path."""
    store = ScientificMemoryStore(storage_path=storage_path)
    store.load()
    return ScientificMemoryManager(store=store)
