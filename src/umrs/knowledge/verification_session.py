"""Verification session — persists evidence IDs and graph traversal.

M3 Phase 3: Knowledge-Driven Planning.

M3 Decision 2: VerificationAnswer is NOT a KnowledgeObject.
The LLM answer is ephemeral and can be regenerated.

What IS persistent is:
  - Question
  - Evidence IDs (which UMRS objects were cited)
  - KnowledgeGraph Traversal (which path through the graph was taken)
  - Timestamp

This allows the answer to be regenerated from the evidence IDs and
graph traversal, avoiding enormous duplication from persisting every
LLM response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from umrs.graph_reasoning import GraphReasoningContext

__all__ = [
    "VerificationSession",
    "VerificationSessionStore",
]


@dataclass
class VerificationSession:
    """A verification session that persists evidence and graph traversal.

    The LLM answer is NOT persisted — it's ephemeral and can be
    regenerated from the evidence IDs and graph traversal.

    Attributes:
        session_id: Unique identifier for this session
        project_id: The project this session belongs to
        question: The question asked
        evidence_ids: UMRS object IDs that were cited as evidence
        graph_traversal: Path through the knowledge graph
        reasoning_context: Complete reasoning context from the graph
        timestamp: When the session was created
        answer: The LLM-generated answer (ephemeral, NOT persisted)
        confidence: Confidence score (0.0–1.0)
    """

    session_id: str = ""
    project_id: str = ""
    question: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    graph_traversal: list[str] = field(default_factory=list)
    reasoning_context: GraphReasoningContext | None = None
    timestamp: str = ""
    answer: str = ""  # Ephemeral — NOT persisted
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = f"vsess-{uuid4().hex[:12]}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_persistent_dict(self) -> dict[str, Any]:
        """Convert to dictionary for PERSISTENT storage.

        This excludes the ephemeral LLM answer.
        """
        return {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "question": self.question,
            "evidence_ids": self.evidence_ids,
            "graph_traversal": self.graph_traversal,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary including ephemeral answer."""
        data = self.to_persistent_dict()
        data["answer"] = self.answer
        return data


class VerificationSessionStore:
    """Store for verification sessions.

    Only persistent data is stored (evidence IDs, graph traversal,
    timestamp). The ephemeral LLM answer is NOT stored.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def save(self, session: VerificationSession) -> str:
        """Save a verification session (persistent data only).

        Args:
            session: The session to save

        Returns:
            The session_id
        """
        self._sessions[session.session_id] = session.to_persistent_dict()
        return session.session_id

    def get(self, session_id: str) -> dict[str, Any] | None:
        """Get a verification session by ID.

        Args:
            session_id: The session ID

        Returns:
            The persistent session data, or None if not found
        """
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        *,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List verification sessions with optional filters.

        Args:
            project_id: Filter by project ID
            limit: Maximum number of sessions to return

        Returns:
            List of matching sessions, newest first
        """
        results = []
        for session in self._sessions.values():
            if project_id and session.get("project_id") != project_id:
                continue
            results.append(session)
        results.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
        return results[:limit]

    def delete(self, session_id: str) -> bool:
        """Delete a verification session.

        Args:
            session_id: The session ID

        Returns:
            True if the session was found and deleted, False otherwise
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
