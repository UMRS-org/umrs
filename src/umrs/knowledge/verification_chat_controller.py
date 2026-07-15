"""Verification Chat Controller — bridges TUI to Verification Chat.

M3.5 Phase 4: Interactive Verification Chat.

The verification chat is NOT a generic chatbot. It queries UMRS
through the Scientific Reasoning Engine and presents evidence,
provenance, confidence, and graph paths.

Architecture:
  TUI → VerificationChatController → ScientificReasoningEngine → GraphNavigator → UMRS

TASK-046: Persistence layer added so chat-resolved findings and
sessions survive UMRS restarts.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from umrs.knowledge.verification_session import (
    VerificationSession,
    VerificationSessionStore,
)
from umrs.knowledge.verification_store import VerificationStore
from umrs.scientific_reasoning_engine import ScientificReasoningEngine

logger = logging.getLogger(__name__)

__all__ = [
    "VerificationAnswer",
    "VerificationChatController",
]


@dataclass
class VerificationAnswer:
    """Answer to a verification question.

    Attributes:
        question: The question asked
        answer: The LLM-generated answer (ephemeral, NOT persisted)
        confidence: Confidence score (0.0–1.0)
        evidence_ids: UMRS object IDs cited as evidence
        graph_paths: Paths through the knowledge graph
        reasoning_steps: Step-by-step reasoning trace
    """

    question: str = ""
    answer: str = ""
    confidence: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    graph_paths: list[list[str]] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
            "graph_paths": self.graph_paths,
            "reasoning_steps": self.reasoning_steps,
        }


class VerificationChatController:
    """Controller for the Interactive Verification Chat.

    Bridges between the TUI and the Scientific Reasoning Engine.

    Optionally backed by persistent stores so that chat-resolved
    findings and sessions survive UMRS restarts (TASK-046).

    Usage:
        controller = VerificationChatController(
            reasoning_engine=reasoning_engine,
            store=store,
            session_store=session_store,
        )
        answer = await controller.ask(
            "Explain this project",
            anchor_object_id="paper-001",
            project_id="my-project",
        )
    """

    def __init__(
        self,
        *,
        reasoning_engine: ScientificReasoningEngine | None = None,
        store: VerificationStore | None = None,
        session_store: VerificationSessionStore | None = None,
    ) -> None:
        self._reasoning_engine = reasoning_engine or ScientificReasoningEngine()
        self._store = store
        self._session_store = session_store

    async def ask(
        self,
        question: str,
        *,
        anchor_object_id: str | None = None,
        project_id: str = "default",
    ) -> VerificationAnswer:
        """Ask a verification question and persist the session."""
        result = self._reasoning_engine.reason(
            query=question,
            anchor_object_id=anchor_object_id,
        )

        answer = VerificationAnswer(
            question=question,
            answer=result.answer,
            confidence=result.confidence,
            evidence_ids=result.evidence_ids,
            graph_paths=result.graph_paths,
            reasoning_steps=result.reasoning_steps,
        )

        # Persist session (evidence + traversal only, NOT ephemeral answer)
        await self._persist_session(answer, project_id=project_id)

        return answer

    async def _persist_session(
        self, answer: VerificationAnswer, *, project_id: str
    ) -> None:
        """Persist the session data to the store (if available)."""
        session_id = f"sess-{hashlib.md5(answer.question.encode()).hexdigest()[:12]}"

        # In-memory session store (for TUI)
        if self._session_store is not None:
            self._session_store.save(
                VerificationSession(
                    session_id=session_id,
                    project_id=project_id,
                    question=answer.question,
                    evidence_ids=answer.evidence_ids,
                    graph_traversal=[str(p) for p in answer.graph_paths[0]] if answer.graph_paths else [],
                    confidence=answer.confidence,
                    answer=answer.answer,
                )
            )

        # Postgres-backed session store (for persistence across restarts)
        if self._store is not None:
            try:
                await self._store.save_verification_session(
                    session_id=session_id,
                    project_id=project_id,
                    question=answer.question,
                    evidence_ids=answer.evidence_ids,
                    graph_traversal=[str(p) for p in answer.graph_paths[0]] if answer.graph_paths else [],
                    confidence=answer.confidence,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist verification session: %s", exc)

    async def save_finding(
        self,
        answer: VerificationAnswer,
        finding_key: str,
        *,
        status: str = "open",
        category: str = "verification_chat",
        severity: str = "warning",
        project_id: str = "default",
        report_id: str = "manual",
    ) -> None:
        """Persist a chat-resolved finding to the knowledge store (TASK-046)."""
        if self._store is None:
            return
        try:
            await self._store.upsert_finding(
                finding_key=finding_key,
                report_id=report_id,
                category=category,
                severity=severity,
                title=answer.question[:80],
                description=answer.answer[:500],
                evidence="|".join(answer.evidence_ids),
                status=status,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist finding %s: %s", finding_key, exc)

    async def transition_finding(
        self,
        finding_key: str,
        from_status: str,
        to_status: str,
    ) -> dict[str, Any]:
        """Transition a finding state (open → resolved, etc.)."""
        if self._store is None:
            return {}
        try:
            finding = await self._store.transition_finding(
                finding_key, from_status, to_status
            )
            return {
                "finding_key": finding.finding_key,
                "status": finding.status,
                "title": finding.title,
            }
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    async def list_findings(
        self,
        *,
        report_id: str | None = None,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """List findings from the store (TASK-046)."""
        if self._store is None:
            return []
        results = await self._store.list_findings(
            report_id=report_id,
            status=status,
            severity=severity,
            limit=limit,
        )
        return [
            {
                "finding_key": f.finding_key,
                "category": f.category,
                "severity": f.severity,
                "status": f.status,
                "title": f.title,
                "description": f.description,
                "evidence": f.evidence,
            }
            for f in results
        ]

    async def list_sessions(
        self,
        *,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List verification sessions (TASK-046)."""
        if self._session_store is not None:
            return self._session_store.list_sessions(
                project_id=project_id, limit=limit
            )
        if self._store is not None:
            return await self._store.list_verification_sessions(
                project_id=project_id, limit=limit
            )
        return []

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get a single verification session by ID (TASK-046)."""
        if self._store is not None:
            return await self._store.get_verification_session(session_id)
        if self._session_store is not None:
            return self._session_store.get(session_id)
        return None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a verification session (TASK-046)."""
        if self._store is not None:
            return await self._store.delete_verification_session(session_id)
        if self._session_store is not None:
            return self._session_store.delete(session_id)
        return False
