"""Verification Chat Controller — bridges TUI to Verification Chat.

M3.5 Phase 4: Interactive Verification Chat.

The verification chat is NOT a generic chatbot. It queries UMRS
through the Scientific Reasoning Engine and presents evidence,
provenance, confidence, and graph paths.

Architecture:
  TUI → VerificationChatController → ScientificReasoningEngine → GraphNavigator → UMRS
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from umrs.scientific_reasoning_engine import (
    ReasoningResult,
    ScientificReasoningEngine,
)

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

    Usage:
        controller = VerificationChatController(
            reasoning_engine=reasoning_engine,
        )
        answer = controller.ask("Explain this project", anchor_object_id="paper-001")
    """

    def __init__(
        self,
        *,
        reasoning_engine: ScientificReasoningEngine | None = None,
    ) -> None:
        self._reasoning_engine = reasoning_engine or ScientificReasoningEngine()

    def ask(
        self,
        question: str,
        *,
        anchor_object_id: str | None = None,
    ) -> VerificationAnswer:
        """Ask a verification question.

        Args:
            question: The verification question
            anchor_object_id: The anchor object to start reasoning from

        Returns:
            VerificationAnswer with answer, confidence, evidence, and graph paths
        """
        result = self._reasoning_engine.reason(
            query=question,
            anchor_object_id=anchor_object_id,
        )

        return VerificationAnswer(
            question=question,
            answer=result.answer,
            confidence=result.confidence,
            evidence_ids=result.evidence_ids,
            graph_paths=result.graph_paths,
            reasoning_steps=result.reasoning_steps,
        )
