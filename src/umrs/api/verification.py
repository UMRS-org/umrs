"""UMRS API routes for M3.5 verification chat."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from umrs.graph_reasoning import GraphNavigator
from umrs.knowledge.verification_chat_controller import VerificationChatController
from umrs.scientific_reasoning_engine import ScientificReasoningEngine

# Reuse the shared M3.5 knowledge graph so verification chat can reason
# over the same state that populates the dashboard/explorer/readiness views.
from .knowledge import _m35_knowledge_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/verification", tags=["verification"])

_m35_navigator = GraphNavigator(graph=_m35_knowledge_graph)
_m35_reasoning_engine = ScientificReasoningEngine(navigator=_m35_navigator)
_verification_controller = VerificationChatController(
    reasoning_engine=_m35_reasoning_engine,
)


class VerificationChatRequest(BaseModel):
    question: str = Field(..., description="Verification question")
    anchor_object_id: str | None = Field(None, description="Anchor object ID")


@router.post("/chat")
async def ask_verification_chat(req: VerificationChatRequest) -> dict[str, Any]:
    """Ask a verification question backed by the scientific reasoning engine."""
    answer = _verification_controller.ask(
        question=req.question,
        anchor_object_id=req.anchor_object_id,
    )
    return answer.to_dict()
