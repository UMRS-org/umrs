"""A2L conservation domain models (ADR-034).

Wire format is the stdlib r4-contracts layer (ADR-005). This module
re-exports those DTOs under domain names and adds the few persistence-only
shapes the contracts deliberately leave out. None of these models copy
execution state from the spine — they reference it by ``job_id`` /
``attempt_id`` only (join-not-copy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from r4_contracts.a2l import (
    A2lAgentExecutionCreate,
    A2lAgentExecutionRecord,
    A2lAgentExecutionUpdate,
    A2lArtifactDeriveRequest,
    A2lArtifactRecord,
)

# Domain aliases — same wire shape as the contracts.
ArtifactRecord = A2lArtifactRecord
ArtifactDerive = A2lArtifactDeriveRequest
AgentExecution = A2lAgentExecutionRecord
AgentExecutionCreate = A2lAgentExecutionCreate
AgentExecutionUpdate = A2lAgentExecutionUpdate


@dataclass
class ArtifactStatusUpdate:
    """FSM-guarded status patch for one conserved artifact."""

    status: str
    error: str = ""


@dataclass
class ExecutionArtifactLink:
    """Link between one agent execution and one artifact (input/output)."""

    execution_id: str
    artifact_id: str
    direction: str  # 'input' | 'output'

    def to_row(self) -> tuple[str, str, str]:
        return (self.execution_id, self.artifact_id, self.direction)


@dataclass
class DecisionCreate:
    """Create request for one human/operator decision (conserved addition)."""

    decided_by: str
    decision: str
    rationale: str = ""
    job_id: str = ""
    attempt_id: str = ""
    artifact_id: str = ""
    payload: dict = field(default_factory=dict)


@dataclass
class DecisionRecord:
    """One conserved decision row."""

    id: str
    decided_by: str
    decision: str
    rationale: str
    job_id: str
    attempt_id: str
    artifact_id: str
    payload: dict
    created_at: str
