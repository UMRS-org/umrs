"""A2L conservation domain (ADR-034).

Bounded UMRS module owning the *conserved additions* to execution history:
deterministic artifact identities, operational lineage edges, agent-execution
annotations, and human/operator decisions — all keyed to the authoritative
execution spine by ``job_id`` / ``attempt_id``. Read models join against the
spine at query time; a2l never copies execution state.

Internal-layer rule: this module must not import from ``umrs.api`` or
``umrs.server`` (enforced by tools/import_boundary_validator.py).

Known blocker upstream: the API package's flat ``__init__`` unconditionally
imports every router on ``from umrs.api import x`` resolution, including
optional routers (e.g. ``companion.py`` at import time) and module-level
singletons (``api.companion_up``) that raise ``ModuleNotFound`` when optional
deps are absent. Chancellor tracked for remediation.
"""

from .models import (
    AgentExecution,
    AgentExecutionCreate,
    AgentExecutionUpdate,
    ArtifactDerive,
    ArtifactRecord,
    ArtifactStatusUpdate,
    DecisionCreate,
    DecisionRecord,
    ExecutionArtifactLink,
)
from .store import ConservationStore, ConservationStoreUnavailable, reset_store

__all__ = [
    "AgentExecution",
    "AgentExecutionCreate",
    "AgentExecutionUpdate",
    "ArtifactDerive",
    "ArtifactRecord",
    "ArtifactStatusUpdate",
    "ConservationStore",
    "ConservationStoreUnavailable",
    "DecisionCreate",
    "DecisionRecord",
    "ExecutionArtifactLink",
    "reset_store",
]
