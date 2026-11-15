"""Ask-the-scientist action vocabulary shared by both loop venues (ADR-066).

The trigger classes, and which classes an engine action kind carries. UMRS's
server driver (``umrs.knowledge.ask_policy``) and r4's local loop
(``r4.runtime.local_policy``) both read this table, so a kind is classed in
one place. A team-plan run (``plan_run``, ADR-092 slice 6) is ``cost``, as
any workflow trigger or agent iteration is. Behaviour built on the table (the
``physical`` flag, emitted ``competing_hypotheses`` / ``direction_change``
actions, maturity-stage presets) stays with each consumer.

Stdlib only (INV-10).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

COST = "cost"
COMPETING = "competing_hypotheses"
DIRECTION = "direction_change"
PHYSICAL = "physical_action"
TRIGGER_CLASSES = (COST, COMPETING, DIRECTION, PHYSICAL)

WORKFLOW_TRIGGER = "workflow_trigger"
AGENT_ITERATION = "agent_iteration"
PLAN_RUN = "plan_run"

#: action kind → the trigger classes it carries before any per-action flag.
ACTION_KIND_CLASSES: Mapping[str, frozenset] = MappingProxyType(
    {
        WORKFLOW_TRIGGER: frozenset({COST}),
        AGENT_ITERATION: frozenset({COST}),
        PLAN_RUN: frozenset({COST}),
    }
)

__all__ = [
    "ACTION_KIND_CLASSES",
    "AGENT_ITERATION",
    "COMPETING",
    "COST",
    "DIRECTION",
    "PHYSICAL",
    "PLAN_RUN",
    "TRIGGER_CLASSES",
    "WORKFLOW_TRIGGER",
]
