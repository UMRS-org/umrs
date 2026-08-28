"""Pending tool-grain approvals for loop-driven action gates (ADR-028).

In-memory registry of approval intents created when a driven loop wants to
perform a tool-grain action (today: a workflow trigger inside a real-mode
research loop) while running under ``mode=ask``. The engine driver parks on
these; the companion approvals route (``POST /approvals/{id}/decision``)
flips their state.

Design contract:
- Approval intent = exactly one tool-grain action per (loop, iteration).
- Decisions: ``once``/``always`` → approved, ``no`` → rejected. ``always``
  additionally flips the loop's ``metadata.mode`` to ``auto`` at the route
  layer (the loop stops gating from then on).
- Durability is deliberately thin (this module is process state): on
  restart, a parked loop still carries ``metadata.awaiting_approval``; its
  gate rebuilds the intent under the same id — see ``request_approval``'s
  idempotent ``approval_id=`` path.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

_LOCK = threading.Lock()
_PENDING: dict[str, dict[str, Any]] = {}

# Decision vocabulary (kept identical to the companion route contract).
APPROVED_DECISIONS = frozenset({"once", "always"})
REJECTED_DECISIONS = frozenset({"no"})
VALID_DECISIONS = APPROVED_DECISIONS | REJECTED_DECISIONS


def request_approval(
    loop_id: str,
    iteration: int,
    action: dict[str, Any],
    *,
    approval_id: str | None = None,
) -> dict[str, Any]:
    """Register an approval intent (idempotent when approval_id is given).

    Re-registering an existing intent (rehydration after restart, or a
    double gate call) returns the existing record unchanged unless it was
    already decided — decided records are terminal.
    """
    with _LOCK:
        if approval_id and approval_id in _PENDING:
            return dict(_PENDING[approval_id])
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "id": approval_id or f"appr-{uuid4().hex[:12]}",
            "loop_id": loop_id,
            "iteration": iteration,
            "action": dict(action),
            "status": "pending",
            "decided_by": "",
            "decided_at": "",
            "created_at": now,
        }
        _PENDING[record["id"]] = record
        return dict(record)


def get(approval_id: str) -> dict[str, Any] | None:
    with _LOCK:
        record = _PENDING.get(approval_id)
        return dict(record) if record else None


def pending(*, loop_id: str | None = None) -> list[dict[str, Any]]:
    with _LOCK:
        rows = [r for r in _PENDING.values() if r["status"] == "pending"]
        if loop_id:
            rows = [r for r in rows if r["loop_id"] == loop_id]
        return [dict(r) for r in rows]


def decide(approval_id: str, decision: str, decided_by: str) -> dict[str, Any]:
    """Flip a pending intent to approved/rejected. Terminal transitions only."""
    if decision not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(VALID_DECISIONS)}")
    with _LOCK:
        record = _PENDING.get(approval_id)
        if record is None:
            raise KeyError(f"approval not found: {approval_id}")
        if record["status"] != "pending":
            raise ValueError(f"approval {approval_id} already decided")
        record["status"] = "approved" if decision in APPROVED_DECISIONS else "rejected"
        record["decided_by"] = decided_by
        record["decided_at"] = datetime.now(timezone.utc).isoformat()
        return dict(record)


def clear_all() -> None:
    """Test utility: wipe all intents."""
    with _LOCK:
        _PENDING.clear()
