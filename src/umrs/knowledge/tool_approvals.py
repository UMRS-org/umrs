"""Tool-grain approval inbox facade for loop-driven action gates (ADR-036 (a)).

The process holds ONE :class:`~umrs.companion_trust.approval_inbox.ApprovalInboxStore`;
the engine driver parks on its rows, the companion approvals routes
(``GET /approvals/pending``, ``POST /approvals/{id}/decision``) read and flip
them, and the sidecar installs the Postgres-backed store at boot via
:func:`configure`. Until configured, the facade runs on a process-local
sqlite store and says so (``durability() == "process"``) — never silently.

Design contract:
- Approval intent = exactly one tool-grain action per (loop, iteration),
  keyed by ``approval_id`` and carrying the loop owner + job/attempt.
- Decisions: ``once``/``always`` → approved, ``no`` → rejected. ``always``
  additionally flips the loop's ``metadata.mode`` to ``auto`` and mints a
  session-scope trust row at the route layer.
- Restart: a parked loop carries ``metadata.awaiting_approval``; its gate
  re-requests under the same id and finds the durable row — including a
  decision recorded while the process was down (no re-ask).
"""

from __future__ import annotations

import threading
from typing import Any

from umrs.companion_trust.approval_inbox import (
    APPROVED_DECISIONS,
    REJECTED_DECISIONS,
    VALID_DECISIONS,
    ApprovalInboxStore,
)

_LOCK = threading.Lock()
_STORE: ApprovalInboxStore | None = None


def configure(store: ApprovalInboxStore | None) -> None:
    """Install the process inbox (sidecar boot); ``None`` resets to process-local."""
    global _STORE
    with _LOCK:
        _STORE = store


def get_store() -> ApprovalInboxStore:
    """The process inbox — a process-local sqlite store until configured."""
    global _STORE
    with _LOCK:
        if _STORE is None:
            _STORE = ApprovalInboxStore()
        return _STORE


def durability() -> str:
    """``postgres`` | ``sqlite`` | ``process`` — surfaced on the approvals route."""
    return get_store().durability


def request_approval(
    loop_id: str,
    iteration: int,
    action: dict[str, Any],
    *,
    approval_id: str | None = None,
    user_id: str = "",
    job_id: str = "",
    attempt_id: str = "",
) -> dict[str, Any]:
    """Register an approval intent (idempotent when approval_id is given)."""
    return get_store().request(
        loop_id,
        iteration,
        action,
        approval_id=approval_id,
        user_id=user_id,
        job_id=job_id,
        attempt_id=attempt_id,
    )


def get(approval_id: str) -> dict[str, Any] | None:
    return get_store().get(approval_id)


def pending(
    *, loop_id: str | None = None, user_id: str | None = None
) -> list[dict[str, Any]]:
    return get_store().pending(loop_id=loop_id, user_id=user_id)


def decide(approval_id: str, decision: str, decided_by: str) -> dict[str, Any]:
    """Flip a pending intent to approved/rejected. Terminal transitions only."""
    return get_store().decide(approval_id, decision, decided_by)


def clear_all() -> None:
    """Test utility: wipe all intents in the current store."""
    get_store().clear_all()


__all__ = [
    "APPROVED_DECISIONS",
    "REJECTED_DECISIONS",
    "VALID_DECISIONS",
    "clear_all",
    "configure",
    "decide",
    "durability",
    "get",
    "get_store",
    "pending",
    "request_approval",
]
