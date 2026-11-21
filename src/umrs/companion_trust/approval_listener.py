"""Approval listener of the companion trust ledger (U3 S3e, decision D7).

ADR-036 §3: an ``always`` decision mints a durable ``session:<loop_id>`` trust
row. The decision route (core ``api/approvals.py``) no longer knows the trust
store: it calls the registered approval listeners
(``ExtensionContext.add_approval_listener``), and the in-tree r4 part
registers this one. Moves to ``umrs_r4`` with the trust cluster (S4).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def mint_session_trust_on_always(record: dict, decision: str, user_id: str) -> None:
    """Best effort: the decision itself is already durable."""
    if decision != "always":
        return
    loop_id = str(record.get("loop_id") or "")
    from umrs.shared.trust_store_access import get_trust_store

    trust_store = get_trust_store()
    if not loop_id or trust_store is None:
        return
    action = record.get("action") or {}
    tool_name = str(
        action.get("workflow_name") or action.get("kind") or "workflow_trigger"
    )
    try:
        trust_store.grant(
            user_id, tool_name, scope=f"session:{loop_id}", created_by=user_id
        )
    except Exception:
        logger.warning(
            "always-decision trust grant failed for loop %s", loop_id, exc_info=True
        )
