"""Circle membership — the ONE SQL resolution (ADR-086).

API routes resolve through the request pool (``umrs.api._access``);
server-side stores join ``circle_members`` directly (the rag corpus and
user-bucket SQL checks), and the workflow catalog via the default pool
provider in ``workflow_registry_handler``. Retired circles never resolve.
"""

from __future__ import annotations

from typing import Any


def circle_refs_for_user(conn: Any, user_id: str) -> list[str]:
    """Active circle refs (``<project_id>/<circle_id>``) of the user.

    Sorted output keeps tests deterministic. The ref shape matches the id
    part of the grammar kind ``circle:<project_id>/<circle_id>`` so guards
    compare strings directly.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT cm.project_id || '/' || cm.circle_id FROM circle_members cm"
        " JOIN circles c ON c.project_id = cm.project_id"
        " AND c.circle_id = cm.circle_id"
        " WHERE cm.user_id = %s AND c.status = 'active'"
        " ORDER BY cm.project_id, cm.circle_id",
        (user_id,),
    )
    return [r[0] for r in cur.fetchall()]


__all__ = ["circle_refs_for_user"]
