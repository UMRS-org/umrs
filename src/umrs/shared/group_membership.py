"""Group membership resolution — the ONE SQL authority (migration 0097).

Groups are platform-wide; a user may hold many memberships; retired groups
never resolve (their grants stop counting immediately). Guards resolve
lazily through this function — API routes via the request pool
(``umrs.api._access``), server-side stores via their own connections (ADR-080
SQL checks join ``group_members`` directly), and the workflow catalog via
the default pool provider in ``umrs.server.workflow_registry_handler``.
"""

from __future__ import annotations

from typing import Any


def group_ids_for_user(conn: Any, user_id: str) -> list[str]:
    """Active group ids the user is a member of (multi-membership).

    Retired groups never resolve — their grants stop counting immediately
    (tombstone semantics from the migration comment).
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT gm.group_id FROM group_members gm"
        " JOIN groups g ON g.group_id = gm.group_id"
        " WHERE gm.user_id = %s AND g.status = 'active'"
        " ORDER BY gm.group_id",
        (user_id,),
    )
    return [r[0] for r in cur.fetchall()]
