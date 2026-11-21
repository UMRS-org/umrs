"""Session-archive participant of the companion trust ledger (U3 S3d).

ADR-035 item 8: archiving a research session revokes its session-scope trust
rows and writes the ``companion_session_archives`` tombstone, in the archive's
own transaction. Core's archive (``research_sessions/store.py``) no longer
knows the trust store: it runs the registered participants
(``ExtensionContext.add_session_archive_participant``), and the in-tree r4
part registers this one. Moves to ``umrs_r4`` with the trust cluster (S4).
"""

from __future__ import annotations

import sqlite3
from typing import Any

_TOMBSTONE_PG = (
    "INSERT INTO companion_session_archives (session_id, archived_by) "
    "VALUES (%s, %s) ON CONFLICT (session_id) DO NOTHING"
)
_TOMBSTONE_SQLITE = "INSERT OR IGNORE INTO companion_session_archives (session_id, archived_by) VALUES (?, ?)"


def revoke_trust_on_archive(cur: Any, session_id: str, archived_by: str) -> None:
    """Tombstone + session-scope revoke on the archive's cursor (no-op unwired)."""
    from umrs.shared.trust_store_access import get_trust_store

    trust_store = get_trust_store()
    if trust_store is None:
        return
    # Tombstone INSIDE the same transaction: check() consults
    # companion_session_archives, so every archive channel must write it.
    # ON CONFLICT no-op: channels may race; first tombstone wins.
    sqlite = isinstance(cur.connection, sqlite3.Connection)
    cur.execute(
        _TOMBSTONE_SQLITE if sqlite else _TOMBSTONE_PG, (session_id, archived_by)
    )
    trust_store.revoke_session_scope_conn(cur.connection, session_id)
