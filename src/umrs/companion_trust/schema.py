"""Schema for the companion trust allowlist (ADR-035 follow-on).

One table:

- companion_trust_allowlist: (user_id, scope, tool_name) → expiry

New component — no existing schema bumped. Mirrors 0066/0071 one-line
COMMENT style. Sqlite translation handled by the store's ``_sqliteify``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

COMPANION_TRUST_SCHEMA_VERSION = 2

CREATE_TRUST_TABLE = """
CREATE TABLE IF NOT EXISTS companion_trust_allowlist (
    user_id TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global',
    tool_name TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by TEXT NOT NULL DEFAULT '',

    PRIMARY KEY (user_id, scope, tool_name)
);

CREATE INDEX IF NOT EXISTS idx_companion_trust_user
    ON companion_trust_allowlist(user_id, expires_at);

COMMENT ON TABLE companion_trust_allowlist IS
    'Per-user-scope TRUST allowlist (ADR-035). Server record consulted by autopilot.';
"""

# v2 (2026-09-03): session lifecycle cascade — tombstone for companion-session
# archive + one audit row per revoked grant so "why did my always-allow
# disappear" has a durable answer (reason=session_archived).
CREATE_SESSION_ARCHIVE_TABLE = """
CREATE TABLE IF NOT EXISTS companion_session_archives (
    session_id TEXT PRIMARY KEY,
    archived_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    archived_by TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_companion_session_archives_at
    ON companion_session_archives(archived_at DESC);

COMMENT ON TABLE companion_session_archives IS
    'Tombstones for archived companion sessions (ADR-035 item-8 cascade).';
"""

CREATE_REVOKE_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS companion_trust_revoke_log (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    reason TEXT NOT NULL,
    revoked_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_companion_trust_revoke_log_session
    ON companion_trust_revoke_log(session_id);

COMMENT ON TABLE companion_trust_revoke_log IS
    'Audit of session-lifecycle trust revocations (one row per revoked grant).';
"""

CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_versions (
    component VARCHAR(64) PRIMARY KEY,
    version INTEGER NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def ensure_schema(conn) -> bool:
    """Ensure the companion_trust schema exists.

    Creates table + index if missing and registers this component's version.
    Returns True when work was done.
    """
    with conn.cursor() as cur:
        cur.execute(CREATE_SCHEMA_VERSION_TABLE)
        cur.execute(
            "SELECT version FROM schema_versions WHERE component = 'companion_trust'"
        )
        row = cur.fetchone()
        current_version = row[0] if row else 0

        if current_version >= COMPANION_TRUST_SCHEMA_VERSION:
            logger.debug(
                "Companion trust schema already at version %s",
                current_version,
            )
            return False

        logger.info(
            "Creating companion trust schema at version %s",
            COMPANION_TRUST_SCHEMA_VERSION,
        )
        cur.execute(CREATE_TRUST_TABLE)
        cur.execute(CREATE_SESSION_ARCHIVE_TABLE)
        cur.execute(CREATE_REVOKE_LOG_TABLE)
        cur.execute(
            """
            INSERT INTO schema_versions (component, version)
            VALUES ('companion_trust', %s)
            ON CONFLICT (component) DO UPDATE SET version = %s, applied_at = NOW()
            """,
            (
                COMPANION_TRUST_SCHEMA_VERSION,
                COMPANION_TRUST_SCHEMA_VERSION,
            ),
        )
    conn.commit()
    return True


def drop_schema(conn, confirm: bool = False) -> None:
    """Drop the companion-trust schema. USE WITH CAUTION — deletes data."""
    if not confirm:
        raise ValueError("Must set confirm=True to drop schema")
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS companion_trust_revoke_log CASCADE")
        cur.execute("DROP TABLE IF EXISTS companion_session_archives CASCADE")
        cur.execute("DROP TABLE IF EXISTS companion_trust_allowlist CASCADE")
        cur.execute("DELETE FROM schema_versions WHERE component = 'companion_trust'")
    conn.commit()
    logger.warning("Companion trust schema dropped")
