"""Database schema for research sessions (ADR-025 D1/D2).

Tables:
- research_sessions: durable investigation umbrella
- research_session_loops: loop membership per session (venue-agnostic)

Versioned via the shared ``schema_versions`` table with component
``research_sessions``. New component — no existing schema bumped.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Schema version for this component (tracked in schema_versions)
RESEARCH_SESSION_SCHEMA_VERSION = 1

CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS research_sessions (
    session_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    research_goal TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'completed', 'archived')),
    authoritative_runtime TEXT NOT NULL DEFAULT 'umrs'
        CHECK (authoritative_runtime IN ('umrs', 'local', 'hybrid')),
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    archived_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_research_sessions_project
    ON research_sessions(project_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_sessions_status
    ON research_sessions(status);

COMMENT ON TABLE research_sessions IS
    'Durable research investigation umbrella (ADR-025). Purely additive component.';
"""

CREATE_LOOPS_TABLE = """
CREATE TABLE IF NOT EXISTS research_session_loops (
    session_id TEXT NOT NULL REFERENCES research_sessions(session_id)
        ON DELETE CASCADE,
    loop_id TEXT NOT NULL,
    venue TEXT NOT NULL DEFAULT 'umrs'
        CHECK (venue IN ('umrs', 'local')),
    loop_role TEXT NOT NULL DEFAULT 'auto',
    attached_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (session_id, loop_id)
);

CREATE INDEX IF NOT EXISTS idx_session_loops_loop
    ON research_session_loops(loop_id);

COMMENT ON TABLE research_session_loops IS
    'Link between sessions and curated loop ids (venue local rows are references only).';
"""

CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_versions (
    component VARCHAR(64) PRIMARY KEY,
    version INTEGER NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def ensure_schema(conn) -> bool:
    """Ensure the research_sessions schema exists.

    Creates tables + indexes if missing and registers this component's
    version. Returns True when work was done.
    """
    with conn.cursor() as cur:
        cur.execute(CREATE_SCHEMA_VERSION_TABLE)
        cur.execute(
            "SELECT version FROM schema_versions WHERE component = 'research_sessions'"
        )
        row = cur.fetchone()
        current_version = row[0] if row else 0

        if current_version >= RESEARCH_SESSION_SCHEMA_VERSION:
            logger.debug(
                "Research sessions schema already at version %s",
                current_version,
            )
            return False

        logger.info(
            "Creating/updating research sessions schema to version %s",
            RESEARCH_SESSION_SCHEMA_VERSION,
        )
        cur.execute(CREATE_SESSIONS_TABLE)
        cur.execute(CREATE_LOOPS_TABLE)
        cur.execute(
            """
            INSERT INTO schema_versions (component, version)
            VALUES ('research_sessions', %s)
            ON CONFLICT (component) DO UPDATE SET version = %s, applied_at = NOW()
            """,
            (
                RESEARCH_SESSION_SCHEMA_VERSION,
                RESEARCH_SESSION_SCHEMA_VERSION,
            ),
        )
    conn.commit()
    return True


def drop_schema(conn, confirm: bool = False) -> None:
    """Drop the research-sessions schema. USE WITH CAUTION — deletes data."""
    if not confirm:
        raise ValueError("Must set confirm=True to drop schema")
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS research_session_loops CASCADE")
        cur.execute("DROP TABLE IF EXISTS research_sessions CASCADE")
        cur.execute("DELETE FROM schema_versions WHERE component = 'research_sessions'")
    conn.commit()
    logger.warning("Research sessions schema dropped")
