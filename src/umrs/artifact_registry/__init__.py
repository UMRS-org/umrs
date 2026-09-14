"""
Artifact Registry — schema and migrations.

Tables:
- artifacts: registry of all produced/consumed objects with metadata, lineage
  links, and execution association
- artifact_lineage: directed edges (artifact_id → parent_artifact_id)

Follows the same pattern as jobs/schema.py: SCHEMA_VERSION constant,
idempotent CREATE TABLE IF NOT EXISTS, version tracking table,
ensure_schema(conn) migration function.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Schema version - increment when schema changes
ARTIFACT_REGISTRY_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

CREATE_ARTIFACTS_TABLE = """
CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_type TEXT NOT NULL,
    name TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    project_id TEXT,
    uri TEXT NOT NULL,
    bucket TEXT,
    s3_key TEXT,
    size_bytes BIGINT,
    checksum TEXT,
    metadata JSONB DEFAULT '{}',
    execution_id TEXT,
    workflow_id TEXT,
    job_id TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE artifacts IS \
    'Artifact Registry — all produced/consumed objects';
COMMENT ON COLUMN artifacts.artifact_type IS (
    'dataset|model|embedding|result|'
    'image|checkpoint|config|log'
);
COMMENT ON COLUMN artifacts.uri IS 's3://bucket/key or file path';
COMMENT ON COLUMN artifacts.checksum IS 'sha256:...';
"""

CREATE_LINEAGE_TABLE = """
CREATE TABLE IF NOT EXISTS artifact_lineage (
    id BIGSERIAL PRIMARY KEY,
    artifact_id UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    parent_artifact_id UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    relation TEXT NOT NULL DEFAULT 'derived_from',
    UNIQUE (artifact_id, parent_artifact_id, relation)
);

COMMENT ON TABLE artifact_lineage IS 'Artifact lineage edges';
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts (artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_owner ON artifacts (owner_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_execution ON artifacts (execution_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts (project_id);
"""

CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS artifact_registry_schema_version (
    id INTEGER PRIMARY KEY DEFAULT 1,
    version INTEGER NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def ensure_schema(conn) -> bool:
    """Ensure artifact registry schema exists.

    Creates tables and indexes if they don't exist.
    Returns True if schema was created/updated.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_SCHEMA_VERSION_TABLE)

            cur.execute(
                "SELECT version FROM artifact_registry_schema_version WHERE id = 1"
            )
            row = cur.fetchone()
            current_version = row[0] if row else 0

            if current_version >= ARTIFACT_REGISTRY_SCHEMA_VERSION:
                logger.debug("Artifact schema already at version %s", current_version)
                return False

            logger.info(
                "Creating/updating artifact schema to version %s",
                ARTIFACT_REGISTRY_SCHEMA_VERSION,
            )

            cur.execute(CREATE_ARTIFACTS_TABLE)
            cur.execute(CREATE_LINEAGE_TABLE)
            cur.execute(CREATE_INDEXES)

            cur.execute(
                """
                INSERT INTO artifact_registry_schema_version (id, version)
                VALUES (1, %s)
                ON CONFLICT (id) DO UPDATE SET version = %s, applied_at = NOW()
            """,
                (ARTIFACT_REGISTRY_SCHEMA_VERSION, ARTIFACT_REGISTRY_SCHEMA_VERSION),
            )

            conn.commit()
            logger.info(
                "Artifact schema updated to version %s",
                ARTIFACT_REGISTRY_SCHEMA_VERSION,
            )
            return True

    except Exception as exc:
        conn.rollback()
        logger.error("Failed to ensure artifact schema: %s", exc)
        raise
