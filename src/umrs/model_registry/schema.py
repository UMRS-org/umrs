"""
Database schema for Model Registry.

Creates and manages the model_catalog_entries table.
See DEVELOPMENT_OBJECTIVES.md Section 1.9.1 for design details.
"""

from __future__ import annotations

import logging
from typing import Optional

import psycopg2

logger = logging.getLogger(__name__)

# Schema version - increment when schema changes
MODEL_REGISTRY_SCHEMA_VERSION = 1

# SQL for creating the model_catalog_entries table (U-801)
CREATE_MODEL_CATALOG_TABLE = """
CREATE TABLE IF NOT EXISTS model_catalog_entries (
    model_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    scope VARCHAR(32) NOT NULL CHECK (scope IN ('system', 'project', 'user')),
    owner_id VARCHAR(128),
    project_id VARCHAR(128),
    storage_ref TEXT NOT NULL,
    model_type VARCHAR(32) NOT NULL CHECK (model_type IN ('llm', 'embedding')),
    huggingface_id VARCHAR(256),
    description TEXT,
    serving_profile JSONB DEFAULT '{}',
    status VARCHAR(32) DEFAULT 'available' CHECK (status IN ('staging', 'available', 'deprecated')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique name per scope (system), per project, or per user
    CONSTRAINT unique_system_model UNIQUE (scope, name) WHERE scope = 'system',
    CONSTRAINT unique_project_model UNIQUE (scope, project_id, name) WHERE scope = 'project',
    CONSTRAINT unique_user_model UNIQUE (scope, owner_id, name) WHERE scope = 'user'
);

-- Comments for documentation
COMMENT ON TABLE model_catalog_entries IS 'Model registry - metadata + ACL for model weights';
COMMENT ON COLUMN model_catalog_entries.model_id IS 'Unique model identifier (mdl-xxxxxxxxxxxx)';
COMMENT ON COLUMN model_catalog_entries.name IS 'Human-readable model name (unique per scope)';
COMMENT ON COLUMN model_catalog_entries.scope IS 'Visibility scope: system (all users), project (members), user (owner)';
COMMENT ON COLUMN model_catalog_entries.owner_id IS 'Owner user ID (required for user scope, NULL for system/project)';
COMMENT ON COLUMN model_catalog_entries.project_id IS 'Project ID (required for project scope, NULL for system/user)';
COMMENT ON COLUMN model_catalog_entries.storage_ref IS 'Filesystem path to model weights';
COMMENT ON COLUMN model_catalog_entries.model_type IS 'Model type: llm (vLLM) or embedding (SentenceTransformers)';
COMMENT ON COLUMN model_catalog_entries.huggingface_id IS 'Source HuggingFace model ID (e.g., Qwen/Qwen2.5-7B-Instruct)';
COMMENT ON COLUMN model_catalog_entries.serving_profile IS 'vLLM/LiteLLM configuration (JSON)';
COMMENT ON COLUMN model_catalog_entries.status IS 'Availability: staging (downloading), available (ready), deprecated (marked for removal)';
"""

# Partial unique indexes for scope-specific uniqueness
# PostgreSQL requires separate CREATE INDEX statements for partial indexes
CREATE_INDEXES = """
-- Index for fast scope filtering
CREATE INDEX IF NOT EXISTS idx_models_scope
    ON model_catalog_entries(scope);

-- Index for project model lookup
CREATE INDEX IF NOT EXISTS idx_models_project
    ON model_catalog_entries(project_id)
    WHERE project_id IS NOT NULL;

-- Index for user model lookup
CREATE INDEX IF NOT EXISTS idx_models_owner
    ON model_catalog_entries(owner_id)
    WHERE owner_id IS NOT NULL;

-- Index for model type filtering
CREATE INDEX IF NOT EXISTS idx_models_type
    ON model_catalog_entries(model_type);

-- Index for status filtering
CREATE INDEX IF NOT EXISTS idx_models_status
    ON model_catalog_entries(status);

-- Unique index for system models (name must be unique)
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_system_model
    ON model_catalog_entries(name)
    WHERE scope = 'system';

-- Unique index for project models (name unique per project)
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_project_model
    ON model_catalog_entries(project_id, name)
    WHERE scope = 'project';

-- Unique index for user models (name unique per user)
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_user_model
    ON model_catalog_entries(owner_id, name)
    WHERE scope = 'user';
"""

# Schema version tracking table
CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS model_registry_schema_version (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    version INTEGER NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def ensure_schema(conn: psycopg2.extensions.connection) -> int:
    """
    Ensure model registry schema exists and is up to date.

    Args:
        conn: PostgreSQL connection

    Returns:
        Current schema version after any migrations
    """
    with conn.cursor() as cur:
        # Create schema version table if not exists
        cur.execute(CREATE_SCHEMA_VERSION_TABLE)

        # Check current version
        cur.execute(
            "SELECT version FROM model_registry_schema_version WHERE id = 1"
        )
        row = cur.fetchone()

        if row is None:
            # Fresh install - create all tables
            logger.info("Creating model registry schema (version %d)", MODEL_REGISTRY_SCHEMA_VERSION)
            _create_tables(cur)
            cur.execute(
                """
                INSERT INTO model_registry_schema_version (id, version, updated_at)
                VALUES (1, %s, NOW())
                """,
                (MODEL_REGISTRY_SCHEMA_VERSION,),
            )
            conn.commit()
            return MODEL_REGISTRY_SCHEMA_VERSION

        current_version = row[0]
        if current_version < MODEL_REGISTRY_SCHEMA_VERSION:
            # Run migrations
            logger.info(
                "Migrating model registry schema from version %d to %d",
                current_version,
                MODEL_REGISTRY_SCHEMA_VERSION,
            )
            _migrate_schema(cur, current_version, MODEL_REGISTRY_SCHEMA_VERSION)
            cur.execute(
                """
                UPDATE model_registry_schema_version
                SET version = %s, updated_at = NOW()
                WHERE id = 1
                """,
                (MODEL_REGISTRY_SCHEMA_VERSION,),
            )
            conn.commit()
            return MODEL_REGISTRY_SCHEMA_VERSION

        logger.debug("Model registry schema is up to date (version %d)", current_version)
        return current_version


def _create_tables(cur: psycopg2.extensions.cursor) -> None:
    """Create all model registry tables."""
    # Create main table (without inline partial unique constraints)
    # PostgreSQL doesn't support WHERE in table-level constraints
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS model_catalog_entries (
        model_id VARCHAR(64) PRIMARY KEY,
        name VARCHAR(256) NOT NULL,
        scope VARCHAR(32) NOT NULL CHECK (scope IN ('system', 'project', 'user')),
        owner_id VARCHAR(128),
        project_id VARCHAR(128),
        storage_ref TEXT NOT NULL,
        model_type VARCHAR(32) NOT NULL CHECK (model_type IN ('llm', 'embedding')),
        huggingface_id VARCHAR(256),
        description TEXT,
        serving_profile JSONB DEFAULT '{}',
        status VARCHAR(32) DEFAULT 'available' CHECK (status IN ('staging', 'available', 'deprecated')),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """
    cur.execute(create_table_sql)

    # Add comments
    cur.execute(
        "COMMENT ON TABLE model_catalog_entries IS "
        "'Model registry - metadata + ACL for model weights'"
    )

    # Create indexes
    cur.execute(CREATE_INDEXES)


def _migrate_schema(
    cur: psycopg2.extensions.cursor,
    from_version: int,
    to_version: int,
) -> None:
    """
    Run schema migrations from one version to another.

    Migrations are cumulative - each version builds on the previous.
    """
    # Currently no migrations needed (version 1)
    # Add migration code here as schema evolves:
    #
    # if from_version < 2:
    #     cur.execute("ALTER TABLE model_catalog_entries ADD COLUMN new_field TEXT")
    #
    # if from_version < 3:
    #     cur.execute("CREATE INDEX ...")
    pass


def drop_schema(conn: psycopg2.extensions.connection) -> None:
    """
    Drop all model registry tables. USE WITH CAUTION.

    This is primarily for testing purposes.
    """
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS model_catalog_entries CASCADE")
        cur.execute("DROP TABLE IF EXISTS model_registry_schema_version CASCADE")
        conn.commit()
    logger.warning("Model registry schema dropped")
