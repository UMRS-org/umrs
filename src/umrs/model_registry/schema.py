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
MODEL_REGISTRY_SCHEMA_VERSION = 3

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

# SQL for creating the model_policies table (U-805)
CREATE_MODEL_POLICIES_TABLE = """
CREATE TABLE IF NOT EXISTS model_policies (
    policy_id VARCHAR(64) PRIMARY KEY,
    model_name VARCHAR(256) NOT NULL,

    -- Access control
    allowed_projects TEXT[] DEFAULT '{}',
    denied_projects TEXT[] DEFAULT '{}',

    -- Quota management
    max_concurrent_endpoints INTEGER DEFAULT NULL,
    max_tokens_per_day BIGINT DEFAULT NULL,
    priority INTEGER DEFAULT 0,

    -- Warm pool settings
    warm_pool_min INTEGER DEFAULT 0,
    warm_pool_max INTEGER DEFAULT 0,
    warm_pool_ttl_seconds INTEGER DEFAULT 3600,
    enable_warm_pool BOOLEAN DEFAULT FALSE,

    -- System model settings
    is_system_model BOOLEAN DEFAULT FALSE,
    default_serving_profile JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for policy lookup by model name
CREATE INDEX IF NOT EXISTS idx_policies_model_name
    ON model_policies(model_name);

-- Index for system model policies
CREATE INDEX IF NOT EXISTS idx_policies_system
    ON model_policies(is_system_model)
    WHERE is_system_model = TRUE;

COMMENT ON TABLE model_policies IS 'Model access policies and quotas (U-805)';
COMMENT ON COLUMN model_policies.model_name IS 'Model name this policy applies to';
COMMENT ON COLUMN model_policies.allowed_projects IS 'Projects allowed to use this model (empty = all)';
COMMENT ON COLUMN model_policies.denied_projects IS 'Projects denied access (takes precedence over allowed)';
COMMENT ON COLUMN model_policies.max_concurrent_endpoints IS 'Maximum concurrent inference endpoints';
COMMENT ON COLUMN model_policies.warm_pool_min IS 'Minimum warm endpoints to keep ready';
COMMENT ON COLUMN model_policies.warm_pool_max IS 'Maximum warm endpoints to maintain';
COMMENT ON COLUMN model_policies.is_system_model IS 'True if this is a system-level shared model';
"""


# SQL for creating the model_releases table (U-802)
CREATE_MODEL_RELEASES_TABLE = """
CREATE TABLE IF NOT EXISTS model_releases (
    release_id VARCHAR(64) PRIMARY KEY,
    model_id VARCHAR(64) NOT NULL REFERENCES model_catalog_entries(model_id) ON DELETE CASCADE,
    version VARCHAR(64) NOT NULL,
    storage_ref TEXT NOT NULL,

    -- Versioning metadata
    changelog TEXT,
    is_latest BOOLEAN DEFAULT FALSE,
    is_stable BOOLEAN DEFAULT FALSE,

    -- Download/source info
    source_type VARCHAR(32) DEFAULT 'manual' CHECK (source_type IN ('manual', 'huggingface', 's3', 'pvc')),
    source_ref TEXT,
    checksum VARCHAR(128),
    size_bytes BIGINT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at TIMESTAMPTZ,
    deprecated_at TIMESTAMPTZ,

    -- Unique version per model
    UNIQUE(model_id, version)
);

-- Index for fast version lookup
CREATE INDEX IF NOT EXISTS idx_releases_model_version
    ON model_releases(model_id, version);

-- Index for finding latest stable version
CREATE INDEX IF NOT EXISTS idx_releases_latest
    ON model_releases(model_id, is_latest)
    WHERE is_latest = TRUE;

COMMENT ON TABLE model_releases IS 'Model version tracking (U-802) - immutable releases of model weights';
COMMENT ON COLUMN model_releases.release_id IS 'Unique release identifier (rel-xxxxxxxxxxxx)';
COMMENT ON COLUMN model_releases.model_id IS 'Parent model in catalog';
COMMENT ON COLUMN model_releases.version IS 'Semantic version string (e.g., 1.0.0, v2-beta)';
COMMENT ON COLUMN model_releases.storage_ref IS 'Filesystem path to model weights for this version';
COMMENT ON COLUMN model_releases.is_latest IS 'True if this is the current latest version';
COMMENT ON COLUMN model_releases.is_stable IS 'True if this is a stable (non-beta) release';
COMMENT ON COLUMN model_releases.checksum IS 'SHA256 hash of model weights for verification';
"""


# SQL for creating the model_artifacts table (U-806)
CREATE_MODEL_ARTIFACTS_TABLE = """
CREATE TABLE IF NOT EXISTS model_artifacts (
    artifact_id VARCHAR(64) PRIMARY KEY,
    model_id VARCHAR(64) NOT NULL REFERENCES model_catalog_entries(model_id) ON DELETE CASCADE,
    release_id VARCHAR(64) REFERENCES model_releases(release_id) ON DELETE SET NULL,

    -- Artifact details
    artifact_type VARCHAR(32) NOT NULL CHECK (artifact_type IN (
        'weights', 'adapter', 'lora', 'tokenizer', 'config',
        'vocab', 'merges', 'gguf', 'safetensors', 'other'
    )),
    artifact_name VARCHAR(256) NOT NULL,
    artifact_path TEXT NOT NULL,

    -- Storage info
    storage_type VARCHAR(32) DEFAULT 'filesystem' CHECK (storage_type IN (
        'filesystem', 's3', 'pvc', 'http'
    )),
    size_bytes BIGINT,
    checksum VARCHAR(128),

    -- Metadata
    format VARCHAR(32),
    quantization VARCHAR(32),
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique artifact name per model/release
    UNIQUE(model_id, release_id, artifact_type, artifact_name)
);

-- Index for artifact lookup by model
CREATE INDEX IF NOT EXISTS idx_artifacts_model
    ON model_artifacts(model_id);

-- Index for artifact lookup by release
CREATE INDEX IF NOT EXISTS idx_artifacts_release
    ON model_artifacts(release_id)
    WHERE release_id IS NOT NULL;

-- Index for artifact type filtering
CREATE INDEX IF NOT EXISTS idx_artifacts_type
    ON model_artifacts(artifact_type);

COMMENT ON TABLE model_artifacts IS 'Model artifact tracking (U-806) - individual files per model/release';
COMMENT ON COLUMN model_artifacts.artifact_type IS 'Type: weights, adapter, lora, tokenizer, config, etc.';
COMMENT ON COLUMN model_artifacts.artifact_path IS 'Path to artifact (relative to model storage_ref)';
COMMENT ON COLUMN model_artifacts.format IS 'File format (safetensors, bin, gguf)';
COMMENT ON COLUMN model_artifacts.quantization IS 'Quantization type if applicable (fp16, int8, int4)';
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

        -- Added in schema version 2 (U-806, U-807)
        model_ref_json JSONB DEFAULT NULL,
        execution_authority_default VARCHAR(32) DEFAULT 'compose'
            CHECK (execution_authority_default IN ('slurm', 'k8s', 'compose')),

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
    cur.execute(
        "COMMENT ON COLUMN model_catalog_entries.model_ref_json IS "
        "'ModelRef abstraction (U-806): filesystem_path, pvc_ref, s3_ref, hf_repo_ref'"
    )
    cur.execute(
        "COMMENT ON COLUMN model_catalog_entries.execution_authority_default IS "
        "'Default execution authority for this model (U-807)'"
    )

    # Create indexes
    cur.execute(CREATE_INDEXES)

    # Create model_policies table (U-805)
    cur.execute(CREATE_MODEL_POLICIES_TABLE)

    # Create model_releases table (U-802)
    cur.execute(CREATE_MODEL_RELEASES_TABLE)

    # Create model_artifacts table (U-806)
    cur.execute(CREATE_MODEL_ARTIFACTS_TABLE)


def _migrate_schema(
    cur: psycopg2.extensions.cursor,
    from_version: int,
    to_version: int,
) -> None:
    """
    Run schema migrations from one version to another.

    Migrations are cumulative - each version builds on the previous.
    """
    if from_version < 2:
        # Version 2: Add model_policies table, model_ref_json, execution_authority_default
        logger.info("Running migration to version 2")

        # Add model_ref_json column (U-806)
        cur.execute("""
            ALTER TABLE model_catalog_entries
            ADD COLUMN IF NOT EXISTS model_ref_json JSONB DEFAULT NULL
        """)
        cur.execute(
            "COMMENT ON COLUMN model_catalog_entries.model_ref_json IS "
            "'ModelRef abstraction (U-806): filesystem_path, pvc_ref, s3_ref, hf_repo_ref'"
        )

        # Add execution_authority_default column (U-807)
        cur.execute("""
            ALTER TABLE model_catalog_entries
            ADD COLUMN IF NOT EXISTS execution_authority_default VARCHAR(32) DEFAULT 'compose'
        """)
        cur.execute("""
            ALTER TABLE model_catalog_entries
            ADD CONSTRAINT check_exec_authority
            CHECK (execution_authority_default IN ('slurm', 'k8s', 'compose'))
        """)
        cur.execute(
            "COMMENT ON COLUMN model_catalog_entries.execution_authority_default IS "
            "'Default execution authority for this model (U-807)'"
        )

        # Create model_policies table (U-805)
        cur.execute(CREATE_MODEL_POLICIES_TABLE)

        logger.info("Migration to version 2 complete")

    if from_version < 3:
        # Version 3: Add model_releases table (U-802), model_artifacts table (U-806)
        logger.info("Running migration to version 3")

        # Create model_releases table (U-802)
        cur.execute(CREATE_MODEL_RELEASES_TABLE)

        # Create model_artifacts table (U-806)
        cur.execute(CREATE_MODEL_ARTIFACTS_TABLE)

        logger.info("Migration to version 3 complete")


def drop_schema(conn: psycopg2.extensions.connection) -> None:
    """
    Drop all model registry tables. USE WITH CAUTION.

    This is primarily for testing purposes.
    """
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS model_artifacts CASCADE")
        cur.execute("DROP TABLE IF EXISTS model_releases CASCADE")
        cur.execute("DROP TABLE IF EXISTS model_policies CASCADE")
        cur.execute("DROP TABLE IF EXISTS model_catalog_entries CASCADE")
        cur.execute("DROP TABLE IF EXISTS model_registry_schema_version CASCADE")
        conn.commit()
    logger.warning("Model registry schema dropped")
