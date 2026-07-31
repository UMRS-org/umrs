-- 0060: Durable stores for research-layer satellites, batch 2 (K1).
--
-- Postgres-backed persistence for embeddings (pgvector), the knowledge
-- graph API, the knowledge model (versioned objects + tombstones), and
-- the tenant registry. Timestamps are TEXT (ISO-8601) to match dataclass
-- contracts except tenants, which uses TIMESTAMPTZ (datetime contract).
--
-- Table names carry the sat_ prefix to stay collision-free next to the
-- knowledge/ store tables (knowledge_nodes, knowledge_edges, ...).

CREATE EXTENSION IF NOT EXISTS vector;

-- ── Embeddings (pgvector, dimension-agnostic) ────────────────────────────

CREATE TABLE IF NOT EXISTS sat_embeddings (
    embedding_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT '',
    object_id TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    vector vector,
    text TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sat_embeddings_project_created
    ON sat_embeddings (project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sat_embeddings_status
    ON sat_embeddings (status);
CREATE INDEX IF NOT EXISTS idx_sat_embeddings_object
    ON sat_embeddings (object_id);

-- ── Knowledge graph API ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sat_kg_nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_sat_kg_nodes_type
    ON sat_kg_nodes (node_type);

CREATE TABLE IF NOT EXISTS sat_kg_edges (
    edge_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL DEFAULT ''
);
-- One logical edge per (source, target, relation_type): dedupe target for
-- add_edge upsert semantics.
CREATE UNIQUE INDEX IF NOT EXISTS idx_sat_kg_edges_triple
    ON sat_kg_edges (source_id, target_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_sat_kg_edges_source
    ON sat_kg_edges (source_id);
CREATE INDEX IF NOT EXISTS idx_sat_kg_edges_target
    ON sat_kg_edges (target_id);
CREATE INDEX IF NOT EXISTS idx_sat_kg_edges_relation
    ON sat_kg_edges (relation_type);

-- ── Knowledge model (versioned objects + tombstones) ─────────────────────

CREATE TABLE IF NOT EXISTS sat_knowledge_objects (
    object_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT '',
    object_type TEXT NOT NULL DEFAULT 'paper',
    content JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    version INT NOT NULL DEFAULT 1,
    parent_version_id TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    is_immutable BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sat_km_objects_project_created
    ON sat_knowledge_objects (project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sat_km_objects_type
    ON sat_knowledge_objects (object_type);
CREATE INDEX IF NOT EXISTS idx_sat_km_objects_status
    ON sat_knowledge_objects (status);

CREATE TABLE IF NOT EXISTS sat_knowledge_object_versions (
    version_id TEXT PRIMARY KEY,
    object_id TEXT NOT NULL REFERENCES sat_knowledge_objects (object_id) ON DELETE CASCADE,
    version INT NOT NULL,
    content JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sat_km_versions_obj_ver
    ON sat_knowledge_object_versions (object_id, version);

CREATE TABLE IF NOT EXISTS sat_knowledge_tombstones (
    object_id TEXT PRIMARY KEY REFERENCES sat_knowledge_objects (object_id) ON DELETE CASCADE,
    object_type TEXT NOT NULL DEFAULT 'paper',
    deleted_at TEXT NOT NULL DEFAULT '',
    deleted_by TEXT NOT NULL DEFAULT '',
    deletion_reason TEXT NOT NULL DEFAULT ''
);

-- ── Tenant registry ──────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sat_tenants (
    tenant_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    project_name TEXT NOT NULL DEFAULT '',
    schema_name TEXT NOT NULL DEFAULT '',
    owner_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'provisioning',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    resource_limits JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sat_tenants_project
    ON sat_tenants (project_id);
CREATE INDEX IF NOT EXISTS idx_sat_tenants_owner
    ON sat_tenants (owner_id);
