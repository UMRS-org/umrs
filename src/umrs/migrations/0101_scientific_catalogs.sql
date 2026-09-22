-- 0101: Scientific Catalog registry (gap row D-39, Phase 1).
--
-- The Scientific Catalog as a named, versioned, portable object: one row =
-- one catalog version bundling the ids of its members (graphs, skills, MCP
-- servers, embedding models) plus the ownership metadata (contributors,
-- licence, provenance). Versions are immutable rows — a new version is a
-- new row under (project_id, name, version); `status='retired'` tombstones,
-- never DELETE (platform rule). The `.kb` bundle (ADR-032) carries the same
-- ids for export (bundle v2, gap rows D-40/D-41).

CREATE TABLE IF NOT EXISTS scientific_catalogs (
    catalog_id          TEXT PRIMARY KEY,
    project_id          TEXT NOT NULL,
    name                TEXT NOT NULL,
    version             TEXT NOT NULL DEFAULT '0.1.0',
    description         TEXT NOT NULL DEFAULT '',
    ontology            TEXT NOT NULL DEFAULT '',
    member_graph_ids    JSONB NOT NULL DEFAULT '[]'::jsonb,
    skill_ids           JSONB NOT NULL DEFAULT '[]'::jsonb,
    mcp_server_ids      JSONB NOT NULL DEFAULT '[]'::jsonb,
    embedding_model_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    contributors        JSONB NOT NULL DEFAULT '[]'::jsonb,
    licence             TEXT NOT NULL DEFAULT '',
    provenance          JSONB NOT NULL DEFAULT '{}'::jsonb,
    owner_id            TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'active',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT scientific_catalogs_unique_version
        UNIQUE (project_id, name, version),
    CONSTRAINT scientific_catalogs_status_check
        CHECK (status IN ('active', 'retired'))
);

CREATE INDEX IF NOT EXISTS scientific_catalogs_project_idx
    ON scientific_catalogs (project_id, status);

COMMENT ON TABLE scientific_catalogs IS
    'Scientific Catalog registry (gap row D-39): named, versioned bundle of member graph/skill/MCP/embedding ids plus contributors, licence and provenance. Versions immutable; retire = tombstone.';
