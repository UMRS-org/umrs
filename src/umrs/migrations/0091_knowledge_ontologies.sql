-- 0091: knowledge_ontologies registry (ADR-049 Leg A / G-P1-7).
--
-- Versioned, shareable ontology records for knowledge graphs. Replaces the
-- inline-in-provenance path as the PREFERRED resolution source; inline stays
-- as the legacy path. Retirement is a tombstone (retired_at), never a
-- DELETE, and uniqueness of a live (ontology_id, version) pair is enforced
-- by a partial index over untombstoned rows so a retired version may be
-- re-registered deliberately.

CREATE TABLE IF NOT EXISTS knowledge_ontologies (
    record_id            TEXT PRIMARY KEY,
    ontology_id          TEXT NOT NULL,
    version              TEXT NOT NULL,
    entity_types         JSONB NOT NULL,
    relation_types       JSONB NOT NULL,
    transitive_relations JSONB NOT NULL DEFAULT '[]'::jsonb,
    description          TEXT NOT NULL DEFAULT '',
    owner_id             TEXT NOT NULL,
    project_id           TEXT,
    status               TEXT NOT NULL DEFAULT 'active',
    retired_at           TIMESTAMPTZ,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT knowledge_ontologies_status_check
        CHECK (status IN ('active', 'retired'))
);

-- One LIVE record per (ontology_id, version); retired rows are free to be
-- re-registered under the same coordinates.
CREATE UNIQUE INDEX IF NOT EXISTS knowledge_ontologies_live_coordinate
    ON knowledge_ontologies (ontology_id, version)
    WHERE retired_at IS NULL;

CREATE INDEX IF NOT EXISTS knowledge_ontologies_project
    ON knowledge_ontologies (project_id, status);

COMMENT ON TABLE knowledge_ontologies IS
    'Versioned ontology registry for knowledge graphs (ADR-049 Leg A). Retirement is a tombstone (retired_at), never a DELETE.';
