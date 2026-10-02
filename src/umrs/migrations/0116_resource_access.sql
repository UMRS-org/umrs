-- 0116: row-carried access scopes for shareable resources (ADR-080).
--
-- One guard, generalized from ADR-078 endpoints: a resource row carries an
-- `access` JSONB list of canonical scope strings (user:<id> | project:<id> |
-- group:<id> | system) — the list may name several users AND several groups;
-- a user may hold many group memberships (0097), any match admits. The list
-- governs USE (reads and runs); lifecycle writes stay with each family's
-- existing write ladder; `system` entries are admin-granted only. Absent or
-- '[]' keeps the historical posture exactly (fail-closed default).
--
-- Families covered here: knowledge graphs, scientific catalogs, personal
-- buckets. Endpoints already carry metadata.access (ADR-078); workflows
-- carry `access` on their catalog entries (JSON catalog, no DDL). RAG
-- corpora have no durable row of their own — the corpus is a project
-- namespace in the vector store — so corpus sharing gets a small registry
-- keyed by the corpus project.

ALTER TABLE knowledge_graphs
    ADD COLUMN IF NOT EXISTS access JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE scientific_catalogs
    ADD COLUMN IF NOT EXISTS access JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE user_buckets
    ADD COLUMN IF NOT EXISTS access JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE TABLE IF NOT EXISTS rag_corpus_access (
    project_id  TEXT PRIMARY KEY,
    access      JSONB NOT NULL DEFAULT '[]'::jsonb,
    set_by      TEXT NOT NULL DEFAULT '',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON COLUMN knowledge_graphs.access IS
    'ADR-080: scope list (user:/project:/group:/system) — who beyond the owning tier may READ this graph. [] = tier ladder only.';
COMMENT ON COLUMN scientific_catalogs.access IS
    'ADR-080: scope list — who beyond the owning project may READ this catalog version. [] = project only.';
COMMENT ON COLUMN user_buckets.access IS
    'ADR-080: scope list — who beyond the owner may read/write OBJECTS in this bucket (lifecycle stays owner-only). [] = owner only.';
COMMENT ON TABLE rag_corpus_access IS
    'ADR-080: per-corpus (project namespace) RAG read sharing — the corpus has no row of its own, this registry carries its access scope list.';
