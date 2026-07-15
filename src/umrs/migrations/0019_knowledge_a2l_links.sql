-- Migration: knowledge_a2l_links
-- ADR-017: A2L bridge — links knowledge artifacts to A2L identity.
-- Stores derived A2L references from UNDERSTAND/VERIFY/PRODUCED_BY steps.

CREATE TABLE IF NOT EXISTS knowledge_a2l_links (
    id            SERIAL PRIMARY KEY,
    knowledge_artifact_id UUID  NOT NULL REFERENCES knowledge_artifacts(id) ON DELETE CASCADE,
    a2l_artifact_id TEXT        NOT NULL,
    job_id        TEXT          NOT NULL,       -- source job that derived the link
    attempt_id    TEXT,                       -- specific attempt
    edge_kind     TEXT        NOT NULL DEFAULT 'derived',
    derived_at    TIMESTAMP   NOT NULL DEFAULT NOW(),
    link_status   TEXT        NOT NULL DEFAULT 'active',
    CONSTRAINT knowledge_a2l_links_unique_artifact_a2l
        UNIQUE (knowledge_artifact_id, a2l_artifact_id)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_a2l_links_artifact
    ON knowledge_a2l_links (knowledge_artifact_id);

CREATE INDEX IF NOT EXISTS idx_knowledge_a2l_links_a2l
    ON knowledge_a2l_links (a2l_artifact_id);

CREATE INDEX IF NOT EXISTS idx_knowledge_a2l_links_status
    ON knowledge_a2l_links (link_status);

COMMENT ON TABLE knowledge_a2l_links IS
    'Knowledge artifact ↔ A2L artifact links (ADR-017). Derive step stores here.';
