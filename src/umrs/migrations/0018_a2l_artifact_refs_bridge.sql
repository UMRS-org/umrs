-- Migration: a2l_artifact_refs bridge table
-- ADR-017: A2L ↔ Knowledge Graph Identity Bridge
-- Adds best-effort cross-reference between UMRS artifacts and A2L identity space.

CREATE TABLE IF NOT EXISTS a2l_artifact_refs (
    id            SERIAL PRIMARY KEY,
    artifact_id   UUID        NOT NULL REFERENCES knowledge_artifacts(id) ON DELETE CASCADE,
    a2l_hash      TEXT        NOT NULL,  -- sha256 of artifact content
    a2l_id        TEXT        NOT NULL,  -- A2L deterministic ID
    synced_at     TIMESTAMP   NOT NULL DEFAULT NOW(),
    source        TEXT        NOT NULL DEFAULT 'auto',
    CONSTRAINT a2l_artifact_refs_unique_hash_artifact
        UNIQUE (artifact_id, a2l_hash)
);

CREATE INDEX IF NOT EXISTS idx_a2l_artifact_refs_hash
    ON a2l_artifact_refs (a2l_hash);

CREATE INDEX IF NOT EXISTS idx_a2l_artifact_refs_artifact
    ON a2l_artifact_refs (artifact_id);

COMMENT ON TABLE a2l_artifact_refs IS
    'A2L↔knowledge graph identity bridge (ADR-017). Best-effort cross-reference.';
