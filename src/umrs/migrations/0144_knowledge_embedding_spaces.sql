-- 0144 (slice KN-S5): the embedding-space registry (ADR-083 D-1, decision KN-3).
--
-- One row per encoder space. A vector in knowledge_node_embeddings belongs to
-- the space whose model_name or legacy_names hold its model_name (a bare name
-- without '/' also matches any path ending in it); the embed writer refuses a
-- model no ACTIVE row owns (umrs.knowledge.embedding_space_guard).
--
-- Seeded with exactly one row: the bundled e5-base-v2 text space under the
-- names sites stage it by, so a stand-alone install has an active space with
-- no admin step. Further spaces come from the site file's
-- knowledge.embedding_spaces (seeded on first write, or by
-- `umrs-server admin embedding-space seed-site`) and from the admin CLI.
-- The seed values equal umrs.knowledge.embedding_spaces.BUNDLED_SPACE (a test
-- compares them).
--
-- space_id and model_name feed index DDL later (hnsw_<space_id>, slice KN-S6),
-- so both are CHECK-restricted to the patterns embedding_spaces.py enforces.
--
-- Number: provisional; migration numbers are assigned in merge-train order at
-- land time (decisions-2026-11-24 §3 item 11).
--
-- Down (manual): DROP TABLE IF EXISTS knowledge_embedding_spaces;

CREATE TABLE IF NOT EXISTS knowledge_embedding_spaces (
    space_id        TEXT PRIMARY KEY
                    CHECK (space_id ~ '^[a-z0-9][a-z0-9_]{0,47}$'),
    model_name      TEXT NOT NULL UNIQUE
                    CHECK (model_name ~ '^[A-Za-z0-9._/:@+-]{1,255}$'),
    model_ref       TEXT NOT NULL DEFAULT '',
    revision        TEXT NOT NULL DEFAULT '',
    dimension       INTEGER NOT NULL CHECK (dimension > 0 AND dimension <= 16000),
    modalities      TEXT[] NOT NULL DEFAULT ARRAY['text']::TEXT[]
                    CHECK (cardinality(modalities) > 0
                           AND modalities <@ ARRAY['text', 'image', 'video', 'audio', 'volume']::TEXT[]),
    metric          TEXT NOT NULL DEFAULT 'cosine' CHECK (metric IN ('cosine')),
    query_prefix    TEXT NOT NULL DEFAULT '',
    document_prefix TEXT NOT NULL DEFAULT '',
    locality        TEXT NOT NULL DEFAULT 'site' CHECK (locality IN ('site')),
    legacy_names    TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'retired')),
    source          TEXT NOT NULL DEFAULT 'admin' CHECK (source IN ('bundled', 'site', 'admin')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE knowledge_embedding_spaces IS
    'ADR-083 D-1 embedding-space registry: writers refuse a model_name no active row owns.';

INSERT INTO knowledge_embedding_spaces (
    space_id, model_name, model_ref, revision, dimension, modalities, metric,
    query_prefix, document_prefix, locality, legacy_names, status, source
) VALUES (
    'e5_base_v2',
    '/opt/umrs/models/e5-base-v2',
    'intfloat/e5-base-v2',
    'f52bf8ec8c7124536f0efb74aca902b2995e5bcd',
    768,
    ARRAY['text']::TEXT[],
    'cosine',
    'query: ',
    'passage: ',
    'site',
    ARRAY['intfloat/e5-base-v2', '/models/e5-base-v2', 'e5-base-v2']::TEXT[],
    'active',
    'bundled'
) ON CONFLICT DO NOTHING;
