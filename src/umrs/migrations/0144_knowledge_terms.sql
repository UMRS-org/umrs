-- 0144: term registry, closure, admitted sets, vocabulary gaps and write
-- batches (ADR-081 D-2/D-5/D-9, W2a; slice KN-S7 of decisions-2026-11-24).
--
-- Additive and idempotent. Number provisional: migration numbers are taken
-- in merge-train order at land time (decisions-2026-11-24 §3 item 11).
--
-- knowledge_terms holds one immutable row per CURIE (a change is a
-- deprecation + replaced_by). knowledge_term_closure is the reflexive
-- minimum-depth subsumption closure over classes and relations, rebuilt in
-- the registering transaction. knowledge_ontology_terms materializes each
-- ontology record's admitted set (own terms + transitive imports). The
-- bundled std@1 / r4-core@2 rows come from package data
-- (umrs/knowledge/ontologies/*.yaml) through the generated seed migration
-- that follows this one; nothing is fetched.
--
-- knowledge_ontologies already has spec (subtype lattice, extraction
-- fields) and source_sha256 since 0139; this adds the ADR-081 D-2 columns
-- it still lacks. knowledge_graphs gains ontology_mode (warn|enforce,
-- default warn: every existing graph keeps accepting every writer),
-- graph_version (bumped per write batch) and graph_ns (id salt, D-6).
--
-- No pg_trgm here (an install burden on some fronts, ADR-093 MD-4).
--
-- Down (manual): DROP TABLE knowledge_write_batches, knowledge_vocab_gaps,
-- knowledge_ontology_terms, knowledge_term_closure, knowledge_terms (after
-- 0145's columns are dropped); ALTER TABLE ... DROP COLUMN for the rest.

CREATE TABLE IF NOT EXISTS knowledge_terms (
    term_id        TEXT PRIMARY KEY,
    iri            TEXT NOT NULL UNIQUE,
    kind           TEXT NOT NULL CHECK (kind IN ('class', 'relation', 'attribute')),
    label          TEXT NOT NULL,
    synonyms       JSONB NOT NULL DEFAULT '[]'::jsonb,
    definition     TEXT NOT NULL DEFAULT '',
    parents        TEXT[] NOT NULL DEFAULT '{}',
    bucket         TEXT CHECK (bucket IN ('class', 'function', 'file', 'module',
                                          'concept', 'entity', 'relationship')),
    domain_classes TEXT[] NOT NULL DEFAULT '{}',
    range_classes  TEXT[] NOT NULL DEFAULT '{}',
    datatype       TEXT CHECK (datatype IS NULL OR datatype IN (
                       'number', 'integer', 'string', 'boolean', 'datetime',
                       'date', 'interval', 'json', 'node')),
    unit           TEXT,
    transitive     BOOLEAN NOT NULL DEFAULT false,
    functional     BOOLEAN NOT NULL DEFAULT false,
    inverse_of     TEXT,
    aliases        TEXT[] NOT NULL DEFAULT '{}',
    mappings       JSONB NOT NULL DEFAULT '{}'::jsonb,
    required_on    TEXT[] NOT NULL DEFAULT '{}',
    origin         TEXT NOT NULL DEFAULT 'minted'
                   CHECK (origin IN ('standard', 'imported', 'minted')),
    verified       BOOLEAN NOT NULL DEFAULT true,
    source         TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'active'
                   CHECK (status IN ('active', 'deprecated')),
    replaced_by    TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT knowledge_terms_class_bucket CHECK (kind <> 'class' OR bucket IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS knowledge_terms_kind ON knowledge_terms (kind, status);
CREATE INDEX IF NOT EXISTS knowledge_terms_aliases ON knowledge_terms USING GIN (aliases);
CREATE INDEX IF NOT EXISTS knowledge_terms_parents ON knowledge_terms USING GIN (parents);

CREATE TABLE IF NOT EXISTS knowledge_term_closure (
    ancestor   TEXT NOT NULL REFERENCES knowledge_terms (term_id) ON DELETE CASCADE,
    descendant TEXT NOT NULL REFERENCES knowledge_terms (term_id) ON DELETE CASCADE,
    depth      SMALLINT NOT NULL CHECK (depth >= 0),
    PRIMARY KEY (ancestor, descendant)
);
CREATE INDEX IF NOT EXISTS knowledge_term_closure_desc ON knowledge_term_closure (descendant);

ALTER TABLE knowledge_ontologies
    ADD COLUMN IF NOT EXISTS iri_base TEXT NOT NULL DEFAULT '';
ALTER TABLE knowledge_ontologies
    ADD COLUMN IF NOT EXISTS prefixes JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE knowledge_ontologies
    ADD COLUMN IF NOT EXISTS imports JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE knowledge_ontologies
    ADD COLUMN IF NOT EXISTS profile JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS knowledge_ontology_terms (
    record_id TEXT NOT NULL REFERENCES knowledge_ontologies (record_id) ON DELETE CASCADE,
    term_id   TEXT NOT NULL REFERENCES knowledge_terms (term_id),
    via       TEXT NOT NULL DEFAULT 'own',
    PRIMARY KEY (record_id, term_id)
);

ALTER TABLE knowledge_graphs
    ADD COLUMN IF NOT EXISTS ontology_mode TEXT NOT NULL DEFAULT 'warn';
ALTER TABLE knowledge_graphs
    ADD COLUMN IF NOT EXISTS graph_version BIGINT NOT NULL DEFAULT 0;
ALTER TABLE knowledge_graphs
    ADD COLUMN IF NOT EXISTS graph_ns TEXT NOT NULL DEFAULT '';
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint
                 WHERE conname = 'knowledge_graphs_ontology_mode_check') THEN
    ALTER TABLE knowledge_graphs ADD CONSTRAINT knowledge_graphs_ontology_mode_check
      CHECK (ontology_mode IN ('warn', 'enforce'));
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS knowledge_vocab_gaps (
    graph_id   TEXT NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN ('class', 'relation', 'attribute')),
    value      TEXT NOT NULL,
    writer     TEXT NOT NULL DEFAULT '',
    n          BIGINT NOT NULL DEFAULT 1,
    sample_id  TEXT NOT NULL DEFAULT '',
    first_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (graph_id, kind, value)
);

-- Write audit; knowledge_events cannot hold it (artifact_id NOT NULL FK and
-- an event_type CHECK). origin_actor is the closed actor vocabulary of
-- umrs.knowledge.actors plus 'system' for server-internal writers.
CREATE TABLE IF NOT EXISTS knowledge_write_batches (
    batch_id     TEXT PRIMARY KEY,
    graph_id     TEXT NOT NULL,
    project_id   TEXT NOT NULL DEFAULT '',
    writer       TEXT NOT NULL,
    origin_actor TEXT NOT NULL CHECK (origin_actor IN ('user', 'agent', 'system')),
    ontology_ref TEXT NOT NULL DEFAULT '',
    mode         TEXT NOT NULL DEFAULT 'warn' CHECK (mode IN ('warn', 'enforce')),
    counts       JSONB NOT NULL DEFAULT '{}'::jsonb,
    violations   JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS knowledge_write_batches_graph
    ON knowledge_write_batches (graph_id, created_at DESC);
