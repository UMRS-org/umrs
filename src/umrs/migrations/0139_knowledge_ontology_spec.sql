-- 0139: full ontology spec + registered source on knowledge_ontologies (ADR-116 D-4).
--
-- 0091 stored only entity/relation/transitive types, so register() dropped
-- subtype_of and extraction_fields. spec keeps the full serialized ontology
-- (Ontology.to_dict(): subtype lattice, extraction fields, aliases,
-- descriptions, relation domain/range, class buckets, abstract classes,
-- prefixes). source_* keep the text the owner registered (LinkML YAML or the
-- ontology JSON) and its sha256. Rows written before 0139 keep spec = '{}' and
-- are read from the 0091 columns. Idempotent.

ALTER TABLE knowledge_ontologies
    ADD COLUMN IF NOT EXISTS spec JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE knowledge_ontologies
    ADD COLUMN IF NOT EXISTS source_format TEXT NOT NULL DEFAULT '';

ALTER TABLE knowledge_ontologies
    ADD COLUMN IF NOT EXISTS source_text TEXT;

ALTER TABLE knowledge_ontologies
    ADD COLUMN IF NOT EXISTS source_sha256 TEXT;

COMMENT ON COLUMN knowledge_ontologies.spec IS
    'Full serialized ontology (Ontology.to_dict()); empty for rows written before 0139 (ADR-116 D-4).';

-- Live coordinates are unique per scope (ADR-116 D-5): each project, and the
-- global scope (project_id NULL), has its own id@version namespace, so one
-- project's ontology never blocks, or reveals itself to, another project.
-- Lookups read the caller's project record first, then the global one. This
-- replaces the cross-project 0091 index; the new index exists before the old
-- one goes (every 0091-unique row set is unique here too), and both
-- statements are idempotent.
CREATE UNIQUE INDEX IF NOT EXISTS knowledge_ontologies_live_scoped_coordinate
    ON knowledge_ontologies (ontology_id, version, coalesce(project_id, ''))
    WHERE retired_at IS NULL;

DROP INDEX IF EXISTS knowledge_ontologies_live_coordinate;
