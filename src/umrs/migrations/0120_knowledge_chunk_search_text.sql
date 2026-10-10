-- 0120: knowledge_nodes.search_text — the lexical lanes' chunk text
-- (ADR-093 D-6 / C6; amends ADR-081 D-1: a writer-maintained column plus an
-- expression index, not a STORED generated column).
--
-- The column holds the full body of a chunk node (md5 = content.chunk_md5).
-- The embed job's chunk writer fills it; existing rows are filled without a
-- re-embed by `python -m umrs.knowledge.retrieval.backfill --project P`, which
-- writes only md5-verified text. Rows that are not chunks stay NULL.
--
-- Catalog-only change: a nullable column with no default rewrites no rows.
-- Both indexes are partial (search_text IS NOT NULL), so they hold chunk rows
-- only. They are built inside this migration's transaction, which takes
-- seconds on the stand-alone stack; a large production table needs a
-- maintenance window or an out-of-band build (ADR-093 MD-4).
--
-- pg_trgm is optional. Where the role may not create it, the migration
-- raises a NOTICE and carries on; the identifier lane then scans, and only
-- while a scope holds 50,000 chunk rows or fewer.

ALTER TABLE knowledge_nodes ADD COLUMN IF NOT EXISTS search_text TEXT;

COMMENT ON COLUMN knowledge_nodes.search_text IS
    'ADR-093 D-6: full chunk text for the lexical lanes (md5 = content.chunk_md5); writer-maintained, NULL for non-chunk rows.';

-- lex lane: IDF-weighted OR full-text search. The query side must use the
-- same expression, to_tsvector('english', search_text), to reach this index.
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_search_text_fts
    ON knowledge_nodes USING gin (to_tsvector('english'::regconfig, search_text))
    WHERE search_text IS NOT NULL;

-- ident lane: escaped ILIKE on identifier tokens (IL-6, TiO2, BRCA1).
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '0120: pg_trgm not created (% %); the ident lane scans only scopes of 50,000 chunk rows or fewer',
        SQLSTATE, SQLERRM;
END
$$;

DO $$
DECLARE
    trgm_schema TEXT;
BEGIN
    SELECT n.nspname INTO trgm_schema
      FROM pg_extension e
      JOIN pg_namespace n ON n.oid = e.extnamespace
     WHERE e.extname = 'pg_trgm';
    IF trgm_schema IS NULL THEN
        RAISE NOTICE '0120: pg_trgm absent; idx_knowledge_nodes_search_text_trgm not built';
        RETURN;
    END IF;
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_search_text_trgm '
        'ON knowledge_nodes USING gin (search_text %I.gin_trgm_ops) '
        'WHERE search_text IS NOT NULL',
        trgm_schema
    );
END
$$;
