-- 0067: converge knowledge_artifacts on the canonical 0055 shape.
--
-- Schema drift repair: environments migrated before 0055 was backfilled got
-- the minimal 0064 shim shape (source_uri, no size_bytes/mime_type), while
-- 0055 defines the canonical read/write surface used by knowledge/store.py
-- (source_ref, size_bytes, mime_type). CREATE TABLE IF NOT EXISTS semantics
-- mean whichever ran first won, so both shapes exist in the wild. This
-- migration is shape-tolerant: it only ADDS missing columns and backfills
-- data, never drops anything.
--
-- Canonical columns (0055): source_ref, size_bytes, mime_type (+ status check
-- domains differ between the two files, so no CHECK is altered here).

ALTER TABLE knowledge_artifacts
    ADD COLUMN IF NOT EXISTS source_ref TEXT DEFAULT '';

ALTER TABLE knowledge_artifacts
    ADD COLUMN IF NOT EXISTS size_bytes BIGINT DEFAULT 0;

ALTER TABLE knowledge_artifacts
    ADD COLUMN IF NOT EXISTS mime_type TEXT DEFAULT '';

-- Backfill source_ref from the 0064-era source_uri column when present.
-- Wrapped in a DO block so the statement is idempotent and safe on
-- environments that only ever had the 0055 shape (source_uri absent).
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'knowledge_artifacts' AND column_name = 'source_uri'
    ) THEN
        EXECUTE 'UPDATE knowledge_artifacts
                 SET source_ref = source_uri
                 WHERE (source_ref IS NULL OR source_ref = '''')
                   AND source_uri IS NOT NULL AND source_uri <> ''''';
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_artifact_hash
    ON knowledge_artifacts(content_hash) WHERE status = 'active';
