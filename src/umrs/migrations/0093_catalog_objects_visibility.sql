-- 0093: visibility tier on the data-catalog plane (ADR-050 wave 2, R-02).
--
-- 0092 added the tier column to knowledge_nodes and sat_knowledge_objects
-- only, while CatalogMetaStore.search (data_catalog/pg_store.py) already
-- filtered on catalog_objects.visibility — on a real Postgres the catalog
-- lane failed with `column "visibility" does not exist`. This migration
-- brings catalog_objects onto the same three-family rule:
--   project — owned by project, visible only to the owning project's callers
--   system  — platform/system-wide rows; visible to every caller
--   ambient — read-gated (hidden from all read lanes; kept, never deleted)
--
-- Default is 'project' for every existing row: this migration changes NO
-- visibility of current data. Marking is an operator action
-- (tools/mark_ambient_corpus.py) outside the migration's scope.

ALTER TABLE catalog_objects
    ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'project';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'catalog_objects_visibility_check'
    ) THEN
        ALTER TABLE catalog_objects
            ADD CONSTRAINT catalog_objects_visibility_check
            CHECK (visibility IN ('project', 'system', 'ambient'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS catalog_objects_visibility_idx
    ON catalog_objects (project_id, visibility);

COMMENT ON COLUMN catalog_objects.visibility IS
    'Visibility tier (ADR-050): project (owner only) | system (every caller) | ambient (read-gated, kept, never deleted).';
