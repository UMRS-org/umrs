-- 0092: visibility tiers on knowledge planes (ADR-050 D-1/D-2).
--
-- Three families, one tier column per plane:
--   project — owned by project, visible only to the owning project's callers
--   system  — platform/system-wide knowledge; visible to every caller
--   ambient — read-gated (hidden from all read lanes; kept, never deleted)
--
-- Default is 'project' for every existing row: this migration changes NO
-- visibility of current data. Ambient marking is an operator action
-- (tools/mark_ambient_corpus.py) outside the migration's scope.

ALTER TABLE knowledge_nodes
    ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'project';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'knowledge_nodes_visibility_check'
    ) THEN
        ALTER TABLE knowledge_nodes
            ADD CONSTRAINT knowledge_nodes_visibility_check
            CHECK (visibility IN ('project', 'system', 'ambient'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS knowledge_nodes_visibility_idx
    ON knowledge_nodes (project_id, visibility);

ALTER TABLE sat_knowledge_objects
    ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'project';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'sat_knowledge_objects_visibility_check'
    ) THEN
        ALTER TABLE sat_knowledge_objects
            ADD CONSTRAINT sat_knowledge_objects_visibility_check
            CHECK (visibility IN ('project', 'system', 'ambient'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS sat_knowledge_objects_visibility_idx
    ON sat_knowledge_objects (project_id, visibility);

COMMENT ON COLUMN knowledge_nodes.visibility IS
    'Visibility tier (ADR-050): project (owner only) | system (every caller) | ambient (read-gated, kept, never deleted).';
COMMENT ON COLUMN sat_knowledge_objects.visibility IS
    'Visibility tier (ADR-050): project (owner only) | system (every caller) | ambient (read-gated, kept, never deleted).';
