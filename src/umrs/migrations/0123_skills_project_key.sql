-- 0123: skills — a project-scope skill is unique per project (hardening plan
-- H-10.1 follow-up, 2026-10-18).
--
-- 0052 keyed the registry on UNIQUE (scope, owner_id, name). A user who
-- registered project skill "fit-peaks" in project A and then in project B
-- upserted A's row: project_id moved to B and A lost its skill. The loop's
-- D-27 proposals (owner 'compile-knowledge-skills', ON CONFLICT DO NOTHING)
-- were dropped for every project after the first one that compiled a skill
-- of that title, and a bundle import counted another project's stub as the
-- target project's own.
--
-- project_key is project_id for project-scope rows and '' for user and
-- system rows, so only project skills gain the project in their key
-- (a user skill is the user's across projects; a system skill is one per
-- owner). Writers name the key in ON CONFLICT
-- (scope, owner_id, project_key, name); umrs/skills/store.py and
-- jobs/knowledge_adapters.py do.
--
-- Existing rows already satisfy the new key (the old one was stricter), so
-- the unique index always builds. Re-running the file changes nothing: the
-- column and index use IF NOT EXISTS, and the DO block drops only a unique
-- constraint on exactly (scope, owner_id, name).
--
-- Everything runs inside the runner's transaction; the STORED column
-- rewrites the skills table once (a registry, small on every deployment).

ALTER TABLE skills
    ADD COLUMN IF NOT EXISTS project_key VARCHAR(128)
    GENERATED ALWAYS AS (
        CASE WHEN scope = 'project' THEN COALESCE(project_id, '') ELSE '' END
    ) STORED;

DO $$
DECLARE
    old_key TEXT;
BEGIN
    FOR old_key IN
        SELECT con.conname
          FROM pg_constraint con
         WHERE con.conrelid = 'skills'::regclass
           AND con.contype = 'u'
           AND (
               SELECT array_agg(att.attname::text ORDER BY att.attname::text)
                 FROM unnest(con.conkey) AS key(attnum)
                 JOIN pg_attribute att
                   ON att.attrelid = con.conrelid AND att.attnum = key.attnum
           ) = ARRAY['name', 'owner_id', 'scope']
    LOOP
        EXECUTE format('ALTER TABLE skills DROP CONSTRAINT %I', old_key);
    END LOOP;
END
$$;

CREATE UNIQUE INDEX IF NOT EXISTS uq_skills_scope_owner_project_name
    ON skills (scope, owner_id, project_key, name);

COMMENT ON COLUMN skills.project_key IS
    'Uniqueness key part: project_id for scope = project, else empty (0123). Writers use ON CONFLICT (scope, owner_id, project_key, name).';
