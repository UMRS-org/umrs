-- 0141 (slice S-D47a, DEC-D47, decisions record 2026-11-24 §2.5):
-- the hub endpoint registry is keyed (venue, job_id).
--
-- Number note for the integrator: 0141 was the next free number when the
-- slice was written; the file carries the slice id so the merge train can
-- renumber it without losing track.
--
-- Bug: 0117 keyed the shared endpoint_registry on job_id alone. Every
-- umrs-front server stamps its site name in `venue` and reads only its own
-- venue, but its upsert was `ON CONFLICT (job_id) DO UPDATE SET ...,
-- venue = EXCLUDED.venue`: a second venue that registered the same job_id
-- (Slurm ids are per cluster, so two clusters can mint the same one) took
-- the first venue's row over. The first venue then no longer saw its own
-- endpoint, and its janitor treated the live backend job as an orphan.
--
-- This migration swaps the primary key to (venue, job_id) under the same
-- constraint name `endpoint_registry_pkey`, which the servers name as their
-- upsert arbiter (registry_backend.PostgresBackend.save). That statement,
-- with its `WHERE endpoint_registry.venue = EXCLUDED.venue` guard, works on
-- BOTH key shapes, so the code deploys first (the converge restarts the
-- servers before it applies migrations) and the swap follows.
--
-- Safety: one transaction (the runner applies this file as one statement
-- batch; it never contains the keyword that switches the runner to
-- per-statement autocommit). Precheck: a duplicate (venue, job_id) pair, which only a
-- table without a primary key could hold, aborts with a named error and
-- changes nothing. Idempotent: an already (venue, job_id)-keyed table is
-- left as it is; a database without the table is skipped. Forward-only: the
-- converge refuses a rollback across a commit that adds a migration, and an
-- umrs build older than S-D47a would fail its `ON CONFLICT (job_id)` upsert
-- against the new key (DEC-D47: rollback is a flag, never a reverse
-- migration).

DO $$
DECLARE
    pk_name TEXT;
    pk_cols TEXT[];
    dupes   BIGINT;
BEGIN
    IF to_regclass('endpoint_registry') IS NULL THEN
        RETURN;
    END IF;

    ALTER TABLE endpoint_registry
        ADD COLUMN IF NOT EXISTS venue VARCHAR(32) NOT NULL DEFAULT '';
    -- Unscoped lookups by job_id keep an index once job_id leads no key.
    CREATE INDEX IF NOT EXISTS idx_registry_job_id ON endpoint_registry(job_id);

    SELECT c.conname, array_agg(a.attname::TEXT ORDER BY k.ord)
      INTO pk_name, pk_cols
      FROM pg_constraint c
      CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord)
      JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
     WHERE c.conrelid = 'endpoint_registry'::regclass AND c.contype = 'p'
     GROUP BY c.conname;

    IF pk_cols = ARRAY['venue', 'job_id'] THEN
        RETURN;
    END IF;

    SELECT count(*) INTO dupes FROM (
        SELECT 1 FROM endpoint_registry GROUP BY venue, job_id HAVING count(*) > 1
    ) d;
    IF dupes > 0 THEN
        RAISE EXCEPTION 'S-D47a precheck: % duplicate (venue, job_id) pair(s) in endpoint_registry; resolve them before the key swap', dupes;
    END IF;

    IF pk_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE endpoint_registry DROP CONSTRAINT %I', pk_name);
    END IF;
    ALTER TABLE endpoint_registry
        ADD CONSTRAINT endpoint_registry_pkey PRIMARY KEY (venue, job_id);
END $$;

COMMENT ON TABLE endpoint_registry IS
    'ADR-060 U-1: one hub endpoint registry in the shared umrs_db; PRIMARY KEY (venue, job_id) since 0141 (S-D47a): each site owns its rows, two venues may share a job_id. Per-cluster front_* copies are retired with their stacks.';
