-- 0118: one-shot markers for the ADR-060 U-1 endpoint-registry copy.
--
-- The converge ran umrs.migrations.hub_endpoint_registry_copy at EVERY
-- apply: each front_<site>.endpoint_registry row whose job_id was missing
-- from the hub table was inserted again (ON CONFLICT DO NOTHING). Since the
-- U-1 flip the servers write only umrs_db and delete rows on remove() and
-- replace, so every re-copy would resurrect removed endpoints with stale
-- host/port/status, loaded at the next boot (the orphan/janitor class of
-- 2026-10-01). From here a source venue is copied at most once: the helper
-- claims the venue's marker in the same transaction as the rows and skips a
-- venue that already has one, without opening its front_* source.
--
-- Seals: the four umrs-front venues converged U-1 on 2026-10-03 with the
-- pre-marker helper. That helper most likely inserted nothing: psycopg2
-- reads JSONB metadata back as a dict and cannot adapt a dict as a
-- parameter, so any non-empty source raised on its first row and rolled
-- back (code reading + a local psycopg2 check; not confirmed live). Their
-- front_* rows are frozen since the flip, so a copy now would add rows
-- that are stale or that a server has removed since: the window is closed.
--
-- Correction (review, 2026-10-03): nothing rebuilds the registry from
-- durable jobs or from Slurm. The boot reconciler
-- (resume_unfinished_provisions) only finishes PROVISIONING/STARTING rows
-- already in the server's registry; 0117's header line "live endpoints
-- re-register through the boot reconciler" is wrong. The copy runs after
-- the post-migration restart, so its rows reach a server only at its next
-- boot, and the janitor's orphan pass (second pass, ~60 s after boot)
-- cancels backend jobs missing from the registry. The U-1 flip window was
-- therefore unprotected; the owner checks it read-only before the converge
-- that applies 0118 (plan doc, "U-1 follow-ups").
--
-- Re-arm one venue only deliberately, and only for live jobs that are
-- recorded nowhere but in its front_* table:
--   DELETE FROM hub_registry_copy_markers WHERE venue = '<venue>';
-- Re-arming re-inserts EVERY frozen row of that venue whose job_id is
-- missing from the hub table, including rows its server removed since the
-- flip (stale host/port/status). The rows reach the server at the restart
-- after the next converge's copy.

CREATE TABLE IF NOT EXISTS hub_registry_copy_markers (
    venue        VARCHAR(32) PRIMARY KEY,
    copied_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_rows  INTEGER,
    inserted     INTEGER,
    note         TEXT NOT NULL DEFAULT ''
);

INSERT INTO hub_registry_copy_markers (venue, note) VALUES
    ('rccs-front', 'sealed by 0118: pre-marker U-1 copy copied 0 rows (live, converge apply 37148979876, 2026-10-04: rccs/ai4s source DB names wrong, bdrt dict-metadata error, octopus 0 rows); frozen front_* rows are stale'),
    ('ai4s', 'sealed by 0118: pre-marker U-1 copy copied 0 rows (live, converge apply 37148979876, 2026-10-04: rccs/ai4s source DB names wrong, bdrt dict-metadata error, octopus 0 rows); frozen front_* rows are stale'),
    ('bdrt-front', 'sealed by 0118: pre-marker U-1 copy copied 0 rows (live, converge apply 37148979876, 2026-10-04: rccs/ai4s source DB names wrong, bdrt dict-metadata error, octopus 0 rows); frozen front_* rows are stale'),
    ('octopus-front', 'sealed by 0118: pre-marker U-1 copy copied 0 rows (live, converge apply 37148979876, 2026-10-04: rccs/ai4s source DB names wrong, bdrt dict-metadata error, octopus 0 rows); frozen front_* rows are stale')
ON CONFLICT (venue) DO NOTHING;

COMMENT ON TABLE hub_registry_copy_markers IS
    'ADR-060 U-1: one row per source venue whose front_* endpoint_registry was copied into the hub table (written in the copy transaction); a venue with a row is never copied again. NULL counts = sealed by 0118 (pre-marker copy).';
