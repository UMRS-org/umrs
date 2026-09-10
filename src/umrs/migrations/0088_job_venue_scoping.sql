-- 0088: jobs.venue — claim-side cluster scoping (WP-14/ADR-034 lineage).
--
-- All four UMRS servers (rccs-front/ai4s/bdrt/octopus) poll the shared
-- umrs_db.jobs. Without venue attribution at claim time, any server steals
-- another cluster's jobs (observed live 2026-09-09 6b-3). Column rules:
--
--   venue = '<site.name>'  → only this site's schedulers may claim the job
--   venue = ''             → shared (row created before venue attribution
--                            existed, or a deliberately global job)
--
-- The empty-string lane is the pre-release-compatible contract: existing rows
-- keep their older ANY-claim semantics; new rows carry the creating site's
-- name and honor it at claim time.

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS venue VARCHAR(32) NOT NULL DEFAULT '';

COMMENT ON COLUMN jobs.venue IS
    'Cluster/site this job belongs to (WP-14 claim scoping). Empty = shared/global; '
    'site-bound rows are claimed only by schedulers on the same site.name';

CREATE INDEX IF NOT EXISTS idx_jobs_venue_status_next_run
    ON jobs (venue, status, next_run_at);
