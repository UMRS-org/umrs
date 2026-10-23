-- 0126: llm_usage_log.venue / session_id / origin — remote-session usage
-- (ADR-099 D-4, RS-1 lane Q; remote-session platform plan 2026-10-22 C-3).
--
-- The inference proxy (umrs-infer) now writes one llm_usage_log row per
-- relayed call, so the per-user and per-project token quotas (ADR-099 D-5)
-- read the ledger UMRS already has instead of a second usage table:
--
--   origin = 'infer_proxy'  → a remote session's call through umrs-infer
--   origin = ''             → UMRS's own /umrs/llm proxy (pre-0126 rows too)
--   venue  = '<site.name>'  → the lease's issuing site (0124); sites share
--                             umrs_db and accounts are per site (0113)
--   session_id              → the remote session (empty for /umrs/llm rows)
--
-- Non-destructive: existing rows keep the empty defaults. The two partial
-- indexes serve the quota SUM (venue + user or project, today's rows).

ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS venue      TEXT NOT NULL DEFAULT '';
ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS session_id TEXT NOT NULL DEFAULT '';
ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS origin     TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_llm_usage_venue_user_time
    ON llm_usage_log (venue, user_id, event_time DESC) WHERE origin = 'infer_proxy';

CREATE INDEX IF NOT EXISTS idx_llm_usage_venue_project_time
    ON llm_usage_log (venue, project_id, event_time DESC) WHERE origin = 'infer_proxy';

COMMENT ON COLUMN llm_usage_log.origin IS
    'Writer of the row: infer_proxy (remote-session inference proxy, ADR-099) '
    'or empty (the UMRS /umrs/llm proxy).';

-- umrs-infer reads the usage of the day and writes its rows (0086 role pattern;
-- the role exists only where a split-role deployment created it).
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'umrs_infer') THEN
        EXECUTE 'GRANT SELECT, INSERT ON llm_usage_log TO umrs_infer';
        EXECUTE 'GRANT USAGE ON SEQUENCE llm_usage_log_id_seq TO umrs_infer';
    END IF;
END
$$;
