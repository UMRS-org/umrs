-- 0100: link the frontend-venue registry row to its durable job (ADR-046 A2).
--
-- The session runtime lifecycle now rides the `jobs` table as job_type
-- `frontend_session` (one attempt = the sandbox's life; venue = claiming
-- site). The 0099 row stays the session-keyed projection the TUI reads
-- (state, lease/egress binding, worker-reported runtime) and carries the
-- job id so status, stop and cancel meet on the same identity.

ALTER TABLE companion_venue_sessions
    ADD COLUMN IF NOT EXISTS job_id TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_companion_venue_sessions_job
    ON companion_venue_sessions (job_id) WHERE job_id <> '';
