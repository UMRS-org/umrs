-- 0084: companion session registry columns (venue + recorded state).
--
-- Plan: docs/development/session-venue-selection-and-unified-listing-plan-2026-09-09.md
-- Snapshots gain two projected columns so the session listing can filter by
-- venue and recorded state server-side. Both are additive with honest
-- defaults: pre-existing rows were mirrored from local companion clients,
-- so venue='local' is factual, and state='' means "no recorded transition"
-- (absent, never fabricated). 'archived' is never stored here — it is
-- computed from companion_session_archives (0079 tombstones).

ALTER TABLE companion_session_snapshots
    ADD COLUMN IF NOT EXISTS venue TEXT NOT NULL DEFAULT 'local';

ALTER TABLE companion_session_snapshots
    ADD COLUMN IF NOT EXISTS state TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_companion_session_snapshots_venue
    ON companion_session_snapshots(user_id, venue, updated_at DESC);

COMMENT ON COLUMN companion_session_snapshots.venue IS
    'Session execution venue: local | frontend | cluster:<name>. Recorded by '
    'the mirroring client at create/resume time; ''local'' is the honest '
    'default for pre-existing rows (they were mirrored from local clients).';

COMMENT ON COLUMN companion_session_snapshots.state IS
    'Recorded lifecycle state: '''' | ongoing | stopped | completed | error. '
    '''archived'' is computed from companion_session_archives tombstones and '
    'is never written to this column.';
