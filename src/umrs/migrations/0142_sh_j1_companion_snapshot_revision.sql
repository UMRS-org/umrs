-- 0142 (slice SH-J1): revision compare-and-swap on the companion session journal.
--
-- ADR-057 amendment D-7 (decision SH-D1, 2026-11-24). Snapshot PUTs were
-- settled by timestamp LWW on update_marker, so two clients writing from one
-- base silently dropped one client's turn. Every applied client PUT (owner
-- route and venue rail) now bumps `revision`; a PUT that carries
-- `base_revision` applies only when it equals the stored revision, else it is
-- refused with a typed 409 `companionsessions.revision_conflict` and the
-- client keeps its copy as a conflict copy. A PUT without `base_revision`
-- keeps the legacy marker LWW (deprecated, supported) and still bumps.
--
-- Server-owned merges (queue, session-bus presence, venue/state projection,
-- subagent child rows) do not bump: a client's payload never carries them
-- (the upsert re-applies the stored queue), so they cannot be a lost update.
--
-- Rows from before this migration read revision 0 ("never versioned"); a
-- first insert writes 1. A client feature-detects the column by `revision`
-- appearing in GET/list/PUT answers.
--
-- Renumber note: the number is the next free one after 0141 (UMRS-org PR
-- #153, unmerged when this was written); the integrator renumbers in
-- merge-train order. The slice id in the name keeps the file findable.
--
-- Down (manual, no data loss beyond the counter):
--   ALTER TABLE companion_session_snapshots DROP COLUMN IF EXISTS revision;

ALTER TABLE companion_session_snapshots
    ADD COLUMN IF NOT EXISTS revision BIGINT NOT NULL DEFAULT 0;

COMMENT ON COLUMN companion_session_snapshots.revision IS
    'Journal revision (ADR-057 D-7): bumped by every applied client snapshot '
    'PUT; base_revision PUTs compare-and-swap on it. 0 = never versioned.';
