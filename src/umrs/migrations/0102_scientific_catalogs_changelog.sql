-- 0102: Scientific Catalog changelog + pin support (gap row D-43, Phase 1).
--
-- Immutable versions were established by 0101 (one row per
-- (project, name, version); membership pinned at creation; retire =
-- tombstone). This migration adds the changelog channel: every version row
-- carries `change_note` (what changed vs the previous version of the same
-- catalog name — the chain is read via the versions listing, ordered by
-- created_at). External pins (jobs, results, findings, papers pointing at
-- one catalog version) are recorded rows in the row's `provenance.pins`
-- array by the pins route — append-only, never rewrite.

ALTER TABLE scientific_catalogs
    ADD COLUMN IF NOT EXISTS change_note TEXT NOT NULL DEFAULT '';

COMMENT ON COLUMN scientific_catalogs.change_note IS
    'Changelog entry for this version (D-43): what changed vs the previous version of the same catalog name.';
