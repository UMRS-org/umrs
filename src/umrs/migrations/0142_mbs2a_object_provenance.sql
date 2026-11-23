-- 0142 (slice MB-S2a): object provenance (ADR-115 D-2).
--
-- One row per object version UMRS wrote: (bucket, key, etag). The server
-- computes sha256 itself while the bytes pass through it (the files upload
-- route, step-lane output collection); nothing a client declares is
-- recorded as verified. An object with no row is a legacy object: it reads
-- back as source_kind 'legacy', sha256 unknown, and is verified on demand
-- (OBJECT_VERIFY, slice MB-S2c).
--
-- Number: the next free after 0141 (wave 1A, endpoint registry venue key).
-- The integrator renumbers in merge-train order; the file name keeps the
-- slice id so the renumber is mechanical. Idempotent (IF NOT EXISTS
-- throughout): a stand-alone stack that re-runs its migrations at boot
-- applies it twice without error. Additive only; no existing table changes.

CREATE TABLE IF NOT EXISTS object_provenance (
    bucket                 TEXT        NOT NULL,
    key                    TEXT        NOT NULL,
    etag                   TEXT        NOT NULL,
    size                   BIGINT      NOT NULL CHECK (size >= 0),
    sha256                 TEXT        CHECK (sha256 IS NULL OR sha256 ~ '^[0-9a-f]{64}$'),
    sha256_state           TEXT        NOT NULL DEFAULT 'pending'
        CHECK (sha256_state IN ('pending', 'verified', 'mismatch')),
    content_type           TEXT,
    source_kind            TEXT        NOT NULL
        CHECK (source_kind IN (
            'local_upload', 'nextcloud', 'step_output', 'review_record', 'legacy'
        )),
    source_ref             JSONB       NOT NULL DEFAULT '{}'::jsonb,
    uploader               TEXT,
    project_id             TEXT,
    batch_id               TEXT,
    execution_id           TEXT,
    step                   TEXT,
    workflow_version       TEXT,
    workflow_source_sha256 TEXT,
    derivative_of          TEXT,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    verified_at            TIMESTAMPTZ,
    PRIMARY KEY (bucket, key, etag)
);

-- "Which objects hold these bytes" (selection, dedup, playable derivative).
CREATE INDEX IF NOT EXISTS object_provenance_sha256_idx
    ON object_provenance (sha256) WHERE sha256 IS NOT NULL;

-- "What did this run write" (executions outputs route, MB-S2b).
CREATE INDEX IF NOT EXISTS object_provenance_execution_idx
    ON object_provenance (execution_id, step) WHERE execution_id IS NOT NULL;
