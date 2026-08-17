-- Migration 0071: Companion turn-attempt registry (wave T3b, ADR-028)
--
-- Unless stated otherwise, one chat turn started through a venue companion
-- worker is registered as an ATTEMPT against the companion_worker job —
-- never a new job per prompt (ADR-028). Attempt-level rules:
--   * exactly-once commits at turn completion (no partial writes),
--   * idempotent re-issue via request_token (UNIQUE per worker),
--   * transcripts land behind the attempt buffer, reattach replays never
--     re-executes.

CREATE TABLE IF NOT EXISTS companion_turns (
    attempt_id         TEXT         PRIMARY KEY,
    job_id             TEXT         NOT NULL REFERENCES jobs(job_id),
    registry_key       TEXT         NOT NULL REFERENCES companion_workers(registry_key),
    session_id         TEXT         NOT NULL DEFAULT '',
    request_token      TEXT         NOT NULL,
    user_message_hash  TEXT         NOT NULL DEFAULT '',
    state              TEXT         NOT NULL DEFAULT 'in_flight'
                       CHECK (state IN ('in_flight', 'committed', 'failed')),
    committed_response_ref TEXT,    -- pointer into transcript/event log (opaque)
    parent_attempt_id  TEXT,        -- for re-attempts (reattempt_of linking)
    research_session_id TEXT,       -- ADR-025 linkage (optional, no implicit overlay)
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (registry_key, request_token)
);

CREATE INDEX IF NOT EXISTS idx_companion_turns_registry
    ON companion_turns(registry_key, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_companion_turns_state
    ON companion_turns(state);

COMMENT ON TABLE companion_turns IS
    'Turn-level attempt registry for companion workers (ADR-028). One row per '
    'started turn: exactly-once commit, idempotent re-issue via request_token, '
    'attempt id carried by every T1 transparency envelope for per-client live '
    'trace rehydration.';
