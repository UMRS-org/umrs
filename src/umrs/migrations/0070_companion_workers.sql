-- Migration 0070: Durable companion-worker registry (wave T3a)
--
-- One row per (owner_user_id, project_id) companion worker job. The registry
-- is the authority for token issuance + resume; liveness transitions are
-- pushed by the attempt loop into CompanionWorkerManager.heartbeat().
--
-- Invariants:
--   * Resume tokens are NEVER stored, only the sha256 fingerprint prefix.
--   * restart exhaustion moves the row through 'draining' → 'failed',
--     not just idle-time eviction (eviction gate, not TTL-only).
--   * 'draining' grants one graceful reconnect window to the in-flight
--     session; 'failed'/'evicted' reject new tokens (WorkerExhausted).

CREATE TABLE IF NOT EXISTS companion_workers (
    registry_key               TEXT         PRIMARY KEY,  -- "<user>@<project>"
    owner_user_id              TEXT         NOT NULL,
    project_id                 TEXT         NOT NULL,
    job_id                     TEXT         NOT NULL,
    session_token_fingerprint  TEXT         NOT NULL DEFAULT '',
    state                      TEXT         NOT NULL DEFAULT 'active'
                               CHECK (state IN ('active', 'draining', 'failed', 'evicted')),
    attempt_index              INTEGER      NOT NULL DEFAULT 0,
    attempts_max               INTEGER      NOT NULL DEFAULT 6,
    last_heartbeat_at          TIMESTAMPTZ,
    last_active_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
    created_at                 TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at                 TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_companion_workers_owner
    ON companion_workers(owner_user_id, project_id);

CREATE INDEX IF NOT EXISTS idx_companion_workers_state
    ON companion_workers(state);

COMMENT ON TABLE companion_workers IS
    'Durable companion-worker registry (wave T3a venue). Resume tokens are '
    'fingerprinted only; liveness decisions come from pushed heartbeats with '
    'attempt_index against attempts_max, not from TTL alone.';
