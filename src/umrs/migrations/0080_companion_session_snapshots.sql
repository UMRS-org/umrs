-- Migration 0080: Companion session snapshots (FWP, ADR-036 F2 first leg)
--
-- Remote snapshot of a client-side companion session. One row per
-- (user_id, session_id), owner-pinned. This is the journaled copy behind
-- the companion SQLite read-model: writes are mirrored from clients;
-- fresh machines rebuild local state by pulling these snapshots.

CREATE TABLE IF NOT EXISTS companion_session_snapshots (
    user_id       TEXT         NOT NULL,
    session_id    TEXT         NOT NULL,
    name          TEXT         NOT NULL DEFAULT '',
    payload       TEXT         NOT NULL,
    update_marker TEXT         NOT NULL,
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),

    PRIMARY KEY (user_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_companion_session_snapshots_user
    ON companion_session_snapshots(user_id, updated_at);

COMMENT ON TABLE companion_session_snapshots IS
    'Companion session snapshot journal (ADR-036 F2 first leg). One row per '
    '(user_id, session_id); payload is the full session document '
    '(messages+skills+memory+metadata). Authoritative while the client is '
    'online; the local SQLite store is the rebuildable read-model over it.';
