-- 0099: durable frontend-venue session runtime rows (wave 2026-09-21, ADR-046 A2).
--
-- One row per session whose turns execute in the sandbox worker on this
-- host: lease + egress binding, provisioning/running/failed/stopped state,
-- the sources the lease carries (names for the UI, ids for the proxy) and
-- the worker-reported runtime (phase, turn_seq, partial text) that the TUI
-- projects while a remote turn runs. The subnet column is the allocator's
-- source of truth (free = not held by a provisioning/running row).
--
-- No credential material ever lands here (owner-scoped references only).

CREATE TABLE IF NOT EXISTS companion_venue_sessions (
    session_id          TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL,
    venue               TEXT NOT NULL DEFAULT 'frontend',
    state               TEXT NOT NULL,                 -- provisioning | running | failed | stopped
    reason              TEXT NOT NULL DEFAULT '',
    lease_id            TEXT NOT NULL DEFAULT '',
    egress_ip           TEXT NOT NULL DEFAULT '',
    subnet              TEXT NOT NULL DEFAULT '',
    source_names        TEXT[] NOT NULL DEFAULT '{}',
    source_ids          UUID[] NOT NULL DEFAULT '{}',
    default_source_id   UUID,
    default_source_name TEXT NOT NULL DEFAULT '',
    web_policy          TEXT NOT NULL DEFAULT 'public',
    runtime             JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    stopped_at          TIMESTAMPTZ
);

COMMENT ON TABLE companion_venue_sessions IS
    'Frontend-venue session runtime rows (ADR-046 A2): lease/egress binding, '
    'state machine, worker-reported runtime. References only — no secrets.';

CREATE INDEX IF NOT EXISTS idx_companion_venue_sessions_user
    ON companion_venue_sessions (user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_companion_venue_sessions_active
    ON companion_venue_sessions (state) WHERE state IN ('provisioning', 'running');
