-- 0149 (slice SH-H1a): the session and MCP hosts registry (ADR-098 D-4).
--
-- One row per host a site may place sandboxes on. The UMRS process's own
-- host registers itself (launch mode `inline`, host id `local`), so placement
-- has one code path on a single node and on a site with agent hosts. Sites
-- share one database, so a host is keyed by (venue, host_id): every front's
-- inline host is `local`, scoped by its site name ("" without a site file).
--
--   state   active    placeable
--           draining  keeps its sessions, gets no new placement (admin)
--           down      missed heartbeats: its sessions ended `host_lost` and
--                     their leases were revoked (site-scoped janitor)
--
-- `labels`, `capacity` and `isolation` are what the host reports (ADR-098
-- D-7: isolation is reported, never assumed); placement reads them (SH-H2).
-- No credential material here; enrollment keys arrive with SH-H1b.
--
-- companion_venue_sessions.host_id names the host a session runs on. Every
-- session so far ran inline, so existing rows read `local`.
--
-- Number: assigned at land time in merge-train order (decisions-2026-11-24
-- §3 item 11); renumber freely, nothing hard-codes it.
--
-- Down (manual):
--   ALTER TABLE companion_venue_sessions DROP COLUMN IF EXISTS host_id;
--   DROP TABLE IF EXISTS hosts;

CREATE TABLE IF NOT EXISTS hosts (
    venue             TEXT NOT NULL DEFAULT '',
    host_id           TEXT NOT NULL,
    launch_mode       TEXT NOT NULL DEFAULT 'inline'
                      CHECK (launch_mode IN ('inline', 'agent')),
    roles             TEXT[] NOT NULL DEFAULT '{session}',
    state             TEXT NOT NULL DEFAULT 'active'
                      CHECK (state IN ('active', 'draining', 'down')),
    reason            TEXT NOT NULL DEFAULT '',
    labels            JSONB NOT NULL DEFAULT '{}'::jsonb,
    capacity          JSONB NOT NULL DEFAULT '{}'::jsonb,
    isolation         JSONB NOT NULL DEFAULT '{}'::jsonb,
    registered_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Non-empty while an admin drain stands; only `undrain` clears it, so a
    -- drained host that goes down comes back `draining`, never `active`.
    drain_reason      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (venue, host_id)
);

COMMENT ON TABLE hosts IS
    'Session and MCP hosts per site (ADR-098 D-4): state, labels, capacity, '
    'reported isolation, last heartbeat. References only - no secrets.';

CREATE INDEX IF NOT EXISTS idx_hosts_venue_state ON hosts (venue, state);

ALTER TABLE companion_venue_sessions
    ADD COLUMN IF NOT EXISTS host_id TEXT NOT NULL DEFAULT 'local';

CREATE INDEX IF NOT EXISTS idx_companion_venue_sessions_host
    ON companion_venue_sessions (host_id)
    WHERE state IN ('provisioning', 'running', 'paused');
