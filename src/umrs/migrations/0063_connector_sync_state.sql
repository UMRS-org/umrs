-- 0063: Durable connector sync state (P2 intake trust).
--
-- Connector checkpoints previously lived in process memory (SyncCheckpoint
-- dataclass): restart lost incremental cursors, forcing full re-crawls, and
-- per-connector health/silence was invisible to the intake board.
--
-- One row per connector *instance* (ConnectorConfig.name is the identity;
-- connector_type retained for introspection). Dead letters are replayable:
-- the DLQ JSONB preserves {source_id, error} triples for operator re-run.

CREATE TABLE IF NOT EXISTS connector_sync_state (
    name TEXT PRIMARY KEY,
    connector_type TEXT NOT NULL,
    cursor TEXT,
    last_sync_at TEXT,
    last_status TEXT NOT NULL DEFAULT 'never',
    docs_synced INTEGER NOT NULL DEFAULT 0,
    errors_count INTEGER NOT NULL DEFAULT 0,
    dlq JSONB NOT NULL DEFAULT '[]'::jsonb,
    checkpoint_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_error TEXT,
    updated_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_connector_sync_state_type
    ON connector_sync_state (connector_type);
CREATE INDEX IF NOT EXISTS idx_connector_sync_state_status
    ON connector_sync_state (last_status);

CREATE TABLE IF NOT EXISTS connector_seen_hashes (
    name TEXT NOT NULL REFERENCES connector_sync_state (name) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    seen_at TEXT NOT NULL,
    PRIMARY KEY (name, content_hash)
);
