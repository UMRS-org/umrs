-- 0096: per-user disk usage ledger (ADR-051 D-3).
--
-- Attribution ledger (NOT the catalog): writes increment bytes on upload
-- completion; tombstones/drops decrement best-effort; hash-dedup rides
-- the uploader's first write (shared bytes are NOT double-charged).

CREATE TABLE IF NOT EXISTS user_disk_usage (
    user_id       TEXT PRIMARY KEY,
    bytes_used    BIGINT NOT NULL DEFAULT 0,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT user_disk_usage_nonnegative CHECK (bytes_used >= 0)
);

CREATE TABLE IF NOT EXISTS user_disk_objects (
    content_hash  TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    bytes         BIGINT NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (content_hash, user_id)
);

COMMENT ON TABLE user_disk_usage IS
    'Per-user disk mileage metered at durable upload points (ADR-051 D-3). Additive ledger; the row never resets unaccountably.';
COMMENT ON TABLE user_disk_objects IS
    'Hash-dedup registry per uploader (ADR-051 D-3): each unique content hash is charged ONCE per uploader.';
