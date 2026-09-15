-- 0045_tool_output_cache.sql (MEM-A3)
--
-- Session-scoped tool output cache.  Stores tool call results keyed by
-- (session_id, inputs_hash) so identical tool invocations within the
-- same session can be served from cache.  Entries have a TTL and are
-- cleaned up by a periodic janitor.

CREATE TABLE IF NOT EXISTS tool_output_cache (
    entry_id          VARCHAR(64)   PRIMARY KEY,
    session_id        VARCHAR(128)  NOT NULL,
    tool_name         VARCHAR(256)  NOT NULL,
    inputs_hash       VARCHAR(64)   NOT NULL,
    inputs_summary    JSONB,
    output            JSONB,
    output_ref        TEXT,
    output_size_bytes INTEGER       NOT NULL DEFAULT 0,
    hit_count         INTEGER       NOT NULL DEFAULT 0,
    ttl_seconds       INTEGER       NOT NULL DEFAULT 300,
    created_at        TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    expires_at        TIMESTAMPTZ,
    last_accessed_at  TIMESTAMPTZ,

    -- Deduplicate: one entry per (session, inputs_hash)
    CONSTRAINT uq_tool_cache_session_inputs UNIQUE (session_id, inputs_hash)
);

-- Fast lookup by session + inputs_hash (covered by the UNIQUE constraint)
-- Additional index for session listing
CREATE INDEX IF NOT EXISTS idx_tool_cache_session
    ON tool_output_cache(session_id, created_at DESC);

-- Expiry sweep by the janitor
CREATE INDEX IF NOT EXISTS idx_tool_cache_expires
    ON tool_output_cache(expires_at)
    WHERE expires_at IS NOT NULL;

COMMENT ON TABLE tool_output_cache IS 'Session-scoped cache for tool call outputs (MEM-A3). TTL-based with periodic cleanup.';
