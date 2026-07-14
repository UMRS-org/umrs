-- Migration 0056: Knowledge events table (TASK-013)
--
-- Creates the knowledge_events table for pub/sub event persistence.
-- Events carry ingestion lifecycle state (parse, extract, embed, index, etc.)
-- Consumers: SSE endpoint, audit log, state machine updates.

-- =============================================================================
-- 1. Knowledge Events (immutable event log)
-- =============================================================================
CREATE TABLE IF NOT EXISTS knowledge_events (
    event_id     TEXT              PRIMARY KEY,
    event_type   TEXT              NOT NULL CHECK (event_type IN (
        'parsed', 'extracted', 'embedded', 'indexed',
        'archived', 'failed', 'resumed'
    )),
    job_id       TEXT              NOT NULL,
    attempt_id   TEXT              NOT NULL,
    artifact_id  TEXT              NOT NULL REFERENCES knowledge_artifacts(artifact_id) ON DELETE SET NULL,
    source_ref   TEXT              DEFAULT '',
    details      JSONB             DEFAULT '{}',
    timestamp    TIMESTAMPTZ       NOT NULL DEFAULT now(),
    created_at   TIMESTAMPTZ       NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_events_job
    ON knowledge_events(job_id, attempt_id);

CREATE INDEX IF NOT EXISTS idx_events_artifact
    ON knowledge_events(artifact_id);

CREATE INDEX IF NOT EXISTS idx_events_type
    ON knowledge_events(event_type);

CREATE INDEX IF NOT EXISTS idx_events_timestamp
    ON knowledge_events(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_events_job_artifact
    ON knowledge_events(job_id, artifact_id);

COMMENT ON TABLE knowledge_events IS
    'Immutable knowledge ingestion lifecycle events. Consumers: SSE, audit, state machine.';
