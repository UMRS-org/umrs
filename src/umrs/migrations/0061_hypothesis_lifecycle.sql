-- 0061: Durable hypothesis lifecycle (K1 tail — closes the research loop).
--
-- HypothesisLifecycle previously kept hypotheses + transition history in
-- process memory, losing the whole scientific claim graph on UMRS restart.
-- Paired tables: hypotheses (current state) + hypothesis_history (append-only
-- transition log, mirroring the in-memory _history contract).

CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    statement TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'proposed',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    supporting_evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    contradicting_evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    originating_planner_version TEXT NOT NULL DEFAULT '',
    associated_experiments JSONB NOT NULL DEFAULT '[]'::jsonb,
    parent_hypothesis_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_hypotheses_status
    ON hypotheses (status);
CREATE INDEX IF NOT EXISTS idx_hypotheses_created
    ON hypotheses (created_at ASC);

CREATE TABLE IF NOT EXISTS hypothesis_history (
    seq BIGSERIAL PRIMARY KEY,
    hypothesis_id TEXT NOT NULL REFERENCES hypotheses (hypothesis_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hypothesis_history_hyp
    ON hypothesis_history (hypothesis_id, seq ASC);
