-- 0059: Durable stores for research-layer satellites (K1 backbone).
--
-- Postgres-backed persistence for scientific memory, experiment registry,
-- and evidence tracking. These satellites previously kept state in process
-- memory (reference implementations), losing everything on UMRS restart.
--
-- Timestamp columns are TEXT (ISO-8601 strings) to match the satellite
-- dataclass contracts exactly and guarantee lossless round-trips with the
-- in-memory reference implementations.

CREATE TABLE IF NOT EXISTS scientific_memory_entries (
    entry_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT '',
    experiment_id TEXT NOT NULL DEFAULT '',
    entry_type TEXT NOT NULL DEFAULT 'note',
    source TEXT NOT NULL DEFAULT 'system',
    content TEXT NOT NULL DEFAULT '',
    content_uri TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL DEFAULT 'active',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sm_entries_project_created
    ON scientific_memory_entries (project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sm_entries_experiment
    ON scientific_memory_entries (experiment_id);
CREATE INDEX IF NOT EXISTS idx_sm_entries_status
    ON scientific_memory_entries (status);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT '',
    hypothesis_id TEXT NOT NULL DEFAULT '',
    protocol JSONB NOT NULL DEFAULT '{}'::jsonb,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    validation_method TEXT NOT NULL DEFAULT 'statistical_test',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL DEFAULT '',
    submitted_at TEXT,
    completed_at TEXT,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_experiments_project_created
    ON experiments (project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiments_hypothesis
    ON experiments (hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_experiments_status
    ON experiments (status);

CREATE TABLE IF NOT EXISTS experiment_results (
    result_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments (experiment_id) ON DELETE CASCADE,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifacts JSONB NOT NULL DEFAULT '[]'::jsonb,
    logs TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'completed',
    created_at TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_experiment_results_experiment
    ON experiment_results (experiment_id, created_at ASC);

CREATE TABLE IF NOT EXISTS evidence_items (
    evidence_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT '',
    claim_id TEXT NOT NULL DEFAULT '',
    source_type TEXT NOT NULL DEFAULT 'manual',
    source_id TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    content_uri TEXT NOT NULL DEFAULT '',
    relevance_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    citation TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_evidence_items_claim
    ON evidence_items (claim_id);
CREATE INDEX IF NOT EXISTS idx_evidence_items_project
    ON evidence_items (project_id, relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_evidence_items_status
    ON evidence_items (status);
