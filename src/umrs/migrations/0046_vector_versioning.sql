-- 0046_vector_versioning.sql (MEM-C2)
--
-- Embedding model version registry and re-embedding job tracking.
-- Supports upgrading embedding models while keeping old vectors
-- available until migration completes.

-- =============================================================================
-- Embedding model version registry
-- =============================================================================

CREATE TABLE IF NOT EXISTS embedding_model_versions (
    version_id          VARCHAR(64)   PRIMARY KEY,
    model_name          TEXT          NOT NULL,
    model_version       TEXT          NOT NULL,
    model_path          TEXT,
    embedding_dimension INTEGER       NOT NULL DEFAULT 0,
    is_active           BOOLEAN       NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    -- Only one active version per model at a time
    CONSTRAINT uq_active_model_version UNIQUE (model_name, model_version)
);

-- Fast lookup: active version for a model
CREATE INDEX IF NOT EXISTS idx_emv_active
    ON embedding_model_versions(model_name, is_active)
    WHERE is_active = TRUE;

-- =============================================================================
-- Re-embedding job tracking
-- =============================================================================

CREATE TABLE IF NOT EXISTS reembed_jobs (
    reembed_id          VARCHAR(64)   PRIMARY KEY,
    source_model        TEXT          NOT NULL,
    source_version      TEXT          NOT NULL,
    target_model        TEXT          NOT NULL,
    target_version      TEXT          NOT NULL,
    target_dimension    INTEGER       NOT NULL DEFAULT 0,
    status              VARCHAR(20)   NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')),
    total_vectors       INTEGER       NOT NULL DEFAULT 0,
    processed_vectors   INTEGER       NOT NULL DEFAULT 0,
    failed_vectors      INTEGER       NOT NULL DEFAULT 0,
    job_id              VARCHAR(64),
    error               TEXT,
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- Active re-embedding jobs
CREATE INDEX IF NOT EXISTS idx_reembed_status
    ON reembed_jobs(status)
    WHERE status IN ('pending', 'running');

-- Linked durable job
CREATE INDEX IF NOT EXISTS idx_reembed_job
    ON reembed_jobs(job_id)
    WHERE job_id IS NOT NULL;

COMMENT ON TABLE embedding_model_versions IS 'Registry of embedding model versions for vector versioning (MEM-C2)';
COMMENT ON TABLE reembed_jobs IS 'Tracks re-embedding migrations between embedding model versions (MEM-C2)';
