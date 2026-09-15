-- 0041_rag_lineage_tracking.sql (MEM-C1)
--
-- Add lineage tracking columns to memory_items and memory_vectors
-- so RAG indexing operations record:
--   - Document source (URI, type)
--   - Embedding model version and path
--   - Embedding dimension
--   - The durable job that performed the indexing
--
-- These columns are additive and nullable; existing rows remain valid.

-- =============================================================================
-- memory_items: document source tracking
-- =============================================================================

ALTER TABLE memory_items
    ADD COLUMN IF NOT EXISTS document_source_uri  TEXT,
    ADD COLUMN IF NOT EXISTS document_source_type TEXT;

COMMENT ON COLUMN memory_items.document_source_uri IS 'Origin of indexed document: s3://bucket/key, file:///path, https://..., or NULL for conversation-sourced items';
COMMENT ON COLUMN memory_items.document_source_type IS 'Source classification: s3, file, upload, api, conversation';

CREATE INDEX IF NOT EXISTS idx_memory_items_source_type
    ON memory_items(document_source_type)
    WHERE document_source_type IS NOT NULL;

-- =============================================================================
-- memory_vectors: embedding model lineage
-- =============================================================================

ALTER TABLE memory_vectors
    ADD COLUMN IF NOT EXISTS model_version       TEXT,
    ADD COLUMN IF NOT EXISTS model_path          TEXT,
    ADD COLUMN IF NOT EXISTS embedding_dimension INTEGER,
    ADD COLUMN IF NOT EXISTS indexing_job_id     VARCHAR(64);

COMMENT ON COLUMN memory_vectors.model_version IS 'Embedding model version string (semver, date, or commit)';
COMMENT ON COLUMN memory_vectors.model_path IS 'Filesystem path or HF repo of the embedding model used';
COMMENT ON COLUMN memory_vectors.embedding_dimension IS 'Dimensionality of the embedding vector (384, 768, 1024, etc.)';
COMMENT ON COLUMN memory_vectors.indexing_job_id IS 'Durable job ID that produced this embedding, for traceability';

-- Lineage query index: find all embeddings from a specific job or model version
CREATE INDEX IF NOT EXISTS idx_memory_vectors_lineage
    ON memory_vectors(indexing_job_id, model_version, created_at DESC)
    WHERE indexing_job_id IS NOT NULL;

-- Model version index: find embeddings by model for re-indexing decisions
CREATE INDEX IF NOT EXISTS idx_memory_vectors_model_version
    ON memory_vectors(model, model_version)
    WHERE model_version IS NOT NULL;
