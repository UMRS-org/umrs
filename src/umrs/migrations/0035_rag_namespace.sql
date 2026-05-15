-- 0035_rag_namespace.sql
-- Additive tenant-scoping column for RAG context tables.
--
-- Design review conclusion (2026-05-15):
--   The existing project_id column on conversations and memory_items
--   already provides adequate tenant scoping for current system/project/user
--   tiers.  Schema-per-tenant is not warranted unless data isolation
--   compliance requirements mandate it.
--
--   This migration adds an optional `namespace` column to memory_items and
--   memory_vectors so RAG indices can be partitioned within a project
--   (e.g. per-dataset or per-experiment) without schema-level isolation.

-- Add namespace to memory_items for RAG index partitioning.
ALTER TABLE memory_items
    ADD COLUMN IF NOT EXISTS namespace TEXT NOT NULL DEFAULT 'default';

-- Composite index: project + namespace for filtered retrieval.
CREATE INDEX IF NOT EXISTS idx_memory_items_project_namespace
    ON memory_items(project_id, namespace)
    WHERE project_id IS NOT NULL;

COMMENT ON COLUMN memory_items.namespace IS 'RAG index partition within a project (default, per-dataset, per-experiment)';
