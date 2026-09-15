-- Migration 0042: Replay inputs support (MEM-B5)
--
-- Adds:
-- 1. inputs_snapshot JSONB column to execution_traces for inline small payloads
-- 2. outputs_snapshot JSONB column for inline small outputs
-- 3. Index on (step_name, status) for replay queries across jobs
-- 4. Index on inputs_hash for deduplication / identical-input detection

-- Inline input/output snapshots for replay (small payloads only; large ones go to S3)
ALTER TABLE execution_traces
    ADD COLUMN IF NOT EXISTS inputs_snapshot  JSONB,
    ADD COLUMN IF NOT EXISTS outputs_snapshot JSONB;

-- Index for querying traces by step name + status (replay use-case)
CREATE INDEX IF NOT EXISTS idx_execution_traces_step_status
    ON execution_traces (step_name, status);

-- Index for finding traces with identical inputs (dedup / comparison)
CREATE INDEX IF NOT EXISTS idx_execution_traces_inputs_hash
    ON execution_traces (inputs_hash)
    WHERE inputs_hash IS NOT NULL;
