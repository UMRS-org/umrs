-- Migration 0053: Add reasoning_tokens column to llm_usage_log
--
-- GLM-5.2 (and other thinking models) report reasoning/thinking tokens
-- separately from completion tokens. vLLM returns these in
-- ``completion_tokens_details.reasoning_tokens`` (OpenAI format).
--
-- The ``completion_tokens`` field INCLUDES ``reasoning_tokens``, so for
-- accurate billing we must track them separately:
--
--   actual_response_tokens = completion_tokens - reasoning_tokens
--
-- This migration also adds prediction-related token columns for
-- speculative decoding (used by some vLLM configurations).

-- Add reasoning_tokens column
ALTER TABLE llm_usage_log
    ADD COLUMN IF NOT EXISTS reasoning_tokens INTEGER NOT NULL DEFAULT 0;

-- Add prediction tokens columns (speculative decoding)
ALTER TABLE llm_usage_log
    ADD COLUMN IF NOT EXISTS accepted_prediction_tokens INTEGER NOT NULL DEFAULT 0;

ALTER TABLE llm_usage_log
    ADD COLUMN IF NOT EXISTS rejected_prediction_tokens INTEGER NOT NULL DEFAULT 0;

-- Add audio_tokens column (for multimodal models)
ALTER TABLE llm_usage_log
    ADD COLUMN IF NOT EXISTS audio_tokens INTEGER NOT NULL DEFAULT 0;

-- Add an index on reasoning_tokens for quick aggregation queries
-- (useful for billing breakdowns by thinking vs non-thinking models)
CREATE INDEX IF NOT EXISTS idx_llm_usage_reasoning_tokens
    ON llm_usage_log (reasoning_tokens)
    WHERE reasoning_tokens > 0;
