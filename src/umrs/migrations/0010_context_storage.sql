-- Context Storage (Phase 3, 2026-05-15)
--
-- Server-side durable conversation history, messages, summaries,
-- and long-term memory with vector retrieval.
--
-- Referenced in development.instructions.md as the UMRS Context Storage
-- module. Supports:
--   - POST /umrs/context/pack (assemble context for LLM)
--   - Conversation threading with revision-based cache invalidation
--   - Memory retrieval via pgvector similarity search

-- Require pgvector for memory_vectors (noop if already loaded)
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- Conversations: thread-level metadata with rev for cache invalidation
-- =============================================================================

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT         NOT NULL,
    project_id      TEXT,
    title           TEXT         NOT NULL DEFAULT '',
    model           TEXT         NOT NULL DEFAULT '',
    system_prompt   TEXT         NOT NULL DEFAULT '',
    rev             INTEGER      NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_user
    ON conversations(user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_project
    ON conversations(project_id, updated_at DESC)
    WHERE project_id IS NOT NULL;

-- =============================================================================
-- Messages: individual turns within a conversation
-- =============================================================================

CREATE TABLE IF NOT EXISTS messages (
    message_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID         NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    role            TEXT         NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content         TEXT         NOT NULL,
    token_count     INTEGER      NOT NULL DEFAULT 0,
    seq             INTEGER      NOT NULL,
    model           TEXT,
    tool_calls      JSONB,
    metadata        JSONB        NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_seq
    ON messages(conversation_id, seq);

-- =============================================================================
-- Conversation Summaries: compacted older messages for context window mgmt
-- =============================================================================

CREATE TABLE IF NOT EXISTS conversation_summaries (
    summary_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID         NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    content         TEXT         NOT NULL,
    token_count     INTEGER      NOT NULL DEFAULT 0,
    messages_start  INTEGER      NOT NULL,
    messages_end    INTEGER      NOT NULL,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_summaries_conversation
    ON conversation_summaries(conversation_id, created_at DESC);

-- =============================================================================
-- Memory Items: long-term retrievable facts
-- =============================================================================

CREATE TABLE IF NOT EXISTS memory_items (
    memory_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT         NOT NULL,
    project_id      TEXT,
    content         TEXT         NOT NULL,
    source          TEXT         NOT NULL DEFAULT 'conversation',
    importance      REAL         NOT NULL DEFAULT 0.5,
    access_count    INTEGER      NOT NULL DEFAULT 0,
    last_accessed   TIMESTAMPTZ,
    metadata        JSONB        NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_items_user
    ON memory_items(user_id, importance DESC);

CREATE INDEX IF NOT EXISTS idx_memory_items_project
    ON memory_items(project_id, importance DESC)
    WHERE project_id IS NOT NULL;

-- =============================================================================
-- Memory Vectors: pgvector embeddings for semantic retrieval
-- =============================================================================

CREATE TABLE IF NOT EXISTS memory_vectors (
    vector_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id       UUID         NOT NULL REFERENCES memory_items(memory_id) ON DELETE CASCADE,
    embedding       vector(1024),
    model           TEXT         NOT NULL DEFAULT 'default',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_vectors_memory
    ON memory_vectors(memory_id);

-- HNSW index for fast approximate nearest-neighbor search
CREATE INDEX IF NOT EXISTS idx_memory_vectors_embedding
    ON memory_vectors USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE conversations IS 'Server-side conversation threads with rev-based cache invalidation';
COMMENT ON TABLE messages IS 'Individual message turns within a conversation';
COMMENT ON TABLE conversation_summaries IS 'Compacted summaries of older messages for context window management';
COMMENT ON TABLE memory_items IS 'Long-term retrievable memory facts per user/project';
COMMENT ON TABLE memory_vectors IS 'pgvector embeddings for semantic memory retrieval';
