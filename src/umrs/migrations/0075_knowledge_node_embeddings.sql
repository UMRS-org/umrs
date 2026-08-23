-- 0075: node-keyed embedding vectors for canonical graph nodes
-- (ADR-027 G5 / P1b).
--
-- Additive-only foundation: nothing reads these rows at runtime yet. The
-- owning write path (query-time RAG embedder of rag_service vs. the batch
-- compute pipeline of umrs.inference.embeddings, see R6.1 ownership
-- boundaries) is decided together with the P2 facade switch (G1').
--
-- The `vector` column is declared WITHOUT a fixed dimension: models in play
-- differ (E5≈1024, hash fallback=384). Cross-dimension similarity queries
-- fail loudly in Postgres, which is the desired behavior — the dimension
-- column + model_name PK segment keep models from mixing.
CREATE TABLE IF NOT EXISTS knowledge_node_embeddings (
    node_id      TEXT NOT NULL REFERENCES knowledge_nodes(node_id) ON DELETE CASCADE,
    model_name   TEXT NOT NULL,
    dimension    INT  NOT NULL CHECK (dimension > 0),
    embedding    vector,
    content_hash TEXT NOT NULL DEFAULT '',
    created_at   TIMESTAMPTZ DEFAULT now(),
    updated_at   TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (node_id, model_name)
);

COMMENT ON TABLE knowledge_node_embeddings IS
    'Node-keyed pgvector rows for canonical knowledge_nodes (ADR-027 G5 / P1b). '
    'Additive foundation for the P2 facade switch; write-path owner pending G1''';

CREATE INDEX IF NOT EXISTS idx_node_embeddings_hnsw
    ON knowledge_node_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_node_embeddings_model
    ON knowledge_node_embeddings (model_name);
