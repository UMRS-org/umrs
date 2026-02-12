-- UMRS Demo Database Initialization
-- This script sets up the PostgreSQL database with pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table for metadata
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    content TEXT,
    metadata JSONB DEFAULT '{}',
    s3_key VARCHAR(1024),
    bucket_alias VARCHAR(255),
    username VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create embeddings table with pgvector
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(255) REFERENCES documents(id) ON DELETE CASCADE,
    embedding vector(384),  -- all-MiniLM-L6-v2 produces 384-dim embeddings
    model_name VARCHAR(255) DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for similarity search (IVFFlat for larger datasets)
CREATE INDEX IF NOT EXISTS embeddings_vector_idx ON embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create buckets configuration table
CREATE TABLE IF NOT EXISTS buckets (
    id SERIAL PRIMARY KEY,
    alias VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(255) NOT NULL,
    modality VARCHAR(50) DEFAULT 'any',  -- text, image, audio, any
    s3_path VARCHAR(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create jobs table for tracking async operations
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) NOT NULL,
    job_type VARCHAR(50) NOT NULL,  -- embedding, upload, download
    status VARCHAR(50) DEFAULT 'pending',  -- pending, running, completed, failed
    input_path TEXT,
    output_path TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for timestamp updates
DROP TRIGGER IF EXISTS documents_updated_at ON documents;
CREATE TRIGGER documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS jobs_updated_at ON jobs;
CREATE TRIGGER jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Insert default bucket for demo
INSERT INTO buckets (alias, username, modality, s3_path)
VALUES ('demo_bucket', 'demo_user', 'text', 's3://demo-bucket/')
ON CONFLICT (alias) DO NOTHING;

-- Create function for similarity search
CREATE OR REPLACE FUNCTION search_similar_documents(
    query_embedding vector(384),
    limit_count INT DEFAULT 5
)
RETURNS TABLE (
    doc_id VARCHAR(255),
    title VARCHAR(500),
    content TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        d.content,
        1 - (e.embedding <=> query_embedding) AS similarity
    FROM embeddings e
    JOIN documents d ON e.doc_id = d.id
    ORDER BY e.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'UMRS demo database initialized successfully';
END $$;
