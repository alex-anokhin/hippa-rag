-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create tables for HIPAA document chunks
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    meta_data JSONB  -- Changed from 'metadata' to 'meta_data'
);

-- Create embeddings table with vector support
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES document_chunks(id) ON DELETE CASCADE,
    embedding vector(1536)
);

-- Create index on embeddings for faster similarity search
CREATE INDEX IF NOT EXISTS embeddings_vector_idx ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);