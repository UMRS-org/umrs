-- Initialize pgvector on the default 'postgres' database and ensure a default admin role exists.
CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'umrs') THEN
      CREATE ROLE umrs WITH LOGIN PASSWORD 'umrs' CREATEDB;
   END IF;
END$$;
