CREATE EXTENSION IF NOT EXISTS vector;

-- below is for passing a minimal health check

-- Create user ejeannot with password and CREATEDB privilege
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'ejeannot') THEN
      CREATE ROLE ejeannot WITH LOGIN PASSWORD 'ejeannot' CREATEDB;
   END IF;
END$$;

-- Grant privileges to ejeannot on public schema and set as database owner
GRANT ALL ON SCHEMA public TO ejeannot;
ALTER DATABASE images_and_texts_ejeannot OWNER TO ejeannot;


