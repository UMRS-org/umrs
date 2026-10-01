-- 0114: per-project external storage binding (ADR-077, gap row D-17).
--
-- A project may point its object store (S3 URL) and its Postgres (libpq
-- URI) at stores outside the UMRS front-end — typically next to an HPC
-- site's large filesystem. No row = the server's own stores (unchanged).
--
-- Both locators carry credentials, so they are stored SEALED only
-- (AES-256-GCM, AAD = project + field; umrs/shared/storage/project_storage.py)
-- and are never returned by any route — responses carry the redacted form
-- (endpoint/bucket/access-key hint; host/port/db/user). A sealing-key change
-- leaves a row unreadable: the data paths then refuse the project (fail
-- closed) until the binding is set again.
--
-- One row per project; the row dies with an explicit clear, never with the
-- project (projects are tombstoned, not deleted).

CREATE TABLE IF NOT EXISTS project_storage_bindings (
    project_id     TEXT PRIMARY KEY REFERENCES projects (project_id),
    s3_url_sealed  TEXT NOT NULL DEFAULT '',
    pg_uri_sealed  TEXT NOT NULL DEFAULT '',
    updated_by     TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT project_storage_bindings_not_empty
        CHECK (s3_url_sealed <> '' OR pg_uri_sealed <> '')
);

COMMENT ON TABLE project_storage_bindings IS
    'ADR-077: a project''s external object store / Postgres. Sealed locators only; routes return redacted forms. No row = the server''s own stores.';
COMMENT ON COLUMN project_storage_bindings.s3_url_sealed IS
    'Sealed http(s)://ACCESS:SECRET@host:port[/bucket][?region=&verify=&backend=] (empty = server default object store).';
COMMENT ON COLUMN project_storage_bindings.pg_uri_sealed IS
    'Sealed postgresql://user:password@host:port/db[?sslmode=] (empty = server default Postgres).';
