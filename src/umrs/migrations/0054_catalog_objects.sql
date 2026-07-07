-- 0054: Data catalog metadata (Postgres source of truth)
--
-- CatalogObjectMeta rows replace S3 _data_catalog/objects/*.meta.json sidecars.
-- Extracted text and structured knowledge blobs remain in S3.
--
-- Rollback (forward-only): drop catalog_objects; redeploy code that reads S3
-- sidecars; optional backfill from remaining .meta.json files if any were kept.

CREATE TABLE IF NOT EXISTS catalog_objects (
    project_id            VARCHAR(128) NOT NULL,
    s3_key                TEXT         NOT NULL,
    object_etag           VARCHAR(128) NOT NULL DEFAULT '',
    state                 VARCHAR(16)  NOT NULL DEFAULT 'pending',
    owner                 VARCHAR(128) NOT NULL DEFAULT '',
    acl_scope             VARCHAR(128) NOT NULL DEFAULT '',
    summary               TEXT         NOT NULL DEFAULT '',
    deterministic_summary TEXT         NOT NULL DEFAULT '',
    perception_summary    TEXT,
    perception_model      VARCHAR(256),
    tags                  JSONB        NOT NULL DEFAULT '[]',
    filename_signals      JSONB        NOT NULL DEFAULT '[]',
    detected_modality     VARCHAR(64)  NOT NULL DEFAULT 'unknown',
    extracted_text_ref    TEXT,
    extracted_knowledge_ref TEXT,
    error_note            TEXT,
    updated_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    search_vector         tsvector GENERATED ALWAYS AS (
        to_tsvector(
            'simple',
            coalesce(summary, '') || ' ' ||
            coalesce(deterministic_summary, '') || ' ' ||
            coalesce(s3_key, '') || ' ' ||
            coalesce(tags::text, '') || ' ' ||
            coalesce(filename_signals::text, '')
        )
    ) STORED,
    PRIMARY KEY (project_id, s3_key),
    CHECK (state IN ('pending', 'ready', 'error'))
);

CREATE INDEX IF NOT EXISTS idx_catalog_objects_project_updated
    ON catalog_objects (project_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_catalog_objects_project_state
    ON catalog_objects (project_id, state);

CREATE INDEX IF NOT EXISTS idx_catalog_objects_tags
    ON catalog_objects USING GIN (tags jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_catalog_objects_search
    ON catalog_objects USING GIN (search_vector);

COMMENT ON TABLE catalog_objects IS 'Data catalog metadata — agent search index (Postgres SoT)';
