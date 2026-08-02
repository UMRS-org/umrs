-- I3: Resumable/batch intake — upload manifest table.
--
-- Chunked uploads (part-presigned, server-assembled) track one manifest per
-- logical upload. received_parts holds the part numbers confirmed by the
-- client; parts_detail holds per-part provenance ({n: {etag, size_bytes}}).
-- Status lifecycle: uploading -> assembled | failed.

CREATE TABLE IF NOT EXISTS intake_upload_manifests (
    manifest_id     TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL,
    owner           TEXT NOT NULL DEFAULT '',
    target_key      TEXT NOT NULL,
    target_bucket   TEXT NOT NULL DEFAULT '',
    expected_parts  INTEGER NOT NULL CHECK (expected_parts > 0),
    received_parts  JSONB NOT NULL DEFAULT '[]'::jsonb,
    status          TEXT NOT NULL DEFAULT 'uploading',
    final_etag      TEXT,
    size_bytes      BIGINT,
    parts_detail    JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_note      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    assembled_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS intake_upload_manifests_project_idx
    ON intake_upload_manifests (project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS intake_upload_manifests_status_idx
    ON intake_upload_manifests (status, created_at DESC);
