-- 0115: self-service personal buckets (ADR-079).
--
-- A verified user creates `<user>.<project>.<alias>` (+ `-output`) on the
-- server's own object store through UMRS (`POST /umrs/buckets`); the row is
-- the ownership record the HPC lane's staging check (ADR-074 D-5) and the
-- files API consult, next to the provisioned buckets of the user configs.
-- No credential is stored or handed out: every S3 operation on these
-- buckets is performed by the server (constitution §2).

CREATE TABLE IF NOT EXISTS user_buckets (
    bucket      TEXT PRIMARY KEY,
    owner_id    TEXT NOT NULL,
    project     TEXT NOT NULL,
    alias       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT user_buckets_name_shape CHECK (bucket !~ '-output$')
);

CREATE INDEX IF NOT EXISTS user_buckets_owner_idx ON user_buckets (owner_id);

COMMENT ON TABLE user_buckets IS
    'ADR-079: self-service personal buckets (input bucket name; its -output pair belongs to the same owner). Ownership record only — no credentials.';
