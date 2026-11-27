-- 0150 (slice SH-H1b): host enrollment, host keys, host-bound leases
-- (ADR-098 D-5, D-6).
--
-- host_enrollments  one-time enrollment tokens an admin issues with
--                   `umrs-server admin host enroll <name>`. Only the SHA-256
--                   of the token is stored; the plaintext is printed once.
--                   A token is consumed by compare-and-set (`used_at`), so a
--                   reused or expired token is refused.
-- host_keys         the ed25519 public key a host enrolled with (raw 32
--                   bytes, base64url). A host authenticates with short-lived
--                   tokens it signs with the matching private key; no
--                   private key ever reaches the server. A public key is
--                   not a secret.
-- secret_leases.host_id  the host a lease is bound to. '' is the inline
--                   host (the UMRS process's own, `local`): such a lease is
--                   presented without a host token, as before. A lease bound
--                   to an agent host needs that host's token; a token of
--                   another host gets 403 `rail.host_mismatch`.
--
-- Both tables are keyed by (venue, host_id), like `hosts` (0149): sites share
-- one database. No row here references `hosts`: a host enrolls before its
-- agent first registers.
--
-- Number: assigned at land time in merge-train order (decisions-2026-11-24
-- §3 item 11); renumber freely, nothing hard-codes it.
--
-- Down (manual):
--   ALTER TABLE secret_leases DROP COLUMN IF EXISTS host_id;
--   DROP TABLE IF EXISTS host_keys;
--   DROP TABLE IF EXISTS host_enrollments;

CREATE TABLE IF NOT EXISTS host_enrollments (
    token_hash  TEXT PRIMARY KEY,
    venue       TEXT NOT NULL DEFAULT '',
    host_id     TEXT NOT NULL,
    created_by  TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ NOT NULL,
    used_at     TIMESTAMPTZ,
    -- Issued with --rekey: may replace an enrolled key. A plain token
    -- redeemed after its host enrolled is refused and stays unused.
    rekey       BOOLEAN NOT NULL DEFAULT false
);

COMMENT ON TABLE host_enrollments IS
    'One-time host enrollment tokens (ADR-098 D-5): SHA-256 only, never the token.';

CREATE INDEX IF NOT EXISTS idx_host_enrollments_host
    ON host_enrollments (venue, host_id);

CREATE TABLE IF NOT EXISTS host_keys (
    venue        TEXT NOT NULL DEFAULT '',
    host_id      TEXT NOT NULL,
    public_key   TEXT NOT NULL,
    key_id       TEXT NOT NULL,
    enrolled_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (venue, host_id)
);

COMMENT ON TABLE host_keys IS
    'Enrolled ed25519 host public keys (ADR-098 D-5). Public keys only.';

ALTER TABLE secret_leases
    ADD COLUMN IF NOT EXISTS host_id TEXT NOT NULL DEFAULT '';
