-- 0087: source grants (R3-1, ADR-046 + ADR-041 lineage).
--
-- Grant a sealed source to another user (grantee) outside the owner's
-- static scope: a grantee's sessions may hold leases referencing the
-- granted source. Revocation is a tombstone (tombstoned_at), never a
-- DELETE, so the audit trail survives; live-grant uniqueness only covers
-- untombstoned rows, so re-granting after a revoke works.

CREATE TABLE IF NOT EXISTS source_grants (
    grant_id      TEXT PRIMARY KEY,
    source_id     uuid NOT NULL REFERENCES user_sources_sealed (id) ON DELETE CASCADE,
    owner         TEXT NOT NULL,                 -- source owner at grant time (denormalized for owner-scoped reads)
    grantee       TEXT NOT NULL,                 -- user id receiving the grant
    allow_paths   TEXT[] NOT NULL DEFAULT '{}',  -- narrows the source's own opt-ins (intersection at issue)
    budget_usd    NUMERIC(12,4),                 -- R3 reserved; NULL = uncapped (honest absence)
    expires_at    TIMESTAMPTZ,
    tombstoned_at TIMESTAMPTZ,
    created_by    TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE source_grants IS
    'R3-1: share a sealed provider source with another user. Tombstone-on-revoke; '
    'lease issue checks (ownership OR live grant) per ADR-046.';

CREATE UNIQUE INDEX IF NOT EXISTS source_grants_live_unique
    ON source_grants (source_id, grantee)
    WHERE tombstoned_at IS NULL;

CREATE INDEX IF NOT EXISTS source_grants_owner_idx
    ON source_grants (owner) WHERE tombstoned_at IS NULL;
CREATE INDEX IF NOT EXISTS source_grants_grantee_idx
    ON source_grants (grantee) WHERE tombstoned_at IS NULL;
CREATE INDEX IF NOT EXISTS source_grants_source_idx
    ON source_grants (source_id) WHERE tombstoned_at IS NULL;
