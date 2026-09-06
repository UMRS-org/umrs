-- 0083: asset grants (ADR-041) — version-pinned sharing with tombstone-on-revoke.
--
-- A grant authorizes one grantee (user or project) to READ one asset
-- (model, knowledge graph, …) outside the asset's static scope tier.
-- Revocation is a tombstone (tombstoned_at), never a DELETE, so the audit
-- trail survives. The live-grant uniqueness index only covers untombstoned
-- rows, so re-granting after a revoke is always possible.

CREATE TABLE IF NOT EXISTS asset_grants (
    grant_id      TEXT PRIMARY KEY,
    asset_kind    TEXT NOT NULL,
    asset_id      TEXT NOT NULL,
    grantee_kind  TEXT NOT NULL,
    grantee_id    TEXT NOT NULL,
    version_pin   TEXT,
    expires_at    TIMESTAMPTZ,
    tombstoned_at TIMESTAMPTZ,
    created_by    TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT asset_grants_grantee_kind_check
        CHECK (grantee_kind IN ('user', 'project')),
    CONSTRAINT asset_grants_asset_kind_check
        CHECK (asset_kind IN ('model', 'knowledge_graph'))
);

-- At most one LIVE grant per (asset, grantee, pin). COALESCE makes NULL pins
-- compare equal so two unpinned live grants cannot coexist; tombstoned rows
-- are excluded so a revoke frees the tuple for re-grant.
CREATE UNIQUE INDEX IF NOT EXISTS asset_grants_live_unique
    ON asset_grants (asset_kind, asset_id, grantee_kind, grantee_id,
                     COALESCE(version_pin, ''))
    WHERE tombstoned_at IS NULL;

CREATE INDEX IF NOT EXISTS asset_grants_asset_idx
    ON asset_grants (asset_kind, asset_id)
    WHERE tombstoned_at IS NULL;

CREATE INDEX IF NOT EXISTS asset_grants_grantee_idx
    ON asset_grants (grantee_kind, grantee_id)
    WHERE tombstoned_at IS NULL;
