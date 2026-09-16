-- 0097: groups + membership authority (ADR-050 §3 groundwork).
--
-- The scope grammar spoke group:<id> from day one, and ADR-041-style
-- grants named it as a grantee — but no store of groups or membership
-- existed, so group resolution had no grounds. This wave creates it:
-- groups are owner-attributed and tombstoned (never DELETEd); membership
-- rows are not tombstoned (membership is presence, not history).
--
-- groups.status: 'active' | 'retired' — a retired group's grants stop
-- resolving immediately (checked group-aware querylanes look
-- status='active' only), the row survives for audit.

CREATE TABLE IF NOT EXISTS groups (
    group_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT '',
    owner_id    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    retired_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT groups_status_check
        CHECK (status IN ('active', 'retired'))
);

CREATE TABLE IF NOT EXISTS group_members (
    group_id    TEXT NOT NULL REFERENCES groups(group_id),
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'member',
    added_by    TEXT NOT NULL,
    added_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (group_id, user_id),
    CONSTRAINT group_members_role_check
        CHECK (role IN ('member'))
);

CREATE INDEX IF NOT EXISTS group_members_user_idx
    ON group_members (user_id, group_id);

COMMENT ON TABLE groups IS
    'Durable group registry (ADR-050 §3). Owner-attributed; retire is a tombstone, never a DELETE. Retired groups stop resolving for group-aware grant lookups immediately.';

COMMENT ON TABLE group_members IS
    'Group membership (ADR-050 §3). Presence rows only — no tombstones: removing membership is a DELETE of the presence, history is the audit trail elsewhere.';
