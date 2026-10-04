-- 0119: circles — the project-scoped context + audience cell (ADR-086).
--
-- Supersedes the 0097 platform-global groups authority: a circle lives
-- INSIDE one project ((project_id, circle_id) keyed), bounds the contexts
-- of works born in it, and is the smallest addressable sharing audience
-- (grammar kind `circle:<project_id>/<circle_id>`, ADR-080). Legacy
-- `group:<id>` entries stop resolving — the 0097 tables stay for audit.
--
-- Semantics carried over from 0097: circles are owner-attributed and
-- tombstoned (retire never DELETEs; a retired circle stops resolving
-- immediately); membership rows are presence-only (no tombstones).
--
-- New vs 0097:
-- - project-scoped: a circle always names its containing project
--   (FK projects.project_id); project affiliation derives from circle
--   membership — no users.projects list.
-- - roles: membership role widens to ('member', 'admin') — circle admins
--   manage the roster beside the owner (the 0097 column was the seam).
-- - personal circle: circle_id == owner_id marks the caller's private
--   context cell in that project; it never gains other members.

CREATE TABLE IF NOT EXISTS circles (
    project_id  TEXT NOT NULL REFERENCES projects(project_id),
    circle_id   TEXT NOT NULL,
    name        TEXT NOT NULL DEFAULT '',
    owner_id    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    retired_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project_id, circle_id),
    CONSTRAINT circles_status_check
        CHECK (status IN ('active', 'retired'))
);

CREATE TABLE IF NOT EXISTS circle_members (
    project_id  TEXT NOT NULL,
    circle_id   TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'member',
    added_by    TEXT NOT NULL,
    added_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project_id, circle_id, user_id),
    FOREIGN KEY (project_id, circle_id)
        REFERENCES circles(project_id, circle_id),
    CONSTRAINT circle_members_role_check
        CHECK (role IN ('member', 'admin'))
);

CREATE INDEX IF NOT EXISTS circle_members_user_idx
    ON circle_members (user_id, project_id, circle_id);

COMMENT ON TABLE circles IS
    'Project-scoped context/audience cells (ADR-086). Owner-attributed; retire is a tombstone, never a DELETE. Retired circles stop resolving for circle-aware access lookups immediately. Supersedes the 0097 groups authority.';
COMMENT ON TABLE circle_members IS
    'Circle membership (ADR-086). Presence rows only — no tombstones. role: member | admin; circle admins manage the roster beside the owner. A circle with circle_id = owner_id is the personal cell: it never gains other members.';
