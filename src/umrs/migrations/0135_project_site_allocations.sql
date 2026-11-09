-- 0135: project links to a user's own site allocation (ADR-112 D-7, wave B1).
--
-- A user links one of their projects to a Slurm account (allocation) the
-- site lists for them: work for that project at this site is then charged to
-- the user's own allocation (wave B2 runs it). A link exists only while the
-- user's site account (0134) exists; the server creates one only for a live
-- member of the project (ADR-097) and an account in the site account's
-- verified_assocs.
--
-- Keyed (venue, project_id, username): the four fronts share this database.
-- status: active | suspended (account revoked, cascade, allocation gone);
-- suspended_reason says which.

CREATE TABLE IF NOT EXISTS project_site_allocations (
    venue            VARCHAR(32) NOT NULL,
    project_id       TEXT        NOT NULL,
    username         TEXT        NOT NULL,
    slurm_account    TEXT        NOT NULL,
    status           TEXT        NOT NULL DEFAULT 'active',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    suspended_reason TEXT        NOT NULL DEFAULT '',
    PRIMARY KEY (venue, project_id, username),
    CONSTRAINT project_site_allocations_status_check
        CHECK (status IN ('active', 'suspended')),
    CONSTRAINT project_site_allocations_account_fk
        FOREIGN KEY (venue, username)
        REFERENCES site_accounts (venue, username) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS project_site_allocations_venue_user
    ON project_site_allocations (venue, username);
