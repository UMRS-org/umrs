-- 0125: session policy rules + the resolved policy of each remote session
-- (ADR-099 D-1, D-3; remote-session platform plan wave RS-1, lane P).
--
-- session_policy_rules: partial policy documents keyed by
--   venue   = '<site.name>' (that site) or '' (every site; a site-named row
--             beats '' at the same key — sites share umrs_db, 0113/0124)
--   scope   = site | project | user
--   subject = '' for site, else the project id or the user id
--   layer   = admin (any scope) | owner (project scope: the project owner's
--             narrowing layer)
-- The resolver (umrs.session_policy.resolve) folds them over the site
-- defaults; the inference proxy reads them (umrs_infer: SELECT) for the
-- token and rate quotas (D-5).
--
-- companion_venue_sessions.policy: the resolved policy computed at start
-- (limits + the user's web choice + the origin of each value). '{}' for
-- sessions started before 0125.
--
-- Non-destructive: one new table, one new column with a default.

CREATE TABLE IF NOT EXISTS session_policy_rules (
    venue      TEXT        NOT NULL DEFAULT '',
    scope      TEXT        NOT NULL CHECK (scope IN ('site', 'project', 'user')),
    subject    TEXT        NOT NULL DEFAULT '',
    layer      TEXT        NOT NULL CHECK (layer IN ('admin', 'owner')),
    doc        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    set_by     TEXT        NOT NULL DEFAULT '',
    note       TEXT        NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (venue, scope, subject, layer),
    CONSTRAINT session_policy_rules_owner_is_project
        CHECK (layer = 'admin' OR scope = 'project'),
    CONSTRAINT session_policy_rules_site_subject
        CHECK (scope <> 'site' OR subject = '')
);

COMMENT ON TABLE session_policy_rules IS
    'ADR-099 D-3 limit rules: admin (any scope) and project-owner layers; '
    'venue '''' applies to every site.';

ALTER TABLE companion_venue_sessions
    ADD COLUMN IF NOT EXISTS policy JSONB NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN companion_venue_sessions.policy IS
    'Resolved session policy at start (ADR-099 D-1): limits, web choice, origin.';

-- The inference proxy (umrs_infer, 0086) reads the rules for its quotas.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'umrs_infer') THEN
        EXECUTE 'GRANT SELECT ON session_policy_rules TO umrs_infer';
    END IF;
END
$$;
