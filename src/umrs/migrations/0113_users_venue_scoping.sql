-- 0113: users.venue — per-site scoping of the D-06 accounts lane (0088 lineage).
--
-- Every umrs-front server (rccs/ai4s/bdrt/octopus) resolves the SHARED
-- umrs_db, but each keeps its own JSON user directory. Under 0110 the table
-- was keyed by username alone: each server's boot sync overwrote the others'
-- rows (last boot wins) and PG-first reads then served another cluster's
-- role / disabled flag / slurm_account / CURVE key. Found before the first
-- deploy of 0110 to umrs-front (2026-09-24).
--
--   venue = '<site.name>'  → the row belongs to that site's user directory
--   venue = ''             → single-stack deployment without a site name
--
-- Non-destructive: pre-0113 rows keep venue = '' (still unique — username
-- was the old key). A named site never reads them (reads filter on its own
-- venue) and its next boot sync writes its rows from its own JSON files; an
-- unnamed single-stack site keeps reading them unchanged.

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS venue VARCHAR(32) NOT NULL DEFAULT '';

ALTER TABLE users DROP CONSTRAINT IF EXISTS users_pkey;
ALTER TABLE users ADD CONSTRAINT users_pkey PRIMARY KEY (venue, username);

DROP INDEX IF EXISTS idx_users_oidc_subject;
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_venue_oidc_subject
    ON users (venue, oidc_subject) WHERE oidc_subject <> '';

COMMENT ON COLUMN users.venue IS
    'Site whose user directory owns this row (site.name; 0088 convention). '
    'Empty = single-stack deployment without a site name.';
