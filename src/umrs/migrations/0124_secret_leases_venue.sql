-- 0124: secret_leases.venue — per-site scoping of broker leases (RS-0 S-2,
-- remote-session platform plan 2026-10-22; 0088/0113 lineage).
--
-- Every umrs-front server (rccs/ai4s/bdrt/octopus) writes leases into the
-- SHARED umrs_db, and user accounts are per site (0113). The account cascade
-- revokes every lease a user owns when that site disables the account or
-- removes it from a project; keyed by owner alone it would also end the same
-- username's sessions on the other sites, and that site's standing sweep would
-- keep ending them.
--
--   venue = '<site.name>'  → issued by that site's server
--   venue = ''             → issued before 0124, or a single-stack deployment
--
-- Non-destructive: existing rows keep venue = ''. The cascade treats '' as
-- possibly its own (a pre-0124 lease of a disabled account must still end);
-- sessions last at most 7 days, so '' rows age out.

ALTER TABLE secret_leases
    ADD COLUMN IF NOT EXISTS venue TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_secret_leases_owner_venue
    ON secret_leases (owner_id, venue) WHERE NOT is_revoked;

COMMENT ON COLUMN secret_leases.venue IS
    'Site whose server issued the lease (site.name; 0088 convention). '
    'Empty = issued before 0124 or a single-stack deployment.';
