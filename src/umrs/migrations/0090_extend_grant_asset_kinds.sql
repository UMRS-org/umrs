-- 0090: extend asset_grants asset_kind CHECK (WP-06 round 2 leg-b).
--
-- Waves 1 kept grants model/knowledge_graph-only (binding is owner-writable
-- scoped). Round 2 adds two kinds whose owners are discoverable through the
-- same catalog chain:
--   - endpoint (registry rebuild by (project_id, job_id enumeration); write
--     [sharing] means caller's project matches the provisioning project).
--   - workflow (workflow registry catalog entries; sharing means the caller
--     OWNs the registration (same admin path as workflow_delete).
-- 'bundle' is intentionally NOT granted this window: there is no durable
-- per-bundle owner record today (fabricating one would lie about
-- provenance). New migrations (CHECK drops + adds adjacent) give the kind a
-- place when durable bundle ownership rows land.
--
-- Postgres supports no ALTER CHECK; the pattern here is drop + re-add —
-- safe add-on (existing rows' kinds remain a strict subset).

ALTER TABLE asset_grants
    DROP CONSTRAINT IF EXISTS asset_grants_asset_kind_check;

ALTER TABLE asset_grants
    ADD CONSTRAINT asset_grants_asset_kind_check
    CHECK (asset_kind IN ('model', 'knowledge_graph', 'endpoint', 'workflow'));

COMMENT ON CONSTRAINT asset_grants_asset_kind_check ON asset_grants IS
    'WP-06 r2: bundles stays deferred pending durable bundle-owner rows; the
     other two kinds gate projects/owners through the catalog chain.';
