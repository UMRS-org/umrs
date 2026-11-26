-- 0144 (slice S-D39a): catalog origin identity for bundle federation.
--
-- DEC-D39 (decisions-2026-11-24 §2): a bundle carries the full catalog
-- identity, and an import records where the catalog came from:
--
--   origin_catalog_id   the source installation's catalog_id ('' = made here)
--   origin_fingerprint  the source installation's fingerprint ('' = made here)
--
-- Both are written once, by the bundle import only (the create route clears
-- them), and are never patched: a version row stays immutable.
--
-- umrs_installation holds one random id per installation (database). The
-- export publishes only its hash (umrs.knowledge.catalog_origin
-- .installation_fingerprint), never the id, a hostname or an address. It is
-- the same on a hosted front and on a self-hosted stand-alone UMRS. A cloned
-- database keeps the id of the database it was cloned from.
--
-- Down (manual):
--   DROP INDEX IF EXISTS scientific_catalogs_origin_idx;
--   ALTER TABLE scientific_catalogs DROP COLUMN IF EXISTS origin_fingerprint;
--   ALTER TABLE scientific_catalogs DROP COLUMN IF EXISTS origin_catalog_id;
--   DROP TABLE IF EXISTS umrs_installation;

ALTER TABLE scientific_catalogs
    ADD COLUMN IF NOT EXISTS origin_catalog_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS origin_fingerprint TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS scientific_catalogs_origin_idx
    ON scientific_catalogs (project_id, origin_catalog_id)
    WHERE origin_catalog_id <> '';

COMMENT ON COLUMN scientific_catalogs.origin_catalog_id IS
    'DEC-D39: catalog_id at the installation this row was imported from; '
    'empty when the catalog was made here.';
COMMENT ON COLUMN scientific_catalogs.origin_fingerprint IS
    'DEC-D39: fingerprint (inst-<32 hex>) of the installation this row was '
    'imported from; empty when the catalog was made here.';

CREATE TABLE IF NOT EXISTS umrs_installation (
    singleton       BOOLEAN     PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    installation_id TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO umrs_installation (singleton, installation_id)
VALUES (TRUE, gen_random_uuid()::text)
ON CONFLICT (singleton) DO NOTHING;

COMMENT ON TABLE umrs_installation IS
    'One random id per installation (DEC-D39). Only its hash leaves the '
    'installation, as the bundle origin fingerprint.';
