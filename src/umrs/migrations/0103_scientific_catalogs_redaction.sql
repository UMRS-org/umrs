-- 0103: Scientific Catalog export redaction rules (gap row D-46, Phase 1).
--
-- Which parts of a catalog must never leave the site is a property of the
-- catalog record, not of the export call. `redaction_rules` is a JSONB array
-- of {"target": <manifest field>, "action": "drop"|"hash"} rules the bundle
-- export applies to the manifest metadata blocks (catalog / ownership /
-- provenance / contributors / licence). "drop" empties the field, "hash"
-- replaces strings with sha256 fingerprints (12-char, traceable without
-- content). Row/table-level redaction is a later leg; the integrity hash
-- always covers the REDACTED manifest, so bundle signatures stay honest.

ALTER TABLE scientific_catalogs
    ADD COLUMN IF NOT EXISTS redaction_rules JSONB NOT NULL DEFAULT '[]'::jsonb;

COMMENT ON COLUMN scientific_catalogs.redaction_rules IS
    'Export redaction rules (D-46): [{"target": manifest_field, "action": "drop"|"hash"}] applied to manifest metadata at bundle export.';
