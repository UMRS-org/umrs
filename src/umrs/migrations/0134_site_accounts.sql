-- 0134: user-registered HPC site accounts (ADR-112 D-3/D-7, wave B1).
--
-- One row per person per site: the login a user registered at this site, the
-- transport key the server generated for them (its id and public fingerprint
-- only; the private half is a file under UMRS_TRANSPORT_KEY_DIR, so a dump
-- or backup of this table carries no key), and what the site itself reported
-- through the user's gate at verify (verified_*).
--
-- Keyed (venue, username): every umrs-front shares this database, and each
-- front reads and writes only its own venue's rows (as users, 0113).
--
-- status:     pending (key issued, gate not verified) | active (verified)
--             | suspended | revoked (transport key destroyed)
-- gate_probe: refused (the gate itself refused the probe) | bypassed
--             (something else answered: a site ForceCommand) | unknown
-- verified_assocs: [{cluster, account, partition, qos}] from the site's own
--             sacctmgr, as the gate's hello reported it.

CREATE TABLE IF NOT EXISTS site_accounts (
    venue                 VARCHAR(32) NOT NULL,
    username              TEXT        NOT NULL,
    site_login            TEXT        NOT NULL DEFAULT '',
    method                TEXT        NOT NULL DEFAULT 'gated_key',
    status                TEXT        NOT NULL DEFAULT 'pending',
    verified_login        TEXT        NOT NULL DEFAULT '',
    verified_assocs       JSONB       NOT NULL DEFAULT '[]'::jsonb,
    verified_at           TIMESTAMPTZ NULL,
    gate_version          TEXT        NOT NULL DEFAULT '',
    gate_sha256           TEXT        NOT NULL DEFAULT '',
    gate_probe            TEXT        NOT NULL DEFAULT 'unknown',
    transport_key_id      TEXT        NOT NULL DEFAULT '',
    transport_fingerprint TEXT        NOT NULL DEFAULT '',
    signer_fingerprints   JSONB       NOT NULL DEFAULT '[]'::jsonb,
    uv_required           BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (venue, username),
    CONSTRAINT site_accounts_method_check CHECK (method IN ('gated_key')),
    CONSTRAINT site_accounts_status_check
        CHECK (status IN ('pending', 'active', 'suspended', 'revoked')),
    CONSTRAINT site_accounts_gate_probe_check
        CHECK (gate_probe IN ('refused', 'bypassed', 'unknown'))
);

CREATE INDEX IF NOT EXISTS site_accounts_venue_status
    ON site_accounts (venue, status);
