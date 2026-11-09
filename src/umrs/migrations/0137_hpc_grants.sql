-- 0137: standing permissions (grants) for runs under the person's own HPC
-- account (ADR-112 wave C, plan §6 C/D.0 C4).
--
-- A grant is a document the person signed (namespace umrs-grant-v1@<domain>)
-- that lets UMRS start runs of the listed workflow versions, within caps and
-- until it expires, without a signature per run. The person's gate is the
-- authority: it verifies the signature and keeps the counters at every
-- submit. This table is the server's mirror: the signed bytes (payload_b64,
-- signature) so every step can hand them to the gate, plus status and a
-- display copy of the gate's usage (grant-check).
--
-- Keyed (venue, grant_id): every umrs-front shares this database (I-11). A
-- grant belongs to one site account and goes with it (ON DELETE CASCADE).
--
-- status:       active | cancelled | expired
-- ended_reason: '' | cancelled | account_revoked | expired
-- gate_cancel:  '' | pending (the gate has not recorded the cancel yet)
--               | recorded
--
-- workflow_executions gains the grant columns of a grant-backed run: the
-- grant it runs under, the server-composed run spec (unsigned; base64 of its
-- canonical bytes and their sha256) and the step skeleton the grant pins.

CREATE TABLE IF NOT EXISTS hpc_grants (
    venue        VARCHAR(32) NOT NULL,
    grant_id     CHAR(32)    NOT NULL,
    username     TEXT        NOT NULL,
    site_login   TEXT        NOT NULL,
    payload_b64  TEXT        NOT NULL,
    signature    TEXT        NOT NULL,
    grant_sha256 CHAR(64)    NOT NULL,
    projects     JSONB       NOT NULL DEFAULT '[]'::jsonb,
    accounts     JSONB       NOT NULL DEFAULT '[]'::jsonb,
    workflows    JSONB       NOT NULL DEFAULT '[]'::jsonb,
    inputs_mode  TEXT        NOT NULL DEFAULT 'listed',
    caps         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    status       TEXT        NOT NULL DEFAULT 'active',
    ended_reason TEXT        NOT NULL DEFAULT '',
    gate_cancel  TEXT        NOT NULL DEFAULT '',
    not_before   TIMESTAMPTZ NULL,
    expires_at   TIMESTAMPTZ NOT NULL,
    usage        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    usage_at     TIMESTAMPTZ NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (venue, grant_id),
    CONSTRAINT hpc_grants_status_check
        CHECK (status IN ('active', 'cancelled', 'expired')),
    CONSTRAINT hpc_grants_ended_reason_check
        CHECK (ended_reason IN ('', 'cancelled', 'account_revoked', 'expired')),
    CONSTRAINT hpc_grants_gate_cancel_check
        CHECK (gate_cancel IN ('', 'pending', 'recorded')),
    CONSTRAINT hpc_grants_inputs_mode_check
        CHECK (inputs_mode IN ('listed', 'chained', 'any')),
    CONSTRAINT hpc_grants_account_fk
        FOREIGN KEY (venue, username)
        REFERENCES site_accounts (venue, username) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS hpc_grants_venue_user_status
    ON hpc_grants (venue, username, status);

ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS grant_id CHAR(32);
ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS run_spec_b64 TEXT;
ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS run_spec_id CHAR(64);
ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS workflow_steps_b64 TEXT;
CREATE INDEX IF NOT EXISTS workflow_executions_grant_id
    ON workflow_executions (grant_id) WHERE grant_id IS NOT NULL;
