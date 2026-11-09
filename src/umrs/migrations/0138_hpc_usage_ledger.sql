-- 0138: the HPC usage ledger of user work (ADR-112 wave D, plan §6 C/D.0 D1).
--
-- One row per state of one attempt of user work at a site: a staged workflow
-- step (gated or on the platform login), an hpc_script job or a cluster loop
-- job. Rows are append-only (a trigger refuses UPDATE and DELETE):
-- corrections and later states are new rows, and hpc_usage_current gives
-- each attempt's latest row.
--
-- Two-phase: a 'pending' row (source 'estimate', the requested resources)
-- goes in before the submit, 'submitted' after it (the Slurm job id), and a
-- terminal Slurm state when the job is measured (source 'gate_usage' or
-- 'sacct'). 'not_submitted' says nothing reached the scheduler; 'unsettled'
-- says the job could not be measured within 7 days and is charged at its
-- estimate.
--
-- Amounts are integers: gpu-seconds = alloc_gpus x elapsed_s, core-seconds =
-- alloc_cpus x elapsed_s. The hour columns are those values over 3600, for
-- display. person has no foreign key, so the account cascade never deletes
-- a row (I-11: venue on every row; the four fronts share this database).

CREATE TABLE IF NOT EXISTS hpc_usage_ledger (
    ledger_id        BIGSERIAL PRIMARY KEY,
    venue            VARCHAR(32) NOT NULL,
    attempt_id       TEXT        NOT NULL,
    run_id           TEXT        NOT NULL DEFAULT '',
    step             TEXT        NOT NULL DEFAULT '',
    job_id           TEXT        NOT NULL DEFAULT '',
    scheduler_job_id TEXT        NOT NULL DEFAULT '',
    tag              TEXT        NOT NULL,
    person           TEXT        NOT NULL,
    project_id       TEXT        NOT NULL DEFAULT '',
    site_login       TEXT        NOT NULL DEFAULT '',
    access_mode      TEXT        NOT NULL,
    method           TEXT        NOT NULL,
    slurm_account    TEXT        NOT NULL DEFAULT '',
    intent_id        TEXT        NOT NULL DEFAULT '',
    grant_id         TEXT        NOT NULL DEFAULT '',
    gate_log_seq     BIGINT      NULL,
    lane             TEXT        NOT NULL,
    submitted_at     TIMESTAMPTZ NULL,
    started_at       TIMESTAMPTZ NULL,
    ended_at         TIMESTAMPTZ NULL,
    state            TEXT        NOT NULL,
    alloc_cpus       INT         NOT NULL DEFAULT 0,
    alloc_gpus       INT         NOT NULL DEFAULT 0,
    elapsed_s        BIGINT      NOT NULL DEFAULT 0,
    core_hours       DOUBLE PRECISION NOT NULL DEFAULT 0,
    gpu_hours        DOUBLE PRECISION NOT NULL DEFAULT 0,
    source           TEXT        NOT NULL,
    recorded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT hpc_usage_ledger_access_mode_check
        CHECK (access_mode IN ('service_account', 'user_account')),
    CONSTRAINT hpc_usage_ledger_method_check
        CHECK (method IN ('platform', 'gated_key')),
    CONSTRAINT hpc_usage_ledger_lane_check
        CHECK (lane IN ('staged_step', 'hpc_script', 'cluster_loop')),
    CONSTRAINT hpc_usage_ledger_source_check
        CHECK (source IN ('estimate', 'gate_usage', 'sacct'))
);

CREATE INDEX IF NOT EXISTS hpc_usage_ledger_attempt
    ON hpc_usage_ledger (venue, attempt_id, ledger_id DESC);
CREATE INDEX IF NOT EXISTS hpc_usage_ledger_scheduler_job
    ON hpc_usage_ledger (venue, scheduler_job_id, ledger_id);
CREATE INDEX IF NOT EXISTS hpc_usage_ledger_project
    ON hpc_usage_ledger (project_id);
CREATE INDEX IF NOT EXISTS hpc_usage_ledger_person
    ON hpc_usage_ledger (venue, person);
CREATE INDEX IF NOT EXISTS hpc_usage_ledger_open_estimates
    ON hpc_usage_ledger (venue, state) WHERE source = 'estimate';

CREATE OR REPLACE FUNCTION hpc_usage_ledger_refuse_change() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'hpc_usage_ledger is append-only: % refused', TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS hpc_usage_ledger_append_only ON hpc_usage_ledger;
CREATE TRIGGER hpc_usage_ledger_append_only
    BEFORE UPDATE OR DELETE ON hpc_usage_ledger
    FOR EACH ROW EXECUTE FUNCTION hpc_usage_ledger_refuse_change();

CREATE OR REPLACE VIEW hpc_usage_current AS
    SELECT DISTINCT ON (venue, run_id, attempt_id, step) *
    FROM hpc_usage_ledger
    ORDER BY venue, run_id, attempt_id, step, ledger_id DESC;
