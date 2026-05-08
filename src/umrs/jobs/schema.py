"""
Database schema and migrations for durable job tracking.

Tables:
- jobs: User intent (immutable spec)
- job_attempts: Execution tries (multiple per job)

Indexes:
- jobs(status, next_run_at): Scheduler optimization
- job_attempts(job_id, attempt_no): Attempt lookup
- jobs(idempotency_key): Unique per project/user
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Schema version - increment when schema changes
SCHEMA_VERSION = 2


# SQL for creating the jobs table (U-701)
CREATE_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id VARCHAR(64) PRIMARY KEY,
    project_id VARCHAR(128) NOT NULL,
    user_id VARCHAR(128) NOT NULL,
    job_type VARCHAR(32) NOT NULL,
    spec JSONB NOT NULL,
    idempotency_key VARCHAR(64) NOT NULL,

    status VARCHAR(32) NOT NULL DEFAULT 'planned',
    priority INTEGER NOT NULL DEFAULT 0,
    desired_max_attempts INTEGER NOT NULL DEFAULT 3,
    restart_policy VARCHAR(32) NOT NULL DEFAULT 'on_crash',

    next_run_at TIMESTAMPTZ,
    active_attempt_id VARCHAR(64),
    commit_attempt_id VARCHAR(64),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_state_change_at TIMESTAMPTZ,
    last_heartbeat_at TIMESTAMPTZ
);

-- Comments for documentation
COMMENT ON TABLE jobs IS 'Durable job tracking - user intent (immutable spec)';
COMMENT ON COLUMN jobs.job_id IS 'Unique job identifier (job-xxxxxxxxxxxx)';
COMMENT ON COLUMN jobs.spec IS 'Job-specific parameters (JSONB)';
COMMENT ON COLUMN jobs.idempotency_key IS 'Prevents duplicate jobs from client retries';
COMMENT ON COLUMN jobs.active_attempt_id IS 'Currently executing attempt (if any)';
COMMENT ON COLUMN jobs.commit_attempt_id IS 'Attempt whose outputs are "current" (commit pointer)';
"""


# SQL for creating the job_attempts table (U-702)
CREATE_ATTEMPTS_TABLE = """
CREATE TABLE IF NOT EXISTS job_attempts (
    attempt_id VARCHAR(64) PRIMARY KEY,
    job_id VARCHAR(64) NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    attempt_no INTEGER NOT NULL,

    backend VARCHAR(32) NOT NULL,
    backend_ref VARCHAR(256),

    status VARCHAR(32) NOT NULL DEFAULT 'created',

    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,

    exit_reason TEXT,
    log_uri VARCHAR(512),

    metrics JSONB DEFAULT '{}',
    umrs_instance_id VARCHAR(64),
    scheduler_claimed_at TIMESTAMPTZ,
    status_detail VARCHAR(256),

    ignored_at TIMESTAMPTZ,
    ignore_reason VARCHAR(256),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(job_id, attempt_no)
);

COMMENT ON TABLE job_attempts IS 'Durable job tracking - execution attempts (multiple per job)';
COMMENT ON COLUMN job_attempts.attempt_no IS '1-based attempt number within job';
COMMENT ON COLUMN job_attempts.backend_ref IS 'Backend-specific ID (Slurm job ID, K8s Job name)';
COMMENT ON COLUMN job_attempts.umrs_instance_id IS 'Which UMRS instance started this attempt';
COMMENT ON COLUMN job_attempts.status_detail IS 'Human-readable progress (e.g., "polling readiness")';
COMMENT ON COLUMN job_attempts.ignored_at IS 'When attempt was marked LOST (UMRS restart)';
"""


# SQL for creating the job_locks table (U-703) - HA distributed locking
CREATE_JOB_LOCKS_TABLE = """
CREATE TABLE IF NOT EXISTS job_locks (
    lock_id VARCHAR(64) PRIMARY KEY,
    lock_type VARCHAR(32) NOT NULL,
    resource_id VARCHAR(256) NOT NULL,

    owner_instance_id VARCHAR(64) NOT NULL,
    owner_hostname VARCHAR(256),

    acquired_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    metadata JSONB DEFAULT '{}',

    UNIQUE(lock_type, resource_id)
);

COMMENT ON TABLE job_locks IS 'Distributed locks for HA UMRS cluster coordination';
COMMENT ON COLUMN job_locks.lock_type IS 'Lock category: scheduler, job, resource';
COMMENT ON COLUMN job_locks.resource_id IS 'What is locked (job_id, gpu_node, etc.)';
COMMENT ON COLUMN job_locks.owner_instance_id IS 'UMRS instance UUID holding the lock';
COMMENT ON COLUMN job_locks.expires_at IS 'Auto-release if heartbeat not updated';
COMMENT ON COLUMN job_locks.heartbeat_at IS 'Last heartbeat from lock owner';

-- Index for finding expired locks
CREATE INDEX IF NOT EXISTS idx_job_locks_expires
    ON job_locks(expires_at);

-- Index for finding locks by owner (cleanup on instance death)
CREATE INDEX IF NOT EXISTS idx_job_locks_owner
    ON job_locks(owner_instance_id);
"""


# SQL for creating the umrs_instances table (U-715) - Heartbeat mechanism
CREATE_UMRS_INSTANCES_TABLE = """
CREATE TABLE IF NOT EXISTS umrs_instances (
    instance_id VARCHAR(64) PRIMARY KEY,
    hostname VARCHAR(256) NOT NULL,
    pid INTEGER,

    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    status VARCHAR(32) NOT NULL DEFAULT 'active',

    capabilities JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',

    CHECK (status IN ('active', 'draining', 'dead'))
);

COMMENT ON TABLE umrs_instances IS 'UMRS instance registry for HA coordination';
COMMENT ON COLUMN umrs_instances.instance_id IS 'Unique UUID for this UMRS process';
COMMENT ON COLUMN umrs_instances.last_heartbeat_at IS 'Updated periodically to prove liveness';
COMMENT ON COLUMN umrs_instances.status IS 'active (normal), draining (shutting down), dead (missed heartbeats)';
COMMENT ON COLUMN umrs_instances.capabilities IS 'What this instance can do (scheduler, monitor, etc.)';

-- Index for finding dead instances
CREATE INDEX IF NOT EXISTS idx_umrs_instances_heartbeat
    ON umrs_instances(last_heartbeat_at)
    WHERE status = 'active';
"""


# SQL for creating indexes (U-705, U-706, U-707)
CREATE_INDEXES = """
-- U-705: Scheduler optimization - find QUEUED jobs ready to run
CREATE INDEX IF NOT EXISTS idx_jobs_scheduler
    ON jobs(status, next_run_at)
    WHERE status = 'queued';

-- U-706: Attempt lookup by job
CREATE INDEX IF NOT EXISTS idx_attempts_job
    ON job_attempts(job_id, attempt_no);

-- U-707: Idempotency key uniqueness (scoped by project)
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency
    ON jobs(project_id, idempotency_key);

-- Additional useful indexes
CREATE INDEX IF NOT EXISTS idx_jobs_project
    ON jobs(project_id, status);

CREATE INDEX IF NOT EXISTS idx_jobs_type
    ON jobs(job_type, status);

CREATE INDEX IF NOT EXISTS idx_attempts_status
    ON job_attempts(status)
    WHERE status IN ('submitted', 'running');
"""


# SQL for update trigger (auto-update updated_at)
CREATE_UPDATE_TRIGGER = """
CREATE OR REPLACE FUNCTION update_jobs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS jobs_updated_at_trigger ON jobs;
CREATE TRIGGER jobs_updated_at_trigger
    BEFORE UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_jobs_updated_at();
"""


# Schema version tracking table
CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_versions (
    component VARCHAR(64) PRIMARY KEY,
    version INTEGER NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def ensure_schema(conn) -> bool:
    """
    Ensure job tracking schema exists.

    Creates tables and indexes if they don't exist.
    Returns True if schema was created/updated.
    """
    try:
        with conn.cursor() as cur:
            # Check if we need to migrate
            cur.execute(CREATE_SCHEMA_VERSION_TABLE)

            cur.execute("""
                SELECT version FROM schema_versions WHERE component = 'jobs'
            """)
            row = cur.fetchone()
            current_version = row[0] if row else 0

            if current_version >= SCHEMA_VERSION:
                logger.debug(f"Job schema already at version {current_version}")
                return False

            logger.info(f"Creating/updating job schema to version {SCHEMA_VERSION}")

            # Create tables
            cur.execute(CREATE_JOBS_TABLE)
            cur.execute(CREATE_ATTEMPTS_TABLE)

            # Create HA tables (U-703, U-715)
            cur.execute(CREATE_JOB_LOCKS_TABLE)
            cur.execute(CREATE_UMRS_INSTANCES_TABLE)

            # Create indexes
            cur.execute(CREATE_INDEXES)

            # Create update trigger
            cur.execute(CREATE_UPDATE_TRIGGER)

            # Update schema version
            cur.execute("""
                INSERT INTO schema_versions (component, version)
                VALUES ('jobs', %s)
                ON CONFLICT (component) DO UPDATE SET version = %s, applied_at = NOW()
            """, (SCHEMA_VERSION, SCHEMA_VERSION))

            conn.commit()
            logger.info(f"Job schema updated to version {SCHEMA_VERSION}")
            return True

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to ensure job schema: {e}")
        raise


def drop_schema(conn, confirm: bool = False) -> None:
    """
    Drop job tracking schema.

    USE WITH CAUTION - this deletes all job data!
    """
    if not confirm:
        raise ValueError("Must set confirm=True to drop schema")

    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS job_attempts CASCADE")
            cur.execute("DROP TABLE IF EXISTS jobs CASCADE")
            cur.execute("DELETE FROM schema_versions WHERE component = 'jobs'")
            conn.commit()
            logger.warning("Job schema dropped")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to drop job schema: {e}")
        raise


def get_schema_version(conn) -> int:
    """Get current schema version, or 0 if not initialized."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT version FROM schema_versions WHERE component = 'jobs'
            """)
            row = cur.fetchone()
            return row[0] if row else 0
    except Exception:
        return 0
