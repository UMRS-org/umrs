import json
from datetime import datetime
from datetime import timezone as _tz

from umrs.jobs.models import AttemptStatus, Job, JobAttempt, JobStatus, JobType, RestartPolicy
from umrs.jobs.store import InvalidTransitionError
from umrs.server import job_tracking_handler
from umrs.server.job_tracking_handler import DurableJobCommandBackend, handle_job_command

JOB_SUBMIT = "JOB_SUBMIT"
JOB_LIST = "JOB_LIST"


def _job_matches(job, wanted: dict) -> bool:
    """Truthy filters must equal the job's attribute (falsy filters are ignored)."""
    return all(not value or getattr(job, key) == value for key, value in wanted.items())


class FakeJobStore:
    def __init__(self):
        self.jobs = {}
        self.attempts = {}
        self.enqueued = []
        self.cancelled = []
        self.idempotency = {}

    def create_job(
        self,
        project_id,
        user_id,
        job_type,
        spec,
        idempotency_key=None,
        priority=0,
        max_attempts=3,
        restart_policy=RestartPolicy.ON_CRASH,
    ):
        idem = idempotency_key or f"idem-{project_id}-{user_id}-{job_type.value}-{spec}"
        if idem in self.idempotency:
            raise job_tracking_handler.DuplicateJobError(self.idempotency[idem])

        job_id = f"job-{len(self.jobs) + 1:03d}"
        job = Job(
            job_id=job_id,
            project_id=project_id,
            user_id=user_id,
            job_type=job_type,
            spec=spec,
            idempotency_key=idem,
            status=JobStatus.PLANNED,
            priority=priority,
            desired_max_attempts=max_attempts,
            restart_policy=restart_policy,
            created_at=datetime.now(_tz.utc),
            updated_at=datetime.now(_tz.utc),
        )
        self.jobs[job_id] = job
        self.attempts[job_id] = []
        self.idempotency[idem] = job_id
        return job_id

    def enqueue_job(self, job_id):
        self.enqueued.append(job_id)
        self.jobs[job_id].status = JobStatus.QUEUED

    def get_job(self, job_id):
        return self.jobs[job_id]

    def get_attempts_for_job(self, job_id):
        return self.attempts[job_id]

    def list_jobs(self, project_id=None, status=None, job_type=None, limit=100, offset=0):
        wanted = {"project_id": project_id, "status": status, "job_type": job_type}
        jobs = [job for job in self.jobs.values() if _job_matches(job, wanted)]
        return jobs[offset : offset + limit]

    def cancel_job(self, job_id, reason="user_cancelled"):
        if self.jobs[job_id].status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            raise InvalidTransitionError(f"Cannot cancel job {job_id} (already terminal)")
        self.cancelled.append((job_id, reason))
        self.jobs[job_id].status = JobStatus.CANCELLED


def test_durable_backend_submit_returns_serialized_job():
    backend = DurableJobCommandBackend(FakeJobStore())

    result = json.loads(
        backend.submit(
            "testuser",
            json.dumps(
                {
                    "project_id": "default",
                    "job_type": "endpoint_provision",
                    "spec": {"model": "llama-3"},
                    "priority": 5,
                    "max_attempts": 4,
                }
            ),
        )
    )

    assert result["status"] == "ok"
    assert result["job"]["job_type"] == "endpoint_provision"
    assert result["job"]["status"] == "queued"
    assert result["job"]["attempts"] == []


def test_durable_backend_list_includes_attempts():
    store = FakeJobStore()
    job_id = store.create_job(
        project_id="default",
        user_id="testuser",
        job_type=JobType.ENDPOINT_PROVISION,
        spec={"model": "llama-3"},
    )
    store.jobs[job_id].status = JobStatus.RUNNING
    store.attempts[job_id].append(
        JobAttempt(
            attempt_id="att-123",
            job_id=job_id,
            attempt_no=1,
            backend="slurm",
            status=AttemptStatus.RUNNING,
            started_at=datetime.now(_tz.utc),
        )
    )
    backend = DurableJobCommandBackend(store)

    result = json.loads(backend.list_jobs("testuser", json.dumps({"status": "running"})))

    assert result["status"] == "ok"
    assert len(result["jobs"]) == 1
    assert result["jobs"][0]["attempts"][0]["attempt_id"] == "att-123"


def test_durable_backend_cancel_uses_job_store():
    store = FakeJobStore()
    job_id = store.create_job(
        project_id="default",
        user_id="testuser",
        job_type=JobType.CLEANUP,
        spec={},
    )
    backend = DurableJobCommandBackend(store)

    result = json.loads(backend.cancel("testuser", json.dumps({"job_id": job_id})))

    assert result["status"] == "ok"
    assert store.cancelled == [(job_id, "cancelled by testuser")]


def test_durable_backend_cancel_completed_endpoint_submits_cleanup_job():
    store = FakeJobStore()
    job_id = store.create_job(
        project_id="demo11",
        user_id="testuser",
        job_type=JobType.ENDPOINT_PROVISION,
        spec={"model": "sam3"},
    )
    store.jobs[job_id].status = JobStatus.COMPLETED
    store.attempts[job_id].append(
        JobAttempt(
            attempt_id="att-123",
            job_id=job_id,
            attempt_no=1,
            backend="slurm",
            backend_ref="backend-123",
            status=AttemptStatus.SUCCEEDED,
            started_at=datetime.now(_tz.utc),
            ended_at=datetime.now(_tz.utc),
            created_at=datetime.now(_tz.utc),
        )
    )
    backend = DurableJobCommandBackend(store)

    result = json.loads(backend.cancel("testuser", json.dumps({"job_id": job_id})))

    assert result["status"] == "ok"
    assert result["target_job_id"] == job_id
    assert result["job"]["job_type"] == "cleanup"
    assert result["job"]["spec"] == {"target": "endpoint", "job_id": job_id}
    assert store.enqueued[-1] == result["job"]["job_id"]


def test_handle_job_command_requires_durable_backend():
    result = json.loads(handle_job_command(JOB_LIST, "testuser", "{}"))

    assert result["status"] == "error"
    assert result["message"] == "Durable job backend unavailable"


def test_handle_job_command_dispatches_to_durable_backend():
    backend = DurableJobCommandBackend(FakeJobStore())

    result = json.loads(
        handle_job_command(
            JOB_SUBMIT,
            "testuser",
            json.dumps(
                {
                    "project_id": "default",
                    "job_type": "endpoint_provision",
                    "spec": {"model": "llama-3"},
                }
            ),
            backend=backend,
        )
    )

    assert result["status"] == "ok"
    assert result["job"]["status"] == "queued"


def test_durable_backend_rejects_invalid_index_build_spec():
    backend = DurableJobCommandBackend(FakeJobStore())

    result = json.loads(
        backend.submit(
            "testuser",
            json.dumps(
                {
                    "project_id": "default",
                    "job_type": "index_build",
                    "spec": {"table": "embeddings"},
                }
            ),
        )
    )

    assert result["status"] == "error"
    assert "index_build spec.database is required" in result["message"]


def test_durable_backend_rejects_unsupported_cleanup_target():
    backend = DurableJobCommandBackend(FakeJobStore())

    result = json.loads(
        backend.submit(
            "testuser",
            json.dumps(
                {
                    "project_id": "default",
                    "job_type": "cleanup",
                    "spec": {"target": "s3", "bucket": "old-experiment"},
                }
            ),
        )
    )

    assert result["status"] == "error"
    assert "cleanup currently supports only spec.target='endpoint'" in result["message"]


def test_build_durable_job_db_pool_reads_env(monkeypatch):
    connect_kwargs = {}
    pool_kwargs = {}
    sentinel_pool = object()

    class FakeAdminCursor:
        def __init__(self):
            self._fetchone = None

        def execute(self, _sql, params=None):
            if params:
                self._fetchone = None

        def fetchone(self):
            return self._fetchone

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeAdminConnection:
        def __init__(self):
            self.autocommit = False
            self.closed = False
            self.cursor_obj = FakeAdminCursor()

        def cursor(self):
            return self.cursor_obj

        def close(self):
            self.closed = True

    admin_conn = FakeAdminConnection()

    def fake_connect(**kwargs):
        connect_kwargs.update(kwargs)
        return admin_conn

    def fake_threaded_pool(**kwargs):
        pool_kwargs.update(kwargs)
        return sentinel_pool

    monkeypatch.setenv("DB_HOST", "db-host")
    monkeypatch.setenv("DB_PORT", "15432")
    monkeypatch.setenv("UMRS_JOBS_DB", "jobs-db")
    monkeypatch.setenv("DB_SUPERUSER", "jobs-user")
    monkeypatch.setenv("DB_SUPERUSER_PASSWORD", "jobs-pass")
    monkeypatch.setattr(job_tracking_handler.psycopg2, "connect", fake_connect)
    monkeypatch.setattr(job_tracking_handler.pg_pool, "ThreadedConnectionPool", fake_threaded_pool)

    pool = job_tracking_handler.build_durable_job_db_pool_from_env()

    assert pool is sentinel_pool
    assert admin_conn.closed is True
    # R-14 (2026-09-13): every construction site carries the session-hygiene
    # kwargs from umrs.shared.pg_settings (application_name + idle-in-tx reaper).
    from umrs.shared.pg_settings import pg_connect_kwargs

    assert connect_kwargs == {
        "dbname": "postgres",
        "user": "jobs-user",
        "password": "jobs-pass",
        "host": "db-host",
        "port": "15432",
        "sslmode": "disable",
        **pg_connect_kwargs("admin"),
    }
    assert pool_kwargs == {
        "minconn": 1,
        "maxconn": 32,  # 6 boot stores lease for life; 10 exhausted live (2026-09-24)
        "dbname": "jobs-db",
        "user": "jobs-user",
        "password": "jobs-pass",
        "host": "db-host",
        "port": "15432",
        "sslmode": "disable",
        **pg_connect_kwargs("durable-jobs"),
    }
    assert connect_kwargs["application_name"] == "umrs-admin"
    assert pool_kwargs["application_name"] == "umrs-durable-jobs"


class _FakeAdminCursorForDefault:
    def execute(self, _sql, params=None):
        pass

    def fetchone(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeAdminConnectionForDefault:
    def __init__(self):
        self.autocommit = False

    def cursor(self):
        return _FakeAdminCursorForDefault()

    def close(self):
        pass


def test_durable_jobs_default_db_is_umrs_db(monkeypatch):
    """G-P4-4 / ADR-034 rollout 2026-09-07: absent UMRS_JOBS_DB (and
    POSTGRES_DB), the durable-job authority resolves to the SHARED 'umrs_db'.
    The umrs-front compose pins UMRS_JOBS_DB=umrs_db explicitly so this
    default is never load-bearing — but if it changes, every cluster's jobs
    silently re-target. Pin it."""
    pool_kwargs: dict = {}
    sentinel_pool = object()

    for var in ("UMRS_JOBS_DB", "POSTGRES_DB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        job_tracking_handler.psycopg2,
        "connect",
        lambda **kw: _FakeAdminConnectionForDefault(),
    )
    monkeypatch.setattr(
        job_tracking_handler.pg_pool,
        "ThreadedConnectionPool",
        lambda **kw: (pool_kwargs.update(kw), sentinel_pool)[1],
    )

    pool = job_tracking_handler.build_durable_job_db_pool_from_env()

    assert pool is sentinel_pool
    assert pool_kwargs["dbname"] == "umrs_db"
