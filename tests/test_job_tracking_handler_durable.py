import json
from datetime import UTC, datetime

from umrs.jobs.models import Job, JobAttempt, JobStatus, JobType, RestartPolicy, AttemptStatus
from umrs.server.job_tracking_handler import DurableJobCommandBackend, handle_job_command

JOB_SUBMIT = "JOB_SUBMIT"
JOB_LIST = "JOB_LIST"


class FakeJobStore:
    def __init__(self):
        self.jobs = {}
        self.attempts = {}
        self.enqueued = []
        self.cancelled = []

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
        job_id = f"job-{len(self.jobs) + 1:03d}"
        job = Job(
            job_id=job_id,
            project_id=project_id,
            user_id=user_id,
            job_type=job_type,
            spec=spec,
            idempotency_key=idempotency_key or f"idem-{job_id}",
            status=JobStatus.PLANNED,
            priority=priority,
            desired_max_attempts=max_attempts,
            restart_policy=restart_policy,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        self.jobs[job_id] = job
        self.attempts[job_id] = []
        return job_id

    def enqueue_job(self, job_id):
        self.enqueued.append(job_id)
        self.jobs[job_id].status = JobStatus.QUEUED

    def get_job(self, job_id):
        return self.jobs[job_id]

    def get_attempts_for_job(self, job_id):
        return self.attempts[job_id]

    def list_jobs(self, project_id=None, status=None, job_type=None, limit=100, offset=0):
        jobs = list(self.jobs.values())
        if project_id:
            jobs = [job for job in jobs if job.project_id == project_id]
        if status:
            jobs = [job for job in jobs if job.status == status]
        if job_type:
            jobs = [job for job in jobs if job.job_type == job_type]
        return jobs[offset : offset + limit]

    def cancel_job(self, job_id, reason="user_cancelled"):
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
            started_at=datetime.now(UTC),
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