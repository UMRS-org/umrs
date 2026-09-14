from types import SimpleNamespace

import pytest
from umrs.server.durable_job_runtime import DurableJobRuntime, DurableJobRuntimeError


class FakePool:
    def __init__(self):
        self.closed = False

    def closeall(self):
        self.closed = True


class FakeService:
    def __init__(self):
        self.job_service = SimpleNamespace(job_store=object())
        self.startup_calls = 0
        self.shutdown_calls = 0

    async def startup(self):
        self.startup_calls += 1

    async def shutdown(self, timeout: float = 10.0):
        self.shutdown_calls += 1


class FailingService(FakeService):
    async def startup(self):
        raise RuntimeError("boom")


def test_runtime_start_and_shutdown_owns_async_service():
    pool = FakePool()
    service = FakeService()
    runtime = DurableJobRuntime(db_pool=pool, service=service)

    backend = runtime.start(timeout=5.0)

    assert backend is runtime.command_backend
    assert service.startup_calls == 1

    runtime.shutdown(timeout=5.0)

    assert service.shutdown_calls == 1
    assert pool.closed is True


def test_runtime_start_surfaces_startup_failure_and_closes_pool():
    pool = FakePool()
    runtime = DurableJobRuntime(db_pool=pool, service=FailingService())

    with pytest.raises(DurableJobRuntimeError, match="Failed to start durable job runtime"):
        runtime.start(timeout=5.0)

    assert pool.closed is True
