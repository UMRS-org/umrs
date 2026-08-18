"""CompanionWorker venue bookkeeping backend tests (venue FK seam)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from umrs.jobs.companion_worker_backend import (
    CompanionWorkerBackend,
    CompanionWorkerStatusChecker,
    validate_companion_worker_spec,
)
from umrs.jobs.models import JobType


def _job(spec: dict):
    return SimpleNamespace(job_id="j-1", spec=spec)


def _attempt(ref: str | None = None):
    return SimpleNamespace(attempt_id="a-1", backend_ref=ref)


def _sync(coro):
    return asyncio.run(coro)


def test_spec_validation_requires_registry_key():
    assert validate_companion_worker_spec({"registry_key": "alice@default"}) == {
        "registry_key": "alice@default"
    }
    for bad in (None, {}, {"registry_key": " "}, "not-a-dict"):
        with pytest.raises(ValueError):
            validate_companion_worker_spec(bad)


def test_backend_submit_returns_typed_ref():
    ref = _sync(
        CompanionWorkerBackend().submit_job(_job({"registry_key": "alice@default"}), _attempt())
    )
    assert ref == "companion_worker|registry_key=alice@default"


def test_status_checker_succeeds_and_fails_cleanly():
    checker = CompanionWorkerStatusChecker()
    assert checker.job_type == JobType.COMPANION_WORKER
    ok = _sync(
        checker.check_status(_attempt("companion_worker|registry_key=alice@default"), _job({}))
    )
    assert ok.status == "succeeded"
    assert ok.metrics["registry_key"] == "alice@default"
    missing = _sync(checker.check_status(_attempt(None), _job({})))
    assert missing.status == "failed"
    malformed = _sync(checker.check_status(_attempt("custom|foo=bar"), _job({})))
    assert malformed.status == "failed"


def test_dispatch_wiring_registered():
    from umrs.server.job_tracking_handler import _JOB_SPEC_VALIDATORS

    assert JobType.COMPANION_WORKER in _JOB_SPEC_VALIDATORS
