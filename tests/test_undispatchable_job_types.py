"""Regression tests for #216: job types without a dispatch backend must be
rejected at submission, not accepted and exploded at dispatch time."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from umrs.api.jobs import _UNDISPATCHABLE_JOB_TYPES, _enforce_dispatchable_job_type
from umrs.jobs.models import JobType
from umrs.jobs.service import JobService


def test_undispatchable_types_rejected_with_400() -> None:
    for job_type in _UNDISPATCHABLE_JOB_TYPES:
        with pytest.raises(HTTPException) as excinfo:
            _enforce_dispatchable_job_type(job_type)
        assert excinfo.value.status_code == 400
        assert "not implemented" in excinfo.value.detail.lower()


def test_implemented_types_pass_the_guard() -> None:
    for job_type in JobType:
        if job_type in _UNDISPATCHABLE_JOB_TYPES:
            continue
        _enforce_dispatchable_job_type(job_type)  # must not raise


def test_submit_embedding_batch_fails_fast() -> None:
    service = object.__new__(JobService)  # guard runs before any state touch
    with pytest.raises(NotImplementedError, match="not implemented"):
        service.submit_embedding_batch("p", "u", "/in", "/out")
