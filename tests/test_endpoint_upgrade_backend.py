"""Phase 1 smoke tests for ENDPOINT_UPGRADE job-type plumbing (Q-13).

The endpoint upgrade backend commits replacement-provisioning-flip-retire
via the existing endpoint infrastructure, but Phase 1 ships ONLY the
structural skeleton (JobType + factory branch + package import). Full
flip orchestration lands in the follow-up wave alongside rollback-on-
health-fail and the concurrency gate.
"""

from __future__ import annotations

import asyncio

import pytest
from umrs.jobs.models import DEFAULT_BACKOFFS, JobType
from umrs.jobs.upgrade import EndpointUpgradeBackend


class TestJobType:
    def test_endpoint_upgrade_job_type_exists(self) -> None:
        assert JobType.ENDPOINT_UPGRADE in JobType
        assert JobType.ENDPOINT_UPGRADE.value == "endpoint_upgrade"

    def test_endpooint_upgrade_backoff_conservative(self) -> None:
        backoff = DEFAULT_BACKOFFS[JobType.ENDPOINT_UPGRADE]
        assert backoff["max_attempts"] == 2
        assert backoff["delays"] == [10, 30, 60]

    def test_job_type_does_not_shadow_endpoint_provision(self) -> None:
        # ENDPOINT_UPGRADE must not be a silent alias of ENDPOINT_PROVISION.
        assert JobType.ENDPOINT_UPGRADE != JobType.ENDPOINT_PROVISION


class TestEndpointUpgradeBackendStub:
    def test_backend_job_type_binding(self) -> None:
        backend = EndpointUpgradeBackend(
            orchestrator=None, registry=None, port_manager=None
        )
        assert backend.job_type == JobType.ENDPOINT_UPGRADE

    def test_submit_job_raises_not_implemented_until_flip_orchestration(self) -> None:
        backend = EndpointUpgradeBackend(
            orchestrator=None, registry=None, port_manager=None
        )
        with pytest.raises(NotImplementedError, match="flip orchestration"):
            asyncio.run(backend.submit_job(job=None, attempt_id="a-1"))

    def test_check_status_raises_not_implemented(self) -> None:
        backend = EndpointUpgradeBackend(
            orchestrator=None, registry=None, port_manager=None
        )
        with pytest.raises(NotImplementedError):
            asyncio.run(backend.check_status(job=None, attempt_id="a-1"))

    def test_cancel_job_idempotent_in_phase_one(self) -> None:
        backend = EndpointUpgradeBackend(
            orchestrator=None, registry=None, port_manager=None
        )
        result = asyncio.run(backend.cancel_job(job=None, attempt_id="a-1"))
        assert result is True
