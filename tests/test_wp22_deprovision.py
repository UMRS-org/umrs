"""Pins for ADR-058 D-2/D-3 (WP-22 legs 2-3): deprovision outcome classes
and the honest route surface (409 cancellation-unachieved / 202 stopping /
200 gone / 404 missing)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _service_with_registry(endpoint=None, backend=None):
    from umrs.orchestrator.service import OrchestrationService

    service = OrchestrationService.__new__(OrchestrationService)
    registry = MagicMock()
    registry.get_by_job_id.return_value = endpoint
    service._registry = registry
    # `backend`/`port_manager` are read-only @property hooks; override by
    # slotting the underscore attrs the property body reads.
    service._backend = backend or MagicMock(
        cleanup=AsyncMock(),
        get_job_status=AsyncMock(return_value=MagicMock(status=None)),
    )
    service._port_manager = MagicMock()
    return service


def _endpoint(backend_job_id=None, job_id="5434"):
    ep = MagicMock()
    ep.job_id = job_id
    ep.backend_job_id = backend_job_id
    ep.port = 13062
    ep.host = "pending"
    ep.metadata = {}
    return ep


class TestOutcomeDispatch:
    @pytest.mark.asyncio
    async def test_missing_row_yields_not_found(self):
        service = _service_with_registry(endpoint=None)
        from umrs.orchestrator.service import DeprovisionOutcome

        out = await service.deprovision("nope")
        assert out is DeprovisionOutcome.NOT_FOUND

    @pytest.mark.asyncio
    async def test_identity_known_cancel_targets_backend_job_id(self):
        from umrs.orchestrator.service import JobStatus

        service = _service_with_registry(
            endpoint=_endpoint(backend_job_id="384897", job_id="5434"),
            backend=MagicMock(
                cleanup=AsyncMock(),
                get_job_status=AsyncMock(
                    return_value=MagicMock(status=JobStatus.FAILED)
                ),
            ),
        )
        from umrs.orchestrator.service import DeprovisionOutcome

        out = await service.deprovision("5434")
        # cancel must have been aimed at the backend identity, not the surrogate
        call_targets = [c.args[0] for c in service.backend.cleanup.await_args_list]
        assert call_targets == ["384897"]
        assert out is DeprovisionOutcome.GONE
        service.registry.remove.assert_called_once_with("5434")

    @pytest.mark.asyncio
    async def test_surrogate_row_reports_unachieved_even_quiet(self):
        from umrs.orchestrator.service import JobStatus

        service = _service_with_registry(
            endpoint=_endpoint(backend_job_id=None, job_id="5434"),
            backend=MagicMock(
                cleanup=AsyncMock(),
                get_job_status=AsyncMock(
                    return_value=MagicMock(status=JobStatus.FAILED)
                ),
            ),
        )
        from umrs.orchestrator.service import DeprovisionOutcome

        out = await service.deprovision("5434")
        assert out is DeprovisionOutcome.UNACHIEVED
        # registry preserved (deletion only happens on GONE)
        service.registry.remove.assert_not_called()

    @pytest.mark.asyncio
    async def test_backend_raise_is_unachieved(self):
        service = _service_with_registry(
            endpoint=_endpoint(backend_job_id="384897", job_id="5434"),
            backend=MagicMock(cleanup=AsyncMock(side_effect=RuntimeError("boom"))),
        )
        from umrs.orchestrator.service import DeprovisionOutcome

        out = await service.deprovision("5434")
        assert out is DeprovisionOutcome.UNACHIEVED

    @pytest.mark.asyncio
    async def test_settling_backend_reports_stopping_with_identity(self):
        from umrs.orchestrator.service import JobStatus

        service = _service_with_registry(
            endpoint=_endpoint(backend_job_id="384897", job_id="5434"),
            backend=MagicMock(
                cleanup=AsyncMock(),
                get_job_status=AsyncMock(
                    return_value=MagicMock(status=JobStatus.RUNNING)
                ),
            ),
        )
        from umrs.orchestrator.service import DeprovisionOutcome

        out = await service.deprovision("5434")
        assert out is DeprovisionOutcome.STOPPING
        service.registry.remove.assert_not_called()

    @pytest.mark.asyncio
    async def test_surrogate_settling_collapses_to_unachieved(self):
        from umrs.orchestrator.service import JobStatus

        service = _service_with_registry(
            endpoint=_endpoint(backend_job_id=None, job_id="5434"),
            backend=MagicMock(
                cleanup=AsyncMock(),
                get_job_status=AsyncMock(
                    return_value=MagicMock(status=JobStatus.RUNNING)
                ),
            ),
        )
        from umrs.orchestrator.service import DeprovisionOutcome

        out = await service.deprovision("5434")
        assert out is DeprovisionOutcome.UNACHIEVED


class ServiceReturns:
    """Script the outcome through the deprovision route."""

    def __init__(self, outcome, endpoint=None):
        self.outcome = outcome
        self.endpoint = endpoint

    async def deprovision(self, job_id, force=False):  # noqa: ARG002 - stub mimics service signature
        return self.outcome


def _route_client(outcome_name, endpoint=None, monkeypatch=None):
    from umrs.api import endpoints as endpoints_mod  # noqa: F401 (kept for symmetry)
    from umrs.api.endpoints import router
    from umrs.orchestrator.endpoint_registry import get_registry
    from umrs.orchestrator.service import DeprovisionOutcome

    app = FastAPI()
    app.include_router(router)

    registry = MagicMock()
    registry.get_by_job_id.return_value = endpoint

    outcome = DeprovisionOutcome(outcome_name)
    service = ServiceReturns(outcome, endpoint=endpoint)

    app.dependency_overrides[get_registry] = lambda: registry
    # `endpoints` lazily binds `get_orchestration_service` inside the handler —
    # patch the attribute object in sys.modules before the request lands.
    import umrs.orchestrator.service as _svc_mod

    monkeypatch.setattr(_svc_mod, "get_orchestration_service", lambda: service)

    client = TestClient(app)
    return client


def _ep_with(backend_job_id, job_id="5434"):
    from umrs.orchestrator.endpoint_registry import EndpointStatus

    ep = MagicMock()
    ep.job_id = job_id
    ep.backend_job_id = backend_job_id
    ep.metadata = {"tier": "user", "owner_id": "alice"}
    ep.status = EndpointStatus.PROVISIONING
    ep.project_id = "default"
    return ep


class TestRouteSurface:
    def test_unachieved_is_409_typed(self, monkeypatch):
        ep = _ep_with(None)
        client = _route_client("cancellation-unachieved", endpoint=ep, monkeypatch=monkeypatch)
        resp = client.request(
            "DELETE",
            "/umrs/endpoints/5434/deprovision",
            headers={"x-user-id": "alice"},
        )
        assert resp.status_code == 409
        body = resp.json()
        detail = body.get("detail") or body.get("error", {}).get("message", "")
        assert "could not terminate the backend job" in detail

    def test_gone_is_200_with_audit(self, monkeypatch):
        ep = _ep_with("384897")
        ep.metadata["cleanup_status"] = ""
        client = _route_client("gone", endpoint=ep, monkeypatch=monkeypatch)
        resp = client.request(
            "DELETE",
            "/umrs/endpoints/5434/deprovision",
            headers={"x-user-id": "alice"},
        )
        assert resp.status_code in (200, 204)

    def test_stopping_is_202(self, monkeypatch):
        ep = _ep_with("384897")
        ep.metadata["cleanup_status"] = "COMPLETING"
        client = _route_client("stopping", endpoint=ep, monkeypatch=monkeypatch)
        resp = client.request(
            "DELETE",
            "/umrs/endpoints/5434/deprovision",
            headers={"x-user-id": "alice"},
        )
        assert resp.status_code == 202
        assert resp.json()["status"] == "stopping"


class TestLineageCancelThrough:
    def test_unachieved_cancels_durable_lineage(self, monkeypatch):
        ep = _ep_with(None)
        ep.metadata["durable_job_id"] = "job-lineage-1"
        store = MagicMock()
        # _cancel_lineage_best_effort lazily binds `from .jobs import get_job_store`
        import umrs.api.jobs as jobs_mod

        monkeypatch.setattr(jobs_mod, "get_job_store", lambda: store)
        client = _route_client(
            "cancellation-unachieved", endpoint=ep, monkeypatch=monkeypatch
        )
        resp = client.request(
            "DELETE",
            "/umrs/endpoints/5434/deprovision",
            headers={"x-user-id": "alice"},
        )
        assert resp.status_code == 409
        store.cancel_job.assert_called_once_with(
            "job-lineage-1", reason="endpoint_deprovision:5434"
        )

    def test_unachieved_without_lineage_is_clean_409(self, monkeypatch):
        ep = _ep_with(None)
        client = _route_client(
            "cancellation-unachieved", endpoint=ep, monkeypatch=monkeypatch
        )
        resp = client.request(
            "DELETE",
            "/umrs/endpoints/5434/deprovision",
            headers={"x-user-id": "alice"},
        )
        assert resp.status_code == 409
        assert "(no backend identity assigned yet" in (
            resp.json().get("detail") or resp.json().get("error", {}).get("message", "")
        )

    def test_lineage_cancel_failure_never_masks_409(self, monkeypatch):
        ep = _ep_with(None)
        ep.metadata["durable_job_id"] = "job-lineage-x"
        store = MagicMock()
        store.cancel_job.side_effect = RuntimeError("invalid transition")
        import umrs.api.jobs as jobs_mod

        monkeypatch.setattr(jobs_mod, "get_job_store", lambda: store)
        client = _route_client(
            "cancellation-unachieved", endpoint=ep, monkeypatch=monkeypatch
        )
        resp = client.request(
            "DELETE",
            "/umrs/endpoints/5434/deprovision",
            headers={"x-user-id": "alice"},
        )
        assert resp.status_code == 409


class TestJanitorLineageReaper:
    def test_cancel_durable_lineage_fires_when_metadata_present(self, monkeypatch):
        from umrs.orchestrator.janitor import _cancel_durable_lineage

        ep = MagicMock()
        ep.job_id = "5434"
        ep.metadata = {"durable_job_id": "job-x"}
        store = MagicMock()
        monkeypatch.setattr("umrs.api.jobs.get_job_store", lambda: store)
        _cancel_durable_lineage(ep)
        store.cancel_job.assert_called_once_with(
            "job-x", reason="janitor_stuck_reaper:5434"
        )

    def test_cancel_durable_lineage_skips_when_no_metadata(self, monkeypatch):
        from umrs.orchestrator.janitor import _cancel_durable_lineage

        ep = MagicMock()
        ep.job_id = "5434"
        ep.metadata = {}
        store = MagicMock()
        monkeypatch.setattr("umrs.api.jobs.get_job_store", lambda: store)
        _cancel_durable_lineage(ep)
        store.cancel_job.assert_not_called()

    def test_lineage_cancel_failure_is_swallowed_with_evidence(self, monkeypatch, caplog):
        from umrs.orchestrator.janitor import _cancel_durable_lineage

        ep = MagicMock()
        ep.job_id = "5434"
        ep.metadata = {"durable_job_id": "job-x"}
        store = MagicMock()
        store.cancel_job.side_effect = RuntimeError("invalid transition")
        monkeypatch.setattr("umrs.api.jobs.get_job_store", lambda: store)
        with caplog.at_level("WARNING", logger="umrs.orchestrator.janitor"):
            _cancel_durable_lineage(ep)
        assert any(
            "lineage cancel FAILED" in rec.getMessage()
            and "durable_job=job-x" in rec.getMessage()
            for rec in caplog.records
        )
