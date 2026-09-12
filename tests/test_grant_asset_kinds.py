"""WP-06 r2 leg-b pins: endpoint/workflow asset kinds + writable checks (fail-without-change)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from umrs.api.grants import (
    _asset_writable,
    _endpoint_writable,
    _workflow_entry,
    _workflow_writable,
)
from umrs.grants.store import ASSET_KINDS


def _caller(uid: str = "", pid: str = "proj-9", is_admin: bool = False) -> SimpleNamespace:
    return SimpleNamespace(user_id=uid, project_id=pid, is_admin=is_admin)


class TestAssetKinds:
    def test_endpoint_and_workflow_registered_but_bundle_not(self):
        assert "endpoint" in ASSET_KINDS
        assert "workflow" in ASSET_KINDS
        assert "bundle" not in ASSET_KINDS


class TestEndpointWritability:
    def _registry(self, ep):
        registry = MagicMock()

        def list_all(*, project_id=None, status_filter=None):
            # registry's list_all(project_id=...) filters by pid — route
            # caller-pid rows for the caller-pidquery, nothing for other pids
            return [e for e in [ep] if not project_id or e.project_id == project_id]

        registry.list_all = list_all
        return registry

    def test_project_match_wins(self):
        ep = MagicMock()
        ep.job_id = "job-77"
        ep.project_id = "proj-9"
        ep.metadata = {}
        registry = self._registry(ep)
        with patch("umrs.orchestrator.endpoint_registry.get_registry", return_value=registry):
            assert _endpoint_writable("job-77", _caller(pid="proj-9")) is True
            assert _endpoint_writable("job-77", _caller(pid="proj-x")) is False

    def test_owner_id_beats_project_match(self):
        ep = MagicMock()
        ep.job_id = "job-77"
        ep.project_id = "proj-9"
        ep.metadata = {"owner_id": "u-ally"}
        registry = self._registry(ep)
        with patch("umrs.orchestrator.endpoint_registry.get_registry", return_value=registry):
            assert _endpoint_writable("job-77", _caller(uid="u-ally", pid="proj-9")) is True
            assert _endpoint_writable("job-77", _caller(uid="u-other", pid="proj-9")) is False

    def test_unknown_endpoint_returns_false(self):
        registry = MagicMock()
        registry.list_all.return_value = []
        with patch("umrs.orchestrator.endpoint_registry.get_registry", return_value=registry):
            assert _endpoint_writable("maybe", _caller()) is False


class TestWorkflowWritability:
    def _catalog(self, entries):
        return {"workflows": entries}

    def test_owner_wins(self):
        episodes = self._catalog(
            [{"name": "wf-1", "user_id": "u-ally", "scope": "user"}]
        )
        with patch("umrs.server.workflow_registry_handler._load_catalog", return_value=episodes):
            assert _workflow_writable("wf-1", _caller(uid="u-ally")) is True
            assert _workflow_writable("wf-1", _caller(uid="u-other")) is False

    def test_system_scope_never_user_writable(self):
        episodes = self._catalog([{"name": "wf-sys", "user_id": "u-ally", "scope": "system"}])
        with patch("umrs.server.workflow_registry_handler._load_catalog", return_value=episodes):
            # system is admin-owned — CSS-only never slips through owner path
            assert _workflow_writable("wf-sys", _caller(uid="u-ally")) is False
            assert _workflow_writable("wf-sys", _caller(uid="u-ally", is_admin=True)) is False

    def test_admin_wins_anywhere_via_asset_ladder(self):
        # ladder semantics: admin + system → the BasePath gate passes it through
        episodes = self._catalog([{"name": "wf-sys", "scope": "system", "user_id": "u-sys"}])
        with patch("umrs.server.workflow_registry_handler._load_catalog", return_value=episodes):
            assert _asset_writable(None, "workflow", "wf-sys", _caller(is_admin=True)) is True

    def test_unknown_workflow_returns_false(self):
        with patch("umrs.server.workflow_registry_handler._load_catalog", return_value=self._catalog([])):
            assert _workflow_entry("missing") is None
            assert _workflow_writable("missing", _caller(uid="u-x")) is False


class TestLadderWiring:
    def test_endpoint_and_workflow_kinds_routed_through_asset_ladder(self):
        pool = object()
        with (
            patch("umrs.api.grants._endpoint_writable", return_value=True) as mock_ep,
            patch("umrs.api.grants._workflow_writable", return_value=True) as mock_wf,
        ):
            assert _asset_writable(pool, "endpoint", "job-1", _caller()) is True
            assert _asset_writable(pool, "workflow", "wf-1", _caller()) is True
            mock_ep.assert_called_once_with("job-1", _caller())
            mock_wf.assert_called_once_with("wf-1", _caller())

    def test_unknown_kind_keeps_false(self):
        assert _asset_writable(None, "phantom", "x", _caller()) is False
