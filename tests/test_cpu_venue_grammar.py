"""D-48 pins: cpu:<host> venue grammar + declaration gate + rail refusal."""

from __future__ import annotations

import pytest
import umrs.api.companion_sessions as cs
from fastapi import HTTPException


class _Cfg:
    def __init__(self, extra):
        self.extra = extra


def _patch_site(monkeypatch, extra):
    monkeypatch.setattr("umrs.orchestrator.site_config.load_site_config", lambda: _Cfg(extra))


def test_grammar_accepts_cpu_host(monkeypatch):
    _patch_site(monkeypatch, {"venues": {"cpu_hosts": ["atom-0", "atom-1"]}})
    assert cs._validate_venue("cpu:atom-0") == "cpu:atom-0"


def test_undeclared_cpu_host_refused_by_name(monkeypatch):
    _patch_site(monkeypatch, {"venues": {"cpu_hosts": ["atom-0"]}})
    with pytest.raises(HTTPException) as exc:
        cs._validate_venue("cpu:stranger")
    assert exc.value.status_code == 400
    assert "not declared" in str(exc.value.detail)


def test_no_site_config_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "umrs.orchestrator.site_config.load_site_config",
        lambda: (_ for _ in ()).throw(RuntimeError("no site")),
    )
    with pytest.raises(HTTPException):
        cs._validate_venue("cpu:atom-0")


def test_bad_grammar_refused():
    with pytest.raises(HTTPException):
        cs._validate_venue("cpu:")
    # cluster + frontend stay valid without any declaration
    assert cs._validate_venue("frontend") == "frontend"
    assert cs._validate_venue("cluster:rccs") == "cluster:rccs"


def test_bad_request_message_names_document_section(monkeypatch):
    _patch_site(monkeypatch, {})
    with pytest.raises(HTTPException) as exc:
        cs._validate_venue("cpu:atom-0")
    assert "extra.venues.cpu_hosts" in str(exc.value.detail)


def test_frontend_rail_refuses_cpu_venued_session(monkeypatch):
    """The frontend rail never silently places a cpu-venued session."""
    from unittest.mock import MagicMock

    import umrs.api.companion_venue as cv

    class _Pool:
        def getconn(self):
            return object()

        def putconn(self, conn):
            pass

    request = MagicMock()
    request.app.state.umrs_db_pool = _Pool()
    monkeypatch.setattr(
        "umrs.api.companion_sessions.snapshot_detail_row",
        lambda _conn, _uid, _sid: {"venue": "cpu:atom-0"},
    )
    with pytest.raises(HTTPException) as exc:
        cv._refuse_cpu_venue_on_frontend_rail(request, "alice", "sess-1")
    assert exc.value.status_code == 400
    assert "frontend rail" in str(exc.value.detail)


def test_frontend_rail_allows_frontend_venued_session(monkeypatch):
    from unittest.mock import MagicMock

    import umrs.api.companion_venue as cv

    class _Pool:
        def getconn(self):
            return object()

        def putconn(self, conn):
            pass

    request = MagicMock()
    request.app.state.umrs_db_pool = _Pool()
    monkeypatch.setattr(
        "umrs.api.companion_sessions.snapshot_detail_row",
        lambda _conn, _uid, _sid: {"venue": "frontend"},
    )
    cv._refuse_cpu_venue_on_frontend_rail(request, "alice", "sess-1")  # no raise
