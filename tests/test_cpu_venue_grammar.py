"""D-48 pins: cpu:<host> venue grammar + declaration gate + the start's pin."""

from __future__ import annotations

import pytest
import umrs_r4.api.companion_sessions as cs
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


# ── the start's pin (ADR-098 D-9, SH-H2; was a 400 "frontend rail" refusal) ──


class _Pool:
    def getconn(self):
        return object()

    def putconn(self, conn):
        pass


def _pin_of(monkeypatch, row_fn):
    from unittest.mock import MagicMock

    import umrs_r4.api.companion_venue as cv

    request = MagicMock()
    request.app.state.umrs_db_pool = _Pool()
    monkeypatch.setattr("umrs_r4.api.companion_sessions.snapshot_detail_row", row_fn)
    return cv._session_pin(request, "alice", "sess-1")


def test_cpu_venued_session_start_carries_its_pin(monkeypatch):
    """A cpu-venued session's start pins that host (placement then runs it
    there or refuses it by name; never a silent frontend placement)."""
    pin = _pin_of(monkeypatch, lambda _c, _u, _s: {"venue": "cpu:atom-0"})
    assert pin == "atom-0"


def test_frontend_venued_session_has_no_pin(monkeypatch):
    assert _pin_of(monkeypatch, lambda _c, _u, _s: {"venue": "frontend"}) == ""
    assert _pin_of(monkeypatch, lambda _c, _u, _s: None) == ""


def test_malformed_cpu_venue_refused_not_dropped(monkeypatch):
    with pytest.raises(HTTPException) as exc:
        _pin_of(monkeypatch, lambda _c, _u, _s: {"venue": "cpu:"})
    assert exc.value.status_code == 400


def test_unreadable_venue_refuses_the_start(monkeypatch):
    """A failed lookup used to read as "no venue" (a cpu pin silently became
    a frontend start); it now refuses the start."""

    def boom(_c, _u, _s):
        raise RuntimeError("db down")

    with pytest.raises(HTTPException) as exc:
        _pin_of(monkeypatch, boom)
    assert exc.value.status_code == 503
