"""D-19/D-49 pins: placement grammar, declaration gates, ADR-061 class map."""

from __future__ import annotations

import pytest

import umrs.shared.venue_placement as vp


class _Cfg:
    def __init__(self, extra):
        self.extra = extra


def _site(monkeypatch, extra):
    monkeypatch.setattr("umrs.orchestrator.site_config.load_site_config", lambda: _Cfg(extra))


def test_local_and_frontend_always_valid():
    assert vp.validate_placement("local") == "local"
    assert vp.validate_placement("frontend") == "frontend"


def test_bad_grammar_refused():
    with pytest.raises(vp.PlacementError, match="invalid placement"):
        vp.validate_placement("basement:lab")


def test_cpu_host_needs_declaration(monkeypatch):
    _site(monkeypatch, {"venues": {"cpu_hosts": ["atom-0"]}})
    assert vp.validate_placement("cpu:atom-0") == "cpu:atom-0"
    with pytest.raises(vp.PlacementError, match="not declared"):
        vp.validate_placement("cpu:ghost")


def test_trusted_host_needs_declaration(monkeypatch):
    _site(monkeypatch, {"venues": {"trusted_hosts": ["lab-scope-1"]}})
    assert vp.validate_placement("trusted:lab-scope-1") == "trusted:lab-scope-1"
    with pytest.raises(vp.PlacementError, match="not declared"):
        vp.validate_placement("trusted:other")


def test_site_down_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "umrs.orchestrator.site_config.load_site_config",
        lambda: (_ for _ in ()).throw(RuntimeError("no site")),
    )
    with pytest.raises(vp.PlacementError):
        vp.validate_placement("trusted:lab-scope-1")


def test_trust_class_map():
    assert vp.validate_trust_class("local", "local-safe") == "local-safe"
    assert vp.validate_trust_class("trusted:lab-1", "trusted-host-only")
    with pytest.raises(vp.PlacementError, match="requires placement trusted"):
        vp.validate_trust_class("frontend", "trusted-host-only")
    with pytest.raises(vp.PlacementError, match="forbids placement"):
        vp.validate_trust_class("trusted:lab-1", "front-end")
    with pytest.raises(vp.PlacementError, match="invalid trust_class"):
        vp.validate_trust_class("local", "sketchy")


def test_register_validates_before_write(monkeypatch):
    """Store chokepoint: bad placement refuses BEFORE any SQL runs."""
    from umrs.mcp.store import MCPServerStore

    conn = _NoSqlConn()
    store = MCPServerStore(conn)
    with pytest.raises(vp.PlacementError):
        store.register_server("x", "http://h:1", placement="trusted:nope")
    assert conn.queries == []


class _NoSqlConn:
    def __init__(self):
        self.queries = []

    def cursor(self):
        self.queries.append("cursor")
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


# ── D-24 who-may-publish policy: trusted-host-only is admin-only ──────


def _register_request(trust_class, project_id="proj-a"):
    from umrs.api.mcp import ServerRegistrationRequest

    return ServerRegistrationRequest(
        name="srv-x", url="http://h:1", project_id=project_id, trust_class=trust_class
    )


class _Scope:
    def __init__(self, is_admin, project_id="proj-a", anonymous=False):
        self.is_admin = is_admin
        self.project_id = project_id
        self.anonymous = anonymous


def test_trusted_host_only_publish_requires_admin(monkeypatch):
    from fastapi import HTTPException

    from umrs.api import mcp as mcp_mod
    from umrs.api.mcp import _enforce_registration_scope

    monkeypatch.setattr(mcp_mod, "_caller_scope", lambda _r: _Scope(is_admin=False))
    with pytest.raises(HTTPException) as exc:
        _enforce_registration_scope(None, _register_request("trusted-host-only"))
    assert exc.value.status_code == 403
    assert "trust_class_denied" in str(exc.value.detail)


def test_admin_publishes_trusted_host(monkeypatch):
    from umrs.api import mcp as mcp_mod
    from umrs.api.mcp import _enforce_registration_scope

    monkeypatch.setattr(mcp_mod, "_caller_scope", lambda _r: _Scope(is_admin=True))
    req = _enforce_registration_scope(None, _register_request("trusted-host-only"))
    assert req.trust_class == "trusted-host-only"


def test_local_safe_and_front_end_stay_project_scoped(monkeypatch):
    from umrs.api import mcp as mcp_mod
    from umrs.api.mcp import _enforce_registration_scope

    monkeypatch.setattr(mcp_mod, "_caller_scope", lambda _r: _Scope(is_admin=False))
    for cls in ("local-safe", "front-end"):
        req = _enforce_registration_scope(None, _register_request(cls))
        assert req.project_id == "proj-a"  # own-project claim, no class refusal
